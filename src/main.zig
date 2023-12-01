const std = @import("std");

const Config = struct {
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length
};

const Weights = struct {
    // Token embedding table
    token_embedding_table: []align(1) const f32, // (vocab_size, dim)
    // Weights for rmsnorms
    rms_att_weight: []align(1) const f32, // (layer, dim) rmsnorm weights
    rms_ffn_weight: []align(1) const f32, // (layer, dim)
    // Weights for matmuls. note dim == n_heads * head_size
    wq: []align(1) const f32, // (layer, dim, n_heads * head_size)
    wk: []align(1) const f32, // (layer, dim, n_kv_heads * head_size)
    wv: []align(1) const f32, // (layer, dim, n_kv_heads * head_size)
    wo: []align(1) const f32, // (layer, n_heads * head_size, dim)
    // Weights for ffn
    w1: []align(1) const f32, // (layer, hidden_dim, dim)
    w2: []align(1) const f32, // (layer, dim, hidden_dim)
    w3: []align(1) const f32, // (layer, hidden_dim, dim)
    // Final rmsnorm
    rms_final_weight: []align(1) const f32, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: []align(1) const f32,
};

const Transformer = struct {
    config: Config,
    fd: std.os.fd_t,
    mapped_buffer: []align(std.mem.page_size) const u8,
    weights: Weights,
};

/// This function returns a slice of f32s from the source u8 slice. It also increments the start
/// address.
fn slice_and_increment(
    src: []align(std.mem.page_size) const u8,
    start: *usize,
    num_f32s: usize,
) []align(1) const f32 {
    const end: usize = start.* + @sizeOf(f32) * num_f32s;
    const out_slice = std.mem.bytesAsSlice(f32, src[(start.*)..end]);
    start.* = end;
    return out_slice;
}

fn transformerInit(checkpoint_path: []const u8) !Transformer {
    var file_size: u64 = 0;
    var shared_weights: bool = false;
    var config: Config = undefined;
    {
        const file = try std.fs.cwd().openFile(checkpoint_path, .{});
        defer file.close();

        // Read in the config header
        const config_slice = std.mem.asBytes(&config);
        _ = try file.read(config_slice);
        // Negative vocab size is hacky way of signaling unshared weights. Bit yikes.
        shared_weights = config.vocab_size > 0;
        config.vocab_size = try std.math.absInt(config.vocab_size);
        // Figure out the file size
        file_size = try file.getEndPos();
    }
    // Memory map the Transformer weights into the data pointer
    var fd: std.os.fd_t = undefined;
    var mapped_buffer: []align(std.mem.page_size) const u8 = undefined;
    {
        fd = try std.os.open(checkpoint_path, 0, 0);
        // TODO: This relies on Linux-specific mmap flags. Need to make this portable.
        mapped_buffer = try std.os.mmap(null, file_size, std.os.linux.PROT.READ, std.os.linux.MAP.PRIVATE, fd, 0);
    }
    // Obtain slices into the mapped buffer for each of the weights
    var weights: Weights = undefined;
    {
        std.debug.assert(config.dim > 0 and config.n_heads > 0 and config.n_layers > 0 and config.vocab_size > 0 and config.n_heads > 0);
        const n_layers: usize = @intCast(config.n_layers);
        const dim: usize = @intCast(config.dim);
        const hidden_dim: usize = @intCast(config.hidden_dim);
        const vocab_size: usize = @intCast(config.vocab_size);
        const n_heads: usize = @intCast(config.n_heads);
        const n_kv_heads: usize = @intCast(config.n_kv_heads);
        const seq_len: usize = @intCast(config.seq_len);
        const head_size: usize = @divTrunc(dim, n_heads);

        var start: usize = @sizeOf(Config);
        weights.token_embedding_table = slice_and_increment(mapped_buffer, &start, vocab_size * dim);
        weights.rms_att_weight = slice_and_increment(mapped_buffer, &start, n_layers * dim);
        weights.wq = slice_and_increment(mapped_buffer, &start, n_layers * dim * (n_heads * head_size));
        weights.wk = slice_and_increment(mapped_buffer, &start, n_layers * dim * (n_kv_heads * head_size));
        weights.wv = slice_and_increment(mapped_buffer, &start, n_layers * dim * (n_kv_heads * head_size));
        weights.wo = slice_and_increment(mapped_buffer, &start, n_layers * dim * (n_heads * head_size));
        weights.rms_ffn_weight = slice_and_increment(mapped_buffer, &start, n_layers * dim);
        weights.w1 = slice_and_increment(mapped_buffer, &start, n_layers * dim * hidden_dim);
        weights.w2 = slice_and_increment(mapped_buffer, &start, n_layers * dim * hidden_dim);
        weights.w3 = slice_and_increment(mapped_buffer, &start, n_layers * dim * hidden_dim);
        weights.rms_final_weight = slice_and_increment(mapped_buffer, &start, dim);
        if (shared_weights) {
            weights.wcls = weights.token_embedding_table;
        } else {
            start += seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
            start += seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
            weights.wcls = slice_and_increment(mapped_buffer, &start, vocab_size * dim);
        }
    }

    return .{
        .config = config,
        .fd = fd,
        .mapped_buffer = mapped_buffer,
        .weights = weights,
    };
}

fn transformerFree(transformer: Transformer) void {
    std.os.close(transformer.fd);
    std.os.munmap(transformer.mapped_buffer);
}

fn errorUsage() !void {
    const stderr = std.io.getStdErr().writer();
    const usage =
        \\Usage:   run <checkpoint> [options]
        \\Example: run model.bin -n 256 -i \"Once upon a time\"
        \\Options:\n");
        \\  -t <float>  temperature in [0,inf], default 1.0
        \\  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
        \\  -s <int>    random seed, default time(NULL)
        \\  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
        \\  -i <string> input prompt
        \\  -z <string> optional path to custom tokenizer
        \\  -m <string> mode: generate|chat, default: generate
        \\  -y <string> (optional) system prompt in chat mode
    ;
    try stderr.print(usage, .{});
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = general_purpose_allocator.allocator();

    var checkpoint_path: ?[]const u8 = null;
    // Parse command line arguments
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    if (args.len < 2) {
        try errorUsage();
        std.os.exit(1);
    }
    checkpoint_path = args[1];

    // TODO: more command line argument validation

    const transformer = try transformerInit(checkpoint_path.?);
    defer transformerFree(transformer);
}
