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
    // (Optional) classifier weights for the logits, on the last layer
    wcls: []align(1) const f32,
};

const RunState = struct {
    // Current wave of activations
    x: []f32, // activation at current time stamp (dim,)
    xb: []f32, // same, but inside a residual branch (dim,)
    xb2: []f32, // an additional buffer just for convenience (dim,)
    hb: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []f32, // query (dim,)
    k: []f32, // key (dim,)
    v: []f32, // value (dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits
    // Key-value cache
    key_cache: []f32, // (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)
};

config: Config,
fd: std.os.fd_t,
mapped_buffer: []align(std.mem.page_size) const u8,
weights: Weights,
run_state: RunState,

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

pub fn init(checkpoint_path: []const u8, allocator: std.mem.Allocator) !@This() {
    var file_size: u64 = 0;
    var shared_weights: bool = false;
    var config: Config = undefined;
    // Read the config from the file
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

    const n_layers: usize = @intCast(config.n_layers);
    const dim: usize = @intCast(config.dim);
    const hidden_dim: usize = @intCast(config.hidden_dim);
    const vocab_size: usize = @intCast(config.vocab_size);
    const n_heads: usize = @intCast(config.n_heads);
    const n_kv_heads: usize = @intCast(config.n_kv_heads);
    const seq_len: usize = @intCast(config.seq_len);

    // Obtain slices into the mapped buffer for each of the weights
    var weights: Weights = undefined;
    {
        std.debug.assert(config.dim > 0 and config.n_heads > 0 and config.n_layers > 0 and config.vocab_size > 0 and config.n_heads > 0);
        const head_size: usize = dim / n_heads;

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
    // Heap-allocate the run state
    var run_state: RunState = undefined;
    {
        const kv_dim: usize = (dim * n_kv_heads) / n_heads;

        run_state.x = try allocator.alloc(f32, dim);
        run_state.xb = try allocator.alloc(f32, dim);
        run_state.xb2 = try allocator.alloc(f32, dim);
        run_state.hb = try allocator.alloc(f32, hidden_dim);
        run_state.hb2 = try allocator.alloc(f32, hidden_dim);
        run_state.q = try allocator.alloc(f32, dim);
        run_state.key_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim);
        run_state.value_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim);
        run_state.att = try allocator.alloc(f32, n_heads * seq_len);
        run_state.logits = try allocator.alloc(f32, vocab_size);
    }

    return .{
        .config = config,
        .fd = fd,
        .mapped_buffer = mapped_buffer,
        .weights = weights,
        .run_state = run_state,
    };
}

pub fn free(self: @This(), allocator: std.mem.Allocator) void {
    std.os.close(self.fd);
    std.os.munmap(self.mapped_buffer);
    allocator.free(self.run_state.x);
    allocator.free(self.run_state.xb);
    allocator.free(self.run_state.xb2);
    allocator.free(self.run_state.hb);
    allocator.free(self.run_state.hb2);
    allocator.free(self.run_state.q);
    allocator.free(self.run_state.key_cache);
    allocator.free(self.run_state.value_cache);
    allocator.free(self.run_state.att);
    allocator.free(self.run_state.logits);
}

fn rmsnorm(o: []f32, x: []f32, weight: []align(1) const f32) void {
    std.debug.assert(o.len == x.len and x.len == weight.len);
    // Calculate sum of squares
    var ss: f32 = 0.0;
    for (0..x.len) |i| {
        ss += x[i] * x[i];
    }
    ss /= @floatFromInt(x.len);
    ss += 1e-5;
    ss = 1.0 / @sqrt(ss);
    // normalize and scale
    for (0..x.len) |i| {
        o[i] = weight[i] * (ss * x[i]);
    }
}

pub fn forward(self: @This(), token: u32, pos: u32) []f32 {
    _ = pos;

    const dim: usize = @intCast(self.config.dim);
    const n_layers: usize = @intCast(self.config.n_layers);
    const n_kv_heads: usize = @intCast(self.config.n_kv_heads);
    const seq_len: usize = @intCast(self.config.seq_len);
    const kv_dim: usize = (dim * n_kv_heads) / n_heads;

    // Copy the token embedding into x
    @memcpy(self.run_state.x, self.weights.token_embedding_table[token * dim .. token * dim + dim]);

    // Forward all the layers
    for (0..n_layers) |l| {
        // Attention rmsnorm
        rmsnorm(self.run_state.xb, self.run_state.x, self.weights.rms_att_weight[l * dim .. l * dim + dim]);
        // Key and value point to the kv cache
        const loff: usize = l * seq_len * kv_dim;
        self.run_state.k = self.run_state.key_cache[loff + pos * kv_dim .. loff + pos * kv_dim + dim];
    }

    return &.{};
}
