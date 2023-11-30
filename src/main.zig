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

const Transformer = struct {
    config: Config,
    fd: std.os.fd_t,
};

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
    {
        fd = try std.os.open(checkpoint_path, 0, 0);
        // TODO: This relies on Linux-specific mmap flags. Need to make this portable.
        const data = try std.os.mmap(null, file_size, std.os.linux.PROT.READ, std.os.linux.MAP.PRIVATE, fd, 0);
        _ = data;
    }

    return .{
        .config = config,
        .fd = fd,
    };
}

fn transformerFree(transformer: Transformer) void {
    _ = std.os.close(transformer.fd);
    // TODO: Unamp the memory
    // std.os.munmap(data);
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
    transformerFree(transformer);
}
