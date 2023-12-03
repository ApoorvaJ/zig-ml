const std = @import("std");
const Transformer = @import("Transformer.zig");
const Tokenizer = @import("Tokenizer.zig");
const Sampler = @import("Sampler.zig");

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

fn generate(
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
    prompt: [:0]const u8,
    step: i32,
    allocator: std.mem.Allocator,
) !void {
    _ = step;
    _ = sampler;
    _ = transformer;
    // encode the (string) prompt into tokens sequence
    var prompt_tokens = try allocator.alloc(u32, prompt.len + 3); // +3 for '\0', ?BOS, ?EOS
    defer allocator.free(prompt_tokens);

    tokenizer.encode(prompt, prompt_tokens);
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

    const transformer = try Transformer.init(checkpoint_path.?, gpa);
    defer transformer.free(gpa);
    const tokenizer = try Tokenizer.init("tokenizer.bin", gpa, @intCast(transformer.config.vocab_size));
    defer tokenizer.free(gpa);
    const sampler = try Sampler.init(gpa, @intCast(transformer.config.vocab_size), 1.0, 0.9, 0);
    defer sampler.free(gpa);

    try generate(transformer, tokenizer, sampler, "Once upon a time", 256, gpa);
}
