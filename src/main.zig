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
    transformer: *Transformer,
    tokenizer: Tokenizer,
    sampler: *Sampler,
    prompt: []const u8,
    steps: u32,
    allocator: std.mem.Allocator,
) !void {
    // Encode the (string) prompt into tokens sequence
    var prompt_token_buffer = try allocator.alloc(u32, prompt.len + 3); // +3 for '\0', ?BOS, ?EOS
    defer allocator.free(prompt_token_buffer);

    var num_prompt_tokens = try tokenizer.encode(prompt, prompt_token_buffer, allocator);
    const prompt_tokens: []u32 = prompt_token_buffer[0..num_prompt_tokens];
    std.debug.assert(num_prompt_tokens >= 1);

    // Start the main loop
    var pos: u32 = 0;
    var next: u32 = 0; // Will store the next token in the sequence
    var token: u32 = prompt_tokens[0];
    while (pos < steps) {
        // Forward the transformer to get logits for the next token.
        var logits = transformer.forward(token, pos);

        // Advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // If we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // Otherwise sample the next token from the logits
            next = sampler.sample(logits);
        }
        pos += 1;

        // Data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) {
            break;
        }

        // Print the token as string, decode it with the Tokenizer object
        const piece: []const u8 = try tokenizer.decode(token, next);
        // piece might be a raw byte token, and we only want to print printable chars or whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        const is_printable_byte: bool = piece.len == 1 and piece[0] >= 32 and piece[0] <= 126 and piece[0] != ' ' and piece[0] != '\n' and piece[0] != '\r' and piece[0] != '\t';
        if (piece.len > 1 or is_printable_byte) {
            std.debug.print("{s}", .{piece});
        }
        token = next;
    }
    std.debug.print("\n", .{});
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

    var transformer = try Transformer.init(checkpoint_path.?, gpa);
    defer transformer.free(gpa);
    const tokenizer = try Tokenizer.init("tokenizer.bin", gpa, @intCast(transformer.config.vocab_size));
    defer tokenizer.free(gpa);
    var sampler = try Sampler.init(gpa, @intCast(transformer.config.vocab_size), 1.0, 0.9, 0);
    defer sampler.free(gpa);

    try generate(&transformer, tokenizer, &sampler, "Once upon a time", 256, gpa);
}
