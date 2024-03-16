const std = @import("std");
const Transformer = @import("Transformer.zig");
const Tokenizer = @import("Tokenizer.zig");
const Sampler = @import("Sampler.zig");

fn errorUsage() !void {
    const stderr = std.io.getStdErr().writer();
    const usage =
        \\Usage:   run <checkpoint> [options]
        \\Example: run model.bin -n 256 -i "Once upon a time"
        \\
        \\Options:
        \\  -t <float>  temperature in [0,inf], default 1.0
        \\  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
        \\  -s <int>    random seed, default time(NULL)
        \\  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
        \\  -i <string> input prompt
        // \\  -z <string> optional path to custom tokenizer
        // \\  -m <string> mode: generate|chat, default: generate
        // \\  -y <string> (optional) system prompt in chat mode
        \\
        \\
    ;
    try stderr.print(usage, .{});
    std.os.exit(1);
}

/// Returns the number of tokens generated
fn generate(
    transformer: *Transformer,
    tokenizer: Tokenizer,
    sampler: *Sampler,
    prompt: []const u8,
    steps: u32,
    allocator: std.mem.Allocator,
) !u32 {
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
    return pos;
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = general_purpose_allocator.allocator();

    var checkpoint_path: ?[]const u8 = null;
    // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    var temperature: f32 = 1.0;
    var topp: f32 = 0.9;
    var rng_seed: u64 = 0;
    var steps: u32 = 256;
    var prompt: []const u8 = "Once upon a time";
    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try errorUsage();
    }
    checkpoint_path = args[1];
    var i: usize = 2;
    while (i < args.len) {
        if (i + 1 >= args.len) {
            // must have arg after flag
            try errorUsage();
        }
        if (args[i][0] != '-') {
            // must start with dash
            try errorUsage();
        }
        if (args[i].len != 2) {
            // must be -x (one dash, one letter)
            try errorUsage();
        }
        if (args[i][1] == 't') {
            if (std.fmt.parseFloat(f32, args[i + 1])) |number| {
                temperature = number;
            } else |_| {
                const stderr = std.io.getStdErr().writer();
                try stderr.print("Parsing error. Pass in a valid temperature as follows `-t <float>`.\n\n", .{});
                std.os.exit(1);
            }
        } else if (args[i][1] == 'p') {
            if (std.fmt.parseFloat(f32, args[i + 1])) |number| {
                topp = number;
            } else |_| {
                const stderr = std.io.getStdErr().writer();
                try stderr.print("Parsing error. Pass in a valid top-p as follows `-p <float>`.\n\n", .{});
                std.os.exit(1);
            }
        } else if (args[i][1] == 's') {
            if (std.fmt.parseInt(u64, args[i + 1], 10)) |number| {
                rng_seed = number;
            } else |_| {
                const stderr = std.io.getStdErr().writer();
                try stderr.print("Parsing error. Pass in a valid random seed as follows `-s <unsigned integer>`.\n\n", .{});
                std.os.exit(1);
            }
        } else if (args[i][1] == 'n') {
            if (std.fmt.parseInt(u32, args[i + 1], 10)) |number| {
                steps = number;
            } else |_| {
                const stderr = std.io.getStdErr().writer();
                try stderr.print("Parsing error. Pass in a valid number of steps as follows `-n <unsigned integer>`.\n\n", .{});
                std.os.exit(1);
            }
        } else if (args[i][1] == 'i') {
            prompt = args[i + 1];
        }
        i += 2;
    }

    // TODO: more command line argument validation

    var start_time = std.time.microTimestamp();
    var transformer = try Transformer.init(checkpoint_path.?, allocator);
    defer transformer.free(allocator);
    const tokenizer = try Tokenizer.init("tokenizer.bin", allocator, @intCast(transformer.config.vocab_size));
    defer tokenizer.free(allocator);
    var sampler = try Sampler.init(allocator, @intCast(transformer.config.vocab_size), temperature, topp, rng_seed);
    defer sampler.free(allocator);

    const num_tokens_generated = try generate(
        &transformer,
        tokenizer,
        &sampler,
        prompt,
        steps,
        allocator,
    );

    // Print perf stats
    {
        var end_time = std.time.microTimestamp();
        const sec: f32 = @as(f32, @floatFromInt(end_time - start_time)) / std.time.us_per_s;
        const tokens_per_sec: f32 = @as(f32, @floatFromInt(num_tokens_generated)) / sec;
        std.debug.print("{d} tokens/sec\n", .{tokens_per_sec});
    }
}

// If we put a string into the tokenizer's encode function and then decode the
// output, we should get the same string back. This tests that.
test "tokenizer encode decode" {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = general_purpose_allocator.allocator();
    const vocab_size: usize = 32000;

    const tokenizer = try Tokenizer.init("tokenizer.bin", allocator, vocab_size);
    defer tokenizer.free(allocator);

    const prompt: []const u8 = "This is a test. ही एक चाचणी आहे. これはテストです. 😀";
    var prompt_token_buffer = try allocator.alloc(u32, prompt.len + 3); // +3 for '\0', ?BOS, ?EOS
    defer allocator.free(prompt_token_buffer);

    var num_prompt_tokens = try tokenizer.encode(prompt, prompt_token_buffer, allocator);
    const prompt_tokens: []u32 = prompt_token_buffer[0..num_prompt_tokens];
    try std.testing.expect(num_prompt_tokens >= 1);

    // Ideally, the decoded buffer doesn't need to be any bigger than the prompt
    // itself. But in the case where there's a bug in the system, it's nice to
    // be able to see the full output, so we allocate a sufficiently large
    // buffer.
    var decoded_buffer = try allocator.alloc(u8, 100);
    var write_idx: usize = 0;

    for (0..prompt_tokens.len - 1) |i| {
        const piece: []const u8 = try tokenizer.decode(prompt_tokens[i], prompt_tokens[i + 1]);
        // Check that there is enough space in the buffer
        try std.testing.expect(write_idx + piece.len <= decoded_buffer.len);
        // Copy the piece into the buffer
        @memcpy(decoded_buffer[write_idx .. write_idx + piece.len], piece);
        write_idx += piece.len;
    }
    try std.testing.expectEqualStrings(prompt, decoded_buffer[0..write_idx]);
}
