const std = @import("std");

const TokenIndex = struct {
    str: [:0]const u8,
    id: i32,
};

vocab: [][:0]const u8,
vocab_scores: []f32,
sorted_vocab: []TokenIndex,
// TODO: Ensure this is necessary. Since the (i * 2)th element contains the character i, we might
// be able to get away with computing this value on the fly.
byte_pieces: [512]u8, // stores all single-byte strings
max_token_length: i32,

fn compareTokenIndex(_: void, a: TokenIndex, b: TokenIndex) bool {
    return std.mem.orderZ(u8, a.str, b.str).compare(std.math.CompareOperator.lt);
}

pub fn init(tokenizer_path: []const u8, allocator: std.mem.Allocator, vocab_size: usize) !@This() {
    var vocab: [][:0]u8 = try allocator.alloc([:0]u8, vocab_size);
    var vocab_scores: []f32 = try allocator.alloc(f32, vocab_size);
    var byte_pieces: [512]u8 = undefined;
    for (0..256) |i| {
        byte_pieces[i * 2] = @intCast(i);
        byte_pieces[i * 2 + 1] = 0;
    }

    // Read the tokenizer file
    var max_token_length: i32 = 0;
    const file = try std.fs.cwd().openFile(tokenizer_path, .{});
    defer file.close();
    // Read the max token length
    {
        const slice = std.mem.asBytes(&max_token_length);
        _ = try file.read(slice);
    }

    for (0..vocab_size) |i| {
        // Read the vocab score
        {
            const slice = std.mem.asBytes(&vocab_scores[i]);
            _ = try file.read(slice);
        }
        // Read the length of the next token
        var len: usize = 0;
        {
            var len_i32: i32 = 0;
            const slice = std.mem.asBytes(&len_i32);
            _ = try file.read(slice);
            len = @intCast(len_i32);
        }
        // Allocate memory for the token
        vocab[i] = try allocator.allocSentinel(u8, len, 0);
        // Read the token
        {
            const slice = vocab[i][0..len];
            _ = try file.read(slice);
        }
    }
    // Allocate and sort the vocab
    // TODO: This can be pre-calculated and written to the tokenizer file before
    // the program even runs.
    var sorted_vocab: []TokenIndex = try allocator.alloc(TokenIndex, vocab_size);
    for (0..vocab_size) |i| {
        sorted_vocab[i].str = vocab[i];
        sorted_vocab[i].id = @intCast(i);
    }
    // Sort sorted_vocab by the string value of each token, using the Zig standard library.
    std.sort.pdq(TokenIndex, sorted_vocab, {}, compareTokenIndex);

    return .{
        .vocab = vocab,
        .vocab_scores = vocab_scores,
        .sorted_vocab = sorted_vocab,
        .byte_pieces = byte_pieces,
        .max_token_length = max_token_length,
    };
}

pub fn free(self: @This(), allocator: std.mem.Allocator) void {
    for (self.vocab) |token| {
        allocator.free(token);
    }
    allocator.free(self.vocab);
    allocator.free(self.vocab_scores);
    allocator.free(self.sorted_vocab);
}

/// Encode the string text (input) into an upper-bound preallocated out_tokens[]
/// array.
pub fn encode(
    self: @This(),
    text: [:0]const u8,
    out_tokens: []u32,
    allocator: std.mem.Allocator,
) !usize {
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    const prepend_bos_token: bool = true;
    const append_eos_token: bool = false;
    _ = text;
    _ = append_eos_token;

    // Create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    const max_len: usize = @intCast(self.max_token_length);
    var str_buffer = try allocator.alloc(u8, max_len * 2 + 1 + 2);
    defer allocator.free(str_buffer);

    var n_tokens: usize = 0;

    // add optional BOS (=1) token, if desired
    if (prepend_bos_token) {
        out_tokens[0] = 1;
        n_tokens += 1;
    }

    return n_tokens;
}
