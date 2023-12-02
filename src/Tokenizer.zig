const std = @import("std");

const TokenIndex = struct {
    str: [][:0]const u8,
    id: i32,
};

vocab: [][:0]const u8,
vocab_scores: []f32,
sorted_vocab: []TokenIndex,
// TODO: Ensure this is necessary. Since the (i * 2)th element contains the character i, we might
// be able to get away with computing this value on the fly.
byte_pieces: [512]u8, // stores all single-byte strings
max_token_length: i32,

pub fn init(tokenizer_path: []const u8, allocator: std.mem.Allocator, vocab_size: usize) !@This() {
    var vocab: [][:0]u8 = try allocator.alloc([:0]u8, vocab_size);
    var vocab_scores: []f32 = try allocator.alloc(f32, vocab_size);
    var sorted_vocab: []TokenIndex = try allocator.alloc(TokenIndex, vocab_size);
    var byte_pieces: [512]u8 = undefined;
    for (0..256) |i| {
        byte_pieces[i * 2] = @intCast(i);
        byte_pieces[i * 2 + 1] = 0;
    }

    // Read the tokenizer file
    var max_token_length: i32 = 0;
    {
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
    }
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
