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
// max_token_length: u32,

pub fn init(tokenizer_path: []const u8, allocator: std.mem.Allocator, vocab_size: usize) !@This() {
    _ = tokenizer_path;
    var vocab: [][:0]u8 = try allocator.alloc([:0]u8, vocab_size);
    var vocab_scores: []f32 = try allocator.alloc(f32, vocab_size);
    var sorted_vocab: []TokenIndex = try allocator.alloc(TokenIndex, vocab_size);
    var byte_pieces: [512]u8 = undefined;
    for (0..256) |i| {
        byte_pieces[i * 2] = @intCast(i);
        byte_pieces[i * 2 + 1] = 0;
    }
    return .{
        .vocab = vocab,
        .vocab_scores = vocab_scores,
        .sorted_vocab = sorted_vocab,
        .byte_pieces = byte_pieces,
    };
}

pub fn free(self: @This(), allocator: std.mem.Allocator) void {
    allocator.free(self.vocab);
    allocator.free(self.vocab_scores);
    allocator.free(self.sorted_vocab);
}
