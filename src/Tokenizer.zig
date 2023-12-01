const std = @import("std");

const TokenIndex = struct {
    str: [][:0]const u8,
    id: i32,
};

vocab: [][:0]const u8,
vocab_scores: []f32,
sorted_vocab: []TokenIndex,
// max_token_length: u32,
// byte_pieces: [512]u8, // stores all single-byte strings

pub fn init(tokenizer_path: []const u8, allocator: std.mem.Allocator, vocab_size: usize) !@This() {
    _ = tokenizer_path;
    var vocab: [][:0]u8 = try allocator.alloc([:0]u8, vocab_size);
    var vocab_scores: []f32 = try allocator.alloc(f32, vocab_size);
    var sorted_vocab: []TokenIndex = try allocator.alloc(TokenIndex, vocab_size);
    return .{
        .vocab = vocab,
        .vocab_scores = vocab_scores,
        .sorted_vocab = sorted_vocab,
    };
}

pub fn free(self: @This(), allocator: std.mem.Allocator) void {
    allocator.free(self.vocab);
    allocator.free(self.vocab_scores);
    allocator.free(self.sorted_vocab);
}
