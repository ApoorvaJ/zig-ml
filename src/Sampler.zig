const std = @import("std");

// Struct used when sorting probabilities during top-p sampling
const ProbIndex = struct {
    prob: f32,
    index: i32,
};

prob_index: []ProbIndex, // Buffer used in top-p sampling
temperature: f32,
topp: f32,
rng_state: u64,

pub fn init(
    allocator: std.mem.Allocator,
    vocab_size: usize,
    temperature: f32,
    topp: f32,
    rng_seed: u64,
) !@This() {
    // Buffer only used with nucleus sampling; may not need but it's ~small
    const prob_index = try allocator.alloc(ProbIndex, vocab_size);
    return .{
        .prob_index = prob_index,
        .temperature = temperature,
        .topp = topp,
        .rng_state = rng_seed,
    };
}

pub fn free(self: @This(), allocator: std.mem.Allocator) void {
    allocator.free(self.prob_index);
}
