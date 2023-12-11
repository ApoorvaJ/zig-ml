const std = @import("std");

// Struct used when sorting probabilities during top-p sampling
const ProbIndex = struct {
    prob: f32,
    index: u32,
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

/// Return the index that has the highest probability
fn sample_argmax(probabilities: []const f32) u32 {
    var max_i: u32 = 0;
    var max_p: f32 = probabilities[0];
    for (1..probabilities.len) |i| {
        if (probabilities[i] > max_p) {
            max_i = @intCast(i);
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// TODO: This funcion exists here and in Transformer.zig. Deduplicate.
fn softmax(x: []f32) void {
    // Find max value (for numerical stability)
    var max_val: f32 = x[0];
    for (1..x.len) |i| {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // Exp and sum
    var sum: f32 = 0.0;
    for (0..x.len) |i| {
        x[i] = @exp(x[i] - max_val);
        sum += x[i];
    }
    // Normalize
    for (0..x.len) |i| {
        x[i] /= sum;
    }
}

fn random_u32(state: *u64) u32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    state.* ^= state.* >> 12;
    state.* ^= state.* << 25;
    state.* ^= state.* >> 27;
    return @intCast((state.* * 0x2545F4914F6CDD1D) >> 32);
}

/// random float32 in [0,1)
fn random_f32(state: *u64) f32 {
    return @as(f32, @floatFromInt(random_u32(state) >> 8)) / 16777216.0;
}

/// Sample index from probabilities (they must sum to 1!)
/// Coin is a random number in [0, 1), usually from random_f32()
fn sample_mult(probabilities: []const f32, coin: f32) u32 {
    var cdf: f32 = 0.0;
    for (0..probabilities.len) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return @intCast(i);
        }
    }
    return @intCast(probabilities.len - 1); // In case of rounding errors
}

fn sortProb(_: void, a: ProbIndex, b: ProbIndex) bool {
    return b.prob < a.prob;
}

/// top-p sampling (or "nucleus sampling") samples from the smallest set of
/// tokens that exceed probability topp. This way we never sample tokens that
/// have very low probabilities and are less likely to go "off the rails".
/// coin is a random number in [0, 1), usually from random_f32()
fn sample_topp(probabilities: []const f32, topp: f32, prob_index: []ProbIndex, coin: f32) u32 {
    var n0: usize = 0;
    // sort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const cutoff: f32 = (1.0 - topp) / @as(f32, @floatFromInt(probabilities.len - 1));
    for (0..probabilities.len) |i| {
        if (probabilities[i] >= cutoff) {
            prob_index[n0].index = @intCast(i);
            prob_index[n0].prob = probabilities[i];
            n0 += 1;
        }
    }
    std.sort.pdq(ProbIndex, prob_index[0..n0], {}, sortProb);

    // Truncate the list where cumulative probability exceeds topp
    var cumulative_prob: f32 = 0.0;
    var last_idx: usize = n0 - 1; // in case of rounding errors consider all elements
    for (0..n0) |i| {
        cumulative_prob += prob_index[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    var r: f32 = coin * cumulative_prob;
    var cdf: f32 = 0.0;
    for (0..last_idx + 1) |i| {
        cdf += prob_index[i].prob;
        if (r < cdf) {
            return prob_index[i].index;
        }
    }
    return prob_index[last_idx].index; // in case of rounding errors
}

/// Samples the token given the logits and some hyperparameters. May mutate the logits.
pub fn sample(self: *@This(), logits: []f32) u32 {
    var next: u32 = 0;

    if (self.temperature == 0.0) {
        // Greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits);
    } else {
        // Apply the temperature to the logits
        for (0..logits.len) |q| {
            logits[q] /= self.temperature;
        }
        // Apply softmax to the logits to get the probabilities for next token
        softmax(logits);
        // Flip a (float) coin (this is our source of entropy for sampling)
        var coin: f32 = random_f32(&self.rng_state);
        // We sample from this distribution to get the next token
        if (self.topp <= 0 or self.topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, self.topp, self.prob_index, coin);
        }
    }

    return next;
}
