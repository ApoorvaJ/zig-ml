/// x is both the input and the output
pub fn softmax(x: []f32) void {
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
