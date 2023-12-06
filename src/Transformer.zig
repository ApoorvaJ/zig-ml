const std = @import("std");
const builtin = @import("builtin");
const MappedFile = @import("MappedFile.zig");

const Config = struct {
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length
};

const Weights = struct {
    // Token embedding table
    token_embedding_table: []align(1) const f32, // (vocab_size, dim)
    // Weights for rmsnorms
    rms_att_weight: []align(1) const f32, // (layer, dim) rmsnorm weights
    rms_ffn_weight: []align(1) const f32, // (layer, dim)
    // Weights for matmuls. note dim == n_heads * head_size
    wq: []align(1) const f32, // (layer, dim, n_heads * head_size)
    wk: []align(1) const f32, // (layer, dim, n_kv_heads * head_size)
    wv: []align(1) const f32, // (layer, dim, n_kv_heads * head_size)
    wo: []align(1) const f32, // (layer, n_heads * head_size, dim)
    // Weights for ffn
    w1: []align(1) const f32, // (layer, hidden_dim, dim)
    w2: []align(1) const f32, // (layer, dim, hidden_dim)
    w3: []align(1) const f32, // (layer, hidden_dim, dim)
    // Final rmsnorm
    rms_final_weight: []align(1) const f32, // (dim,)
    // (Optional) classifier weights for the logits, on the last layer
    wcls: []align(1) const f32,
};

const RunState = struct {
    // Current wave of activations
    x: []f32, // activation at current time stamp (dim,)
    xb: []f32, // same, but inside a residual branch (dim,)
    xb2: []f32, // an additional buffer just for convenience (dim,)
    hb: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []f32, // query (dim,)
    k: []f32, // key (kv_dim,)
    v: []f32, // value (kv_dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits
    // Key-value cache
    key_cache: []f32, // (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)
};

config: Config,
weights: Weights,
run_state: RunState,
mapped_file: MappedFile,

/// This function returns a slice of f32s from the source u8 slice. It also increments the start
/// address.
fn slice_and_increment(
    src: []align(std.mem.page_size) const u8,
    start: *usize,
    num_f32s: usize,
) []align(1) const f32 {
    const end: usize = start.* + @sizeOf(f32) * num_f32s;
    const out_slice = std.mem.bytesAsSlice(f32, src[(start.*)..end]);
    start.* = end;
    return out_slice;
}

pub fn init(checkpoint_path: []const u8, allocator: std.mem.Allocator) !@This() {
    var file_size: u64 = 0;
    var shared_weights: bool = false;
    var config: Config = undefined;
    // Read the config from the file
    {
        const file = try std.fs.cwd().openFile(checkpoint_path, .{});
        defer file.close();

        // Read in the config header
        const config_slice = std.mem.asBytes(&config);
        _ = try file.read(config_slice);
        // Negative vocab size is hacky way of signaling unshared weights. Bit yikes.
        shared_weights = config.vocab_size > 0;
        config.vocab_size = try std.math.absInt(config.vocab_size);
        // Figure out the file size
        file_size = try file.getEndPos();
    }
    // Memory map the Transformer weights into the data pointer
    var mapped_file = try MappedFile.init(checkpoint_path, .{ .mode = .read_only });

    const n_layers: usize = @intCast(config.n_layers);
    const dim: usize = @intCast(config.dim);
    const hidden_dim: usize = @intCast(config.hidden_dim);
    const vocab_size: usize = @intCast(config.vocab_size);
    const n_heads: usize = @intCast(config.n_heads);
    const n_kv_heads: usize = @intCast(config.n_kv_heads);
    const seq_len: usize = @intCast(config.seq_len);

    // Obtain slices into the mapped buffer for each of the weights
    var weights: Weights = undefined;
    {
        std.debug.assert(config.dim > 0 and config.n_heads > 0 and config.n_layers > 0 and config.vocab_size > 0 and config.n_heads > 0);
        const head_size: usize = dim / n_heads;

        var start: usize = @sizeOf(Config);
        weights.token_embedding_table = slice_and_increment(mapped_file.mem, &start, vocab_size * dim);
        weights.rms_att_weight = slice_and_increment(mapped_file.mem, &start, n_layers * dim);
        weights.wq = slice_and_increment(mapped_file.mem, &start, n_layers * dim * (n_heads * head_size));
        weights.wk = slice_and_increment(mapped_file.mem, &start, n_layers * dim * (n_kv_heads * head_size));
        weights.wv = slice_and_increment(mapped_file.mem, &start, n_layers * dim * (n_kv_heads * head_size));
        weights.wo = slice_and_increment(mapped_file.mem, &start, n_layers * dim * (n_heads * head_size));
        weights.rms_ffn_weight = slice_and_increment(mapped_file.mem, &start, n_layers * dim);
        weights.w1 = slice_and_increment(mapped_file.mem, &start, n_layers * dim * hidden_dim);
        weights.w2 = slice_and_increment(mapped_file.mem, &start, n_layers * dim * hidden_dim);
        weights.w3 = slice_and_increment(mapped_file.mem, &start, n_layers * dim * hidden_dim);
        weights.rms_final_weight = slice_and_increment(mapped_file.mem, &start, dim);
        if (shared_weights) {
            weights.wcls = weights.token_embedding_table;
        } else {
            start += seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
            start += seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
            weights.wcls = slice_and_increment(mapped_file.mem, &start, vocab_size * dim);
        }
    }
    // Heap-allocate the run state
    var run_state: RunState = undefined;
    {
        const kv_dim: usize = (dim * n_kv_heads) / n_heads;

        run_state.x = try allocator.alloc(f32, dim);
        run_state.xb = try allocator.alloc(f32, dim);
        run_state.xb2 = try allocator.alloc(f32, dim);
        run_state.hb = try allocator.alloc(f32, hidden_dim);
        run_state.hb2 = try allocator.alloc(f32, hidden_dim);
        run_state.q = try allocator.alloc(f32, dim);
        run_state.key_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim);
        run_state.value_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim);
        run_state.att = try allocator.alloc(f32, n_heads * seq_len);
        run_state.logits = try allocator.alloc(f32, vocab_size);
    }

    return .{
        .config = config,
        .weights = weights,
        .run_state = run_state,
        .mapped_file = mapped_file,
    };
}

pub fn free(self: @This(), allocator: std.mem.Allocator) void {
    allocator.free(self.run_state.x);
    allocator.free(self.run_state.xb);
    allocator.free(self.run_state.xb2);
    allocator.free(self.run_state.hb);
    allocator.free(self.run_state.hb2);
    allocator.free(self.run_state.q);
    allocator.free(self.run_state.key_cache);
    allocator.free(self.run_state.value_cache);
    allocator.free(self.run_state.att);
    allocator.free(self.run_state.logits);
    self.mapped_file.unmap();
}

fn rmsnorm(o: []f32, x: []const f32, weight: []align(1) const f32) void {
    std.debug.assert(o.len == x.len and x.len == weight.len);
    // Calculate sum of squares
    var ss: f32 = 0.0;
    for (0..x.len) |i| {
        ss += x[i] * x[i];
    }
    ss /= @floatFromInt(x.len);
    ss += 1e-5;
    ss = 1.0 / @sqrt(ss);
    // normalize and scale
    for (0..x.len) |i| {
        o[i] = weight[i] * (ss * x[i]);
    }
}

/// x is both the input and the output
fn softmax(x: []f32) void {
    // find max value (for numerical stability)
    var max_val: f32 = x[0];
    for (1..x.len) |i| {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (0..x.len) |i| {
        x[i] = @exp(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (0..x.len) |i| {
        x[i] /= sum;
    }
}

fn matmul(xout: []f32, x: []const f32, w: []align(1) const f32) void {
    // W (d,n) @ x (n,) -> xout (d,)
    const d = xout.len;
    const n = x.len;
    std.debug.assert(w.len == d * n);

    for (0..d) |i| {
        var val: f32 = 0.0;
        for (0..n) |j| {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

pub fn forward(self: *@This(), token: u32, pos: u32) []f32 {
    std.debug.assert(self.config.seq_len > pos);

    const dim: usize = @intCast(self.config.dim);
    const n_layers: usize = @intCast(self.config.n_layers);
    const n_heads: usize = @intCast(self.config.n_heads);
    const n_kv_heads: usize = @intCast(self.config.n_kv_heads);
    const seq_len: usize = @intCast(self.config.seq_len);
    const kv_dim: usize = (dim * n_kv_heads) / n_heads;
    const kv_mul: usize = n_heads / n_kv_heads; // Integer multiplier of the kv sharing in multiquery
    const head_size: usize = dim / n_heads;

    // Copy the token embedding into x
    @memcpy(self.run_state.x, self.weights.token_embedding_table[token * dim .. token * dim + dim]);

    // Forward all the layers
    for (0..n_layers) |l| {
        // Attention rmsnorm
        rmsnorm(self.run_state.xb, self.run_state.x, self.weights.rms_att_weight[l * dim .. l * dim + dim]);

        // Key and value point to the kv cache
        const loff: usize = l * seq_len * kv_dim;
        self.run_state.k = self.run_state.key_cache[loff + pos * kv_dim .. loff + pos * kv_dim + kv_dim];
        self.run_state.v = self.run_state.value_cache[loff + pos * kv_dim .. loff + pos * kv_dim + kv_dim];

        // qkv matmuls for this position
        matmul(self.run_state.q, self.run_state.xb, self.weights.wq[l * dim * dim .. l * dim * dim + (dim * dim)]);
        matmul(self.run_state.k, self.run_state.xb, self.weights.wk[l * dim * kv_dim .. l * dim * kv_dim + (dim * kv_dim)]);
        matmul(self.run_state.v, self.run_state.xb, self.weights.wv[l * dim * kv_dim .. l * dim * kv_dim + (dim * kv_dim)]);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        var i: usize = 0;
        while (i < dim) {
            const head_dim = i % head_size;
            const freq: f32 = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(head_dim)) / @as(f32, @floatFromInt(head_size)));
            const val: f32 = @as(f32, @floatFromInt(pos)) * freq;
            const fcr: f32 = @cos(val);
            const fci: f32 = @sin(val);
            const rotn: u2 = if (i < kv_dim) 2 else 1; // How many vectors? 2 = q & k, 1 = q only
            for (0..rotn) |v| {
                const vec: []f32 = if (v == 0) self.run_state.q else self.run_state.k; // the vector to rotate (query or key)
                const v0: f32 = vec[i];
                const v1: f32 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
            i += 2;
        }

        // Multihead attention. Iterate over all heads.
        for (0..n_heads) |h| {
            // Get the query vector for this head
            const q: []f32 = self.run_state.q[h * head_size .. (h + 1) * head_size];
            // Attention scores for this head
            const att: []f32 = self.run_state.att[h * seq_len .. (h + 1) * seq_len];
            // Iterate over all timesteps, including the current one
            for (0..pos + 1) |t| {
                // Get the key vector for this head and at this timestep
                const koff: usize = loff + t * kv_dim + (h / kv_mul) * head_size;
                const k: []f32 = self.run_state.key_cache[koff .. koff + head_size];
                // Calculate the attention score as the dot product of q and k
                var score: f32 = 0.0;
                for (0..head_size) |j| {
                    score += q[j] * k[j];
                }
                score /= @sqrt(@as(f32, @floatFromInt(head_size)));
                // save the score to the attention buffer
                att[t] = score;
            }

            // Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att[0 .. pos + 1]);

            // Weighted sum of the values, store back into xb
            const xb: []f32 = self.run_state.xb[h * head_size .. (h + 1) * head_size];
            @memset(xb, 0);
            for (0..pos + 1) |t| {
                // Get the value vector for this head and at this timestep
                const voff: usize = loff + t * kv_dim + (h / kv_mul) * head_size;
                const v: []f32 = self.run_state.value_cache[voff .. voff + head_size];
                // Get the attention weight for this timestep
                const a = att[t];
                // Accumulate the weighted value into xb
                for (0..head_size) |j| {
                    xb[j] += a * v[j];
                }
            }
        }

        // Final matmul to get the output of the attention
        matmul(self.run_state.xb2, self.run_state.xb, self.weights.wo[l * dim * dim .. (l + 1) * dim * dim]);

        // residual connection back into x
        for (0..dim) |j| {
            self.run_state.x[j] += self.run_state.xb2[j];
        }

        // ffn rmsnorm
        rmsnorm(self.run_state.xb, self.run_state.x, self.weights.rms_ffn_weight[l * dim .. (l + 1) * dim]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        const hidden_dim: usize = @intCast(self.config.hidden_dim);
        matmul(self.run_state.hb, self.run_state.xb, self.weights.w1[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim]);
        matmul(self.run_state.hb2, self.run_state.xb, self.weights.w3[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim]);

        // SwiGLU non-linearity
        for (0..hidden_dim) |j| {
            var val: f32 = self.run_state.hb[j];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0 / (1.0 + @exp(-val)));
            // elementwise multiply with w3(x)
            val *= self.run_state.hb2[j];
            self.run_state.hb[j] = val;
        }

        // Final matmul to get the output of the ffn
        matmul(self.run_state.xb, self.run_state.hb, self.weights.w2[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim]);

        // Residual connection
        for (0..dim) |j| {
            self.run_state.x[j] += self.run_state.xb[j];
        }
    }

    // Final rmsnorm
    // TODO: The first two slices here alias. Need to make sure this is okay.
    rmsnorm(self.run_state.x, self.run_state.x, self.weights.rms_final_weight);

    // classifier into logits
    matmul(self.run_state.logits, self.run_state.x, self.weights.wcls);
    return self.run_state.logits;
}
