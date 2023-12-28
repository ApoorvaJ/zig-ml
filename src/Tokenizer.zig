const std = @import("std");

const TokenIndex = struct {
    str: []const u8,
    id: u32,
};

vocab: [][]const u8,
vocab_scores: []f32,
sorted_vocab: []TokenIndex,
byte_pieces: [256]u8, // stores all single-byte strings
max_token_length: i32,

fn sortToken(_: void, a: TokenIndex, b: TokenIndex) bool {
    return std.mem.order(u8, a.str, b.str).compare(std.math.CompareOperator.lt);
}

fn compareToken(_: void, key: []const u8, mid_item: TokenIndex) std.math.Order {
    return std.mem.order(u8, key, mid_item.str);
}

pub fn init(tokenizer_path: []const u8, allocator: std.mem.Allocator, vocab_size: usize) !@This() {
    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var vocab_scores: []f32 = try allocator.alloc(f32, vocab_size);
    var byte_pieces: [256]u8 = undefined;
    for (0..256) |i| {
        byte_pieces[i] = @intCast(i);
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
        vocab[i] = try allocator.alloc(u8, len);
        // Read the token
        {
            const slice = vocab[i][0..len];
            _ = try file.read(slice);
        }
    }
    // Allocate and sort the vocab
    var sorted_vocab: []TokenIndex = try allocator.alloc(TokenIndex, vocab_size);
    for (0..vocab_size) |i| {
        sorted_vocab[i].str = vocab[i];
        sorted_vocab[i].id = @intCast(i);
    }
    // Sort sorted_vocab by the string value of each token, using the Zig standard library.
    std.sort.pdq(TokenIndex, sorted_vocab, {}, sortToken);

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
    text: []const u8,
    out_tokens: []u32,
    allocator: std.mem.Allocator,
) !usize {
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    const prepend_bos_token: bool = true;
    const append_eos_token: bool = false;

    var n_tokens: usize = 0;

    // add optional BOS (=1) token, if desired
    if (prepend_bos_token) {
        out_tokens[n_tokens] = 1;
        n_tokens += 1;
    }

    // Add the dummy prefix token
    {
        const key: []const u8 = " ";
        const opt_idx: ?usize = std.sort.binarySearch(TokenIndex, key, self.sorted_vocab, {}, compareToken);
        std.debug.assert(opt_idx != null);
        const token_id = self.sorted_vocab[opt_idx.?].id;
        out_tokens[n_tokens] = token_id;
        n_tokens += 1;
    }

    // Convert each UTF-8 codepoint into a token
    var i: usize = 0;
    while (i < text.len) {
        // The number of bytes in the codepoint
        var codept_len: usize = 0;
        if (std.unicode.utf8ByteSequenceLength(text[i])) |number| {
            // ^ `c` is the first byte of a UTF-8 codepoint. `number` is the number of bytes in the
            // codepoint.
            codept_len = number;
        } else |_| {
            // ^ `c` is a continuation byte of a UTF-8 codepoint.
            // This should never happen, because we're always parsing whole codepoints.
            return error.InputIsInvalidUTF8;
        }
        // Get a slice into the text that contains the codepoint
        const codept: []const u8 = text[i .. i + codept_len];
        if (std.sort.binarySearch(TokenIndex, codept, self.sorted_vocab, {}, compareToken)) |idx| {
            // We found this codepoint in vocab. Add it as a token.
            out_tokens[n_tokens] = self.sorted_vocab[idx].id;
            n_tokens += 1;
        } else {
            // We didn't find this codepoint in vocab.
            // Use byte fallback encoding: just encode each byte as a token.
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (codept) |byte| {
                out_tokens[n_tokens] = byte + 3;
                n_tokens += 1;
            }
        }

        i += codept_len;
    }

    // Create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    const max_len: usize = @intCast(self.max_token_length);
    var str_buffer = try allocator.alloc(u8, max_len * 2 + 1 + 2);
    defer allocator.free(str_buffer);

    while (true) {
        var best_score: f32 = -1e10;
        var opt_best_id: ?u32 = null;
        var opt_best_idx: ?usize = null;

        for (0..n_tokens - 1) |j| {
            const t0: []const u8 = self.vocab[out_tokens[j]];
            const t1: []const u8 = self.vocab[out_tokens[j + 1]];
            @memcpy(str_buffer[0..t0.len], t0);
            @memcpy(str_buffer[t0.len .. t0.len + t1.len], t1);
            // At this point, the str_buffer contains the concatenation of the two tokens, plus
            // some gargabe at the end. We take a slice of it that contains only the two tokens.
            const combined_token: []const u8 = str_buffer[0 .. t0.len + t1.len];

            const opt_idx = std.sort.binarySearch(TokenIndex, combined_token, self.sorted_vocab, {}, compareToken);
            if (opt_idx) |idx| {
                const id = self.sorted_vocab[idx].id;
                if (best_score < self.vocab_scores[id]) {
                    best_score = self.vocab_scores[id];
                    opt_best_id = @intCast(id);
                    opt_best_idx = j;
                }
            }
        }

        if (opt_best_idx == null) {
            // ^ We couldn't find any more pairs to merge, so we're done.
            break;
        }

        // Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        out_tokens[opt_best_idx.?] = opt_best_id.?;
        // Left shift the rest of the array. Since the source and the destination overlap, we have
        // to use this function instead of @memcpy.
        std.mem.copyForwards(u32, out_tokens[opt_best_idx.? + 1 .. n_tokens - 1], out_tokens[opt_best_idx.? + 2 .. n_tokens]);
        n_tokens -= 1;
    }

    // add optional EOS (=2) token, if desired
    if (append_eos_token) {
        out_tokens[n_tokens] = 2;
        n_tokens += 1;
    }

    return n_tokens;
}

pub fn decode(
    self: @This(),
    prev_token: u32,
    token: u32,
) ![]const u8 {
    var slice: []const u8 = self.vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if (prev_token == 1 and slice[0] == ' ') {
        slice = slice[1..];
    }
    // Careful, some tokens designate raw bytes, and look like e.g. '<0x01>'.
    // Parse this and convert and return the actual byte.
    // Check if the string starts with "<0x" and ends with ">".
    if (std.mem.startsWith(u8, slice, "<0x") and std.mem.endsWith(u8, slice, ">")) {
        // Extract the hexadecimal part
        const hexPart = slice[3 .. slice.len - 1];

        // Parse the hexadecimal string
        var byte_val: u8 = try std.fmt.parseInt(u8, hexPart, 16);
        return self.byte_pieces[byte_val .. byte_val + 1];
    }

    return slice;
}
