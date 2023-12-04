const std = @import("std");

const TokenIndex = struct {
    str: []const u8,
    id: u32,
};

vocab: [][]const u8,
vocab_scores: []f32,
sorted_vocab: []TokenIndex,
// TODO: Ensure this is necessary. Since the (i * 2)th element contains the character i, we might
// be able to get away with computing this value on the fly.
byte_pieces: [512]u8, // stores all single-byte strings
max_token_length: i32,

fn sortToken(_: void, a: TokenIndex, b: TokenIndex) bool {
    return std.mem.order(u8, a.str, b.str).compare(std.math.CompareOperator.lt);
}
// fn(context:@TypeOf(context), key:@TypeOf(key), mid_item:T)math.Order
fn compareToken(_: void, key: []const u8, mid_item: TokenIndex) std.math.Order {
    return std.mem.order(u8, key, mid_item.str);
}

pub fn init(tokenizer_path: []const u8, allocator: std.mem.Allocator, vocab_size: usize) !@This() {
    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
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
        vocab[i] = try allocator.alloc(u8, len);
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
    _ = allocator;
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    const prepend_bos_token: bool = true;
    const append_eos_token: bool = false;
    _ = append_eos_token;

    var n_tokens: usize = 0;

    // add optional BOS (=1) token, if desired
    if (prepend_bos_token) {
        out_tokens[n_tokens] = 1;
        n_tokens += 1;
    }

    // Add the dummy prefix token
    {
        // TODO: This binary search takes a significant amount of time. It can be avoided by pre-storing
        // the index of this token.
        const key: []const u8 = " ";
        const opt_idx: ?usize = std.sort.binarySearch(TokenIndex, key, self.sorted_vocab, {}, compareToken);
        std.debug.assert(opt_idx != null);
        const token_id = self.sorted_vocab[opt_idx.?].id;
        out_tokens[n_tokens] = token_id;
        n_tokens += 1;
    }

    // Create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    const max_len: usize = @intCast(self.max_token_length);
    _ = max_len;
    // var str_buffer = try allocator.alloc(u8, max_len * 2 + 1 + 2);
    // defer allocator.free(str_buffer);

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

        const codept: []const u8 = text[i .. i + codept_len];
        const opt_idx: ?usize = std.sort.binarySearch(TokenIndex, codept, self.sorted_vocab, {}, compareToken);

        if (opt_idx != null) {
            // We found this codepoint in vocab. Add it as a token.
            out_tokens[n_tokens] = self.sorted_vocab[opt_idx.?].id;
            n_tokens += 1;
        } else {
            // We didn't find this codepoint in vocab.
            // Use byte_fallback encoding: just encode each byte as a token.
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (codept) |byte| {
                out_tokens[n_tokens] = byte + 3;
                n_tokens += 1;
            }
        }

        i += codept_len;
    }

    return n_tokens;
}
