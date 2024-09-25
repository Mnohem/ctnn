const std = @import("std");
const mem = std.mem;

/// Not a robust implementation, just need to be able to open the MNIST dataset
pub fn openIdxFile(comptime filename: [:0]const u8) struct { i32, [:0]const u8 } {
    const file_string = @embedFile(filename);
    // first two bytes are 0
    // third byte is data type, in our case it is 8, meaning an unsigned byte
    // fourth byte is the number of dimensions
    const dim_amount = file_string[3];
    comptime std.debug.assert(file_string[2] == 8);

    const data_start = 4 * dim_amount + 4;
    // next dim_amount ints (4 bytes) are high endian ints which are the size of each dimension
    const bigend_dim_sizes = mem.bytesAsSlice(i32, file_string[4..data_start]);

    return .{
        mem.bigToNative(i32, bigend_dim_sizes[0]),
        file_string[data_start..],
    };
}
