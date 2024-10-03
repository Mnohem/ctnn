const std = @import("std");

// power operator is hidden in the unused (_) bits of the Operator enum
// unused bits are interpreted as a signed int to be raised to
// .external means this value came from an operation in ManyValueManager
const Operator = enum(i8) { noop, add, mul, exp, neg, external, _ };
const Idx = enum(u24) { _ };
// Index is crammed with Operator, meaning our index is 24 bit
// Thus we can only store 16,777,215 values
pub const ValueRef = packed struct {
    op: Operator,
    idx: Idx,
};
// Require caller to give a one_value for easy interop with Vectors
pub fn ValueManager(Scalar: type, vector_size: comptime_int) type {
    switch (@typeInfo(Scalar)) {
        .float => {},
        else => @compileError(std.fmt.comptimePrint("Expected a float scalar type, found {} instead", .{Scalar})),
    }
    const data_is_scalar = vector_size == 0;
    const Data = if (data_is_scalar) Scalar else @Vector(vector_size, Scalar);
    const one: Data = if (data_is_scalar) 1 else @splat(1);
    // const one_half: Data = one / (one + one);

    // The lowest amount of bits Im willing to give to the power operation
    const LOWEST_POWER_SIZE = 4;
    const op_without_int_size: comptime_int = @ceil(@log2(@as(comptime_float, @floatFromInt(std.meta.fields(Operator).len))));
    const size_for_int = 8 - op_without_int_size;
    if (size_for_int < LOWEST_POWER_SIZE) @compileError(std.fmt.comptimePrint("Operator enum is too large to store int: {d} bits left", .{size_for_int}));
    const PowInt = std.meta.Int(.signed, size_for_int);

    comptime std.debug.assert(@sizeOf(ValueRef) == 4);

    return struct {
        allocator: std.mem.Allocator,
        // data_storage and grad_storage indexed by Idx Type
        data_storage: std.ArrayListUnmanaged(Data),
        grad_storage: std.ArrayListUnmanaged(Data),
        child_idx_map: std.AutoArrayHashMapUnmanaged(Idx, usize),
        // children_storage indexed by the usize returned from child_idx_map
        // children from one parent are stored together after the index so they may be taken as a slice
        // children_storage stores ValueRefs for local operations, i32 for pow
        // stores *ValueManager over two indices and other children for external operations
        children_storage: std.ArrayListUnmanaged(u32),

        const Self = @This();

        pub fn init(a: std.mem.Allocator, capacity: usize) !Self {
            return Self{
                .allocator = a,
                .data_storage = try std.ArrayListUnmanaged(Data).initCapacity(a, capacity),
                .grad_storage = std.ArrayListUnmanaged(Data){ .items = &[_]Data{}, .capacity = 0 },
                .child_idx_map = try std.AutoArrayHashMapUnmanaged(Idx, usize).init(a, &[_]Idx{}, &[_]usize{}),
                .children_storage = try std.ArrayListUnmanaged(u32).initCapacity(a, capacity),
            };
        }
        pub fn deinit(self: *Self) void {
            self.data_storage.deinit(self.allocator);
            self.grad_storage.deinit(self.allocator);
            self.child_idx_map.deinit(self.allocator);
            self.children_storage.deinit(self.allocator);
        }
        pub fn new(self: *Self, data: Data) ValueRef {
            self.data_storage.append(self.allocator, data) catch |err| {
                std.debug.panic("Failed to store data {}: {}", .{ data, err });
            };

            return .{
                .op = .noop,
                .idx = @enumFromInt(self.data_storage.items.len - 1),
            };
        }
        pub fn getDataPtr(self: *const Self, ref: ValueRef) *Data {
            return &self.data_storage.items[@intFromEnum(ref.idx)];
        }
        pub fn getData(self: *const Self, ref: ValueRef) Data {
            return self.data_storage.items[@intFromEnum(ref.idx)];
        }
        // Calling this function is only valid after using backward
        pub fn getGrad(self: *const Self, ref: ValueRef) Data {
            return self.grad_storage.items[@intFromEnum(ref.idx)];
        }
        fn newExpr(self: *Self, op: Operator, data: Data, children: []const u32) !ValueRef {
            var new_ref = self.new(data);
            errdefer {
                _ = self.data_storage.pop();
            }
            new_ref.op = op;

            try self.child_idx_map.put(self.allocator, new_ref.idx, self.children_storage.items.len);
            errdefer _ = self.child_idx_map.swapRemove(new_ref.idx);
            try self.children_storage.appendSlice(self.allocator, children);

            return new_ref;
        }

        pub fn add(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.newExpr(.add, self.getData(id1) + self.getData(id2), &[_]u32{ @bitCast(id1), @bitCast(id2) }) catch |err| {
                std.debug.panic("Failed to add refs {} and {}: {}", .{ id1, id2, err });
            };
        }
        pub fn mul(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.newExpr(.mul, self.getData(id1) * self.getData(id2), &[_]u32{ @bitCast(id1), @bitCast(id2) }) catch |err| {
                std.debug.panic("Failed to multiply refs {} and {}: {}", .{ id1, id2, err });
            };
        }
        pub fn div(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.mul(id1, self.powi(id2, -1));
        }
        // pow cannot be raised to a Value power because this changes derivs, we will have more complications
        // Though we still need to hold the num for backward, so we cram it into the Operator enum
        // PowInt is type ix where x is the number of unused bits in Operator
        // We cant store PowInt as a Value since it can't be a vector and we cant rely on Data being a scalar
        pub fn powi(self: *Self, ref: ValueRef, num: PowInt) ValueRef {
            const data = self.getData(ref);
            var result: Data = one;
            for (0..@abs(num)) |_| {
                result *= data;
            }
            result = if (std.math.sign(num) == -1) one / result else result;
            return self.newExpr(@enumFromInt(@as(i8, num) << op_without_int_size), result, &[_]u32{@bitCast(ref)}) catch |err| {
                std.debug.panic("Failed to raise ref {} to the power of {d}: {}", .{ ref, num, err });
            };
        }
        pub fn exp(self: *Self, ref: ValueRef) ValueRef {
            return self.newExpr(.exp, @exp(self.getData(ref)), &[_]u32{@bitCast(ref)}) catch |err| {
                std.debug.panic("Failed to exponentiate ref {}: {}", .{ ref, err });
            };
        }
        pub fn neg(self: *Self, ref: ValueRef) ValueRef {
            return self.newExpr(.neg, -self.getData(ref), &[_]u32{@bitCast(ref)}) catch |err| {
                std.debug.panic("Failed to negate ref {}: {}", .{ ref, err });
            };
        }
        pub fn sub(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.add(id1, self.neg(id2));
        }

        pub fn eval(self: *Self, comptime expr: []const u8, variables: anytype) !@import("eval.zig").VarRefs(@TypeOf(variables), ValueRef) {
            return @import("eval.zig").parse(Data, self, expr, variables);
        }

        pub fn zeroGrad(self: *Self) void {
            @memset(self.grad_storage.items, std.mem.zeroes(Data));
        }

        fn recalculate(self: *Self, ref: ValueRef) void {
            if (ref.op == .noop or ref.op == .external) return;
            const c = self.children_storage.items[self.child_idx_map.get(ref.idx).?..];

            switch (ref.op) {
                .noop, .external => unreachable,
                .add, .mul => {
                    self.recalculate(@bitCast(c[0]));
                    self.recalculate(@bitCast(c[1]));
                },
                _ => self.recalculate(@bitCast(c[0])),
                .exp, .neg => self.recalculate(@bitCast(c[0])),
            }

            const data: *Data = &self.data_storage.items[@intFromEnum(ref.idx)];

            data.* = switch (ref.op) {
                .noop, .external => unreachable,
                .add => self.getData(@bitCast(c[0])) + self.getData(@bitCast(c[1])),
                .mul => self.getData(@bitCast(c[0])) * self.getData(@bitCast(c[1])),
                .exp => @exp(self.getData(@bitCast(c[0]))),
                .neg => -self.getData(@bitCast(c[0])),
                _ => blk: {
                    const num = @as(PowInt, @intCast(@intFromEnum(ref.op) >> op_without_int_size));
                    const d = self.getData(@bitCast(c[0]));
                    var result: Data = one;
                    for (0..@abs(num)) |_| {
                        result *= d;
                    }
                    break :blk if (std.math.sign(num) == -1) one / result else result;
                },
            };
        }

        // recalculate is currently the naive, recursive solution
        // Want to find a better, iterative method
        pub fn forward(self: *Self, ref: ValueRef) void {
            return self.recalculate(ref);
        }
        // can be optimized to not visit leaves
        pub fn backwardWithGrad(self: *Self, ref: ValueRef, grad: Data) void {
            if (self.grad_storage.items.len != self.data_storage.items.len) {
                self.grad_storage.resize(self.allocator, self.data_storage.items.len) catch |err| {
                    std.debug.panic("Failed to resize gradient: {}", .{err});
                };
                self.zeroGrad();
            }

            self.grad_storage.items[@intFromEnum(ref.idx)] = grad;

            var child_list = std.ArrayList(ValueRef).init(self.allocator);
            defer child_list.deinit();

            child_list.append(ref) catch |err| {
                std.debug.panic("Failed to calculate backward for {}: {}", .{ ref, err });
            };

            var i: usize = 0;
            while (i < child_list.items.len) : (i += 1) {
                const curr = child_list.items[i];

                if (curr.op != .noop and curr.op != .external) {
                    const curr_grad = self.getGrad(curr);
                    const children = self.children_storage.items[self.child_idx_map.get(curr.idx).?..];

                    switch (curr.op) {
                        .noop, .external => unreachable,
                        .add => {
                            self.grad_storage.items[@intFromEnum(@as(ValueRef, @bitCast(children[0])).idx)] += curr_grad;
                            self.grad_storage.items[@intFromEnum(@as(ValueRef, @bitCast(children[1])).idx)] += curr_grad;
                        },
                        .mul => {
                            self.grad_storage.items[@intFromEnum(@as(ValueRef, @bitCast(children[0])).idx)] += self.getData(@bitCast(children[1])) * curr_grad;
                            self.grad_storage.items[@intFromEnum(@as(ValueRef, @bitCast(children[1])).idx)] += self.getData(@bitCast(children[0])) * curr_grad;
                        },
                        .exp => {
                            self.grad_storage.items[@intFromEnum(@as(ValueRef, @bitCast(children[0])).idx)] += self.getData(curr) * curr_grad;
                        },
                        .neg => {
                            self.grad_storage.items[@intFromEnum(@as(ValueRef, @bitCast(children[0])).idx)] -= curr_grad;
                        },
                        //.pow
                        _ => {
                            const num = @as(PowInt, @intCast(@intFromEnum(curr.op) >> op_without_int_size));
                            const data_num: Data = if (data_is_scalar) @floatFromInt(num) else @splat(@floatFromInt(num));
                            self.grad_storage.items[@intFromEnum(@as(ValueRef, @bitCast(children[0])).idx)] +=
                                data_num * (self.getData(curr) / self.getData(@bitCast(children[0]))) * curr_grad;
                        },
                    }
                    const children_len: usize = switch (curr.op) {
                        .noop, .external => continue,
                        .exp, .neg => 1,
                        .add, .mul => 2,
                        _ => 1,
                    };
                    child_list.appendSlice(@ptrCast(children[0..children_len])) catch |err| {
                        std.debug.panic("Failed to calculate backward for {} at {}: {}", .{ ref, curr, err });
                    };
                }
            }
        }
        pub fn backward(self: *Self, ref: ValueRef) void {
            self.backwardWithGrad(ref, one);
        }
    };
}

pub const Orientation = enum(u1) { by_row, by_column };
pub fn ManyRef(m: comptime_int, n: comptime_int, canonical: bool) type {
    return struct {
        val_ref: ValueRef,
        // if oriented by row, we store m vectors of size n
        // if oriented by column, we store n vectors of size m
        // The canonical representation is that we store by what there is fewer of
        comptime oriented: Orientation = @enumFromInt(~@as(u1, @bitCast(canonical)) ^ @intFromEnum(@as(Orientation, if (m > n) .by_column else if (m <= n) .by_row))),
        comptime rows: usize = m,
        comptime columns: usize = n,
        inline fn numVectors(self: @This()) comptime_int {
            return comptime switch (self.oriented) {
                .by_row => self.rows,
                .by_column => self.columns,
            };
        }
        inline fn vectorSize(self: @This()) comptime_int {
            return comptime switch (self.oriented) {
                .by_row => self.columns,
                .by_column => self.rows,
            };
        }
    };
}
test "Ensure ManyRef Size" {
    std.debug.assert(@sizeOf(ManyRef(0, 0, true)) == 4);
}
// Assume each value in vector_size is unique
pub fn ManyValueManager(Scalar: type, vector_sizes: []const comptime_int) type {
    switch (@typeInfo(Scalar)) {
        .float => {},
        else => @compileError("Scalar type must be float"),
    }
    comptime var vector_list: [vector_sizes.len]type = undefined;
    inline for (0..vector_sizes.len) |i| {
        vector_list[i] = @Vector(vector_sizes[i], Scalar);
    }

    comptime var vm_list: [vector_sizes.len]type = undefined;
    inline for (0..vector_sizes.len) |i| {
        vm_list[i] = ValueManager(Scalar, vector_sizes[i]);
    }
    const Vms = std.meta.Tuple(&vm_list);

    return struct {
        allocator: std.mem.Allocator,
        vms: Vms,

        const Self = @This();

        // returns the idx for which vm this vector is in
        fn validateVector(Vec: type) comptime_int {
            inline for (vector_list, 0..) |Vector, i| {
                if (Vec == Vector) {
                    return i;
                }
            } else {
                @compileError(std.fmt.comptimePrint("There is no ValueManager for vector {d}", .{Vec}));
            }
        }
        fn refVmIdx(orientation: Orientation, rows: comptime_int, columns: comptime_int) comptime_int {
            return for (vector_sizes, 0..) |size, i| {
                if ((orientation == .by_row and columns == size) or (orientation == .by_column and rows == size)) {
                    break i;
                }
            } else {
                @compileError(std.fmt.comptimePrint("There is no ValueManager for ref with orientation {}, rows {} and columns {}", .{ orientation, rows, columns }));
            };
        }

        fn vectorSize(Vec: type) comptime_int {
            switch (@typeInfo(Vec)) {
                .vector => |v| {
                    switch (@typeInfo(v.child)) {
                        .float => |f| if (f.bits == @bitSizeOf(Scalar)) return v.len else @compileError("Float types must match"),
                        else => @compileError(std.fmt.comptimePrint("{} is not a vector of floats", .{Vec})),
                    }
                },
                else => @compileError(std.fmt.comptimePrint("{} is not a vector of floats", .{Vec})),
            }
        }

        pub fn getData(self: *Self, ref: anytype) switch (ref.oriented) {
            .by_row => [ref.rows]@Vector(ref.columns, Scalar),
            .by_column => [ref.columns]@Vector(ref.rows, Scalar),
        } {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            return self.vms[vm_idx].data_storage.items[@intFromEnum(ref.val_ref.idx)..][0..ref.numVectors()].*;
        }

        pub fn init(a: std.mem.Allocator, capacity: usize) !Self {
            var vms: Vms = undefined;
            inline for (&vms) |*vm| {
                vm.* = try @TypeOf(vm.*).init(a, capacity);
            }
            return .{ .allocator = a, .vms = vms };
        }
        pub fn deinit(self: *Self) void {
            inline for (&self.vms) |*vm| {
                vm.deinit();
            }
        }
        pub fn newRow(self: *Self, row: anytype) ManyRef(1, vectorSize(@TypeOf(row)), true) {
            const vm_idx = validateVector(@TypeOf(row));

            return .{ .val_ref = self.vms[vm_idx].new(row), .oriented = .by_row };
        }
        pub fn manyNewRows(self: *Self, rows: anytype) ManyRef(rows.len, vectorSize(@TypeOf(rows[0])), rows.len < vectorSize(@TypeOf(rows[0]))) {
            inline for (rows[1..], rows[0 .. rows.len - 1]) |v1, v2| {
                if (@TypeOf(v1) != @TypeOf(v2)) @compileError("Expected Array of Same typed vectors");
            }
            const vm_idx = validateVector(@TypeOf(rows[0]));

            const ref = ValueRef{
                .op = .noop,
                .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
            };

            self.vms[vm_idx].data_storage.appendSlice(self.allocator, &rows) catch |err| {
                std.debug.panic("Failed to store data {any}: {}", .{ rows, err });
            };

            return .{ .val_ref = ref, .oriented = .by_row };
        }
        pub fn newColumn(self: *Self, column: anytype) ManyRef(vectorSize(@TypeOf(column)), 1, true) {
            const vm_idx = validateVector(@TypeOf(column));

            return .{ .val_ref = self.vms[vm_idx].new(column), .oriented = .by_column };
        }
        pub fn manyNewColumns(self: *Self, columns: anytype) ManyRef(vectorSize(@TypeOf(columns[0])), columns.len, columns.len <= vectorSize(@TypeOf(columns[0]))) {
            inline for (columns[1..], columns[0 .. columns.len - 1]) |v1, v2| {
                if (@TypeOf(v1) != @TypeOf(v2)) @compileError("Expected Array of Same typed vectors");
            }
            const vm_idx = validateVector(@TypeOf(columns[0]));

            const ref = ValueRef{
                .op = .noop,
                .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
            };

            self.vms[vm_idx].data_storage.appendSlice(self.allocator, &columns) catch |err| {
                std.debug.panic("Failed to store data {any}: {}", .{ columns, err });
            };

            return .{ .val_ref = ref, .oriented = .by_column };
        }

        // User facing functions are type checked but must be canonical
        // This can be ensured by the user using self.reorient
        fn sameRefTypes(ref1: anytype, ref2: anytype) void {
            if (ref1.rows != ref2.rows or ref1.columns != ref2.columns) {
                @compileError(std.fmt.comptimePrint("{} and {} are not the same shape", .{ ref1, ref2 }));
            } else if (ref1.oriented != ref2.oriented) {
                @compileError(std.fmt.comptimePrint("{} and {} are not oriented the same way", .{ ref1, ref2 }));
            }
        }

        pub fn add(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            const vm_id1 = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);
            const vm_id2 = refVmIdx(ref2.oriented, ref2.rows, ref2.columns);

            if (vm_id1 == vm_id2) {
                const result = .{
                    .val_ref = ValueRef{
                        .op = .add,
                        .idx = @enumFromInt(self.vms[vm_id1].data_storage.items.len),
                    },
                    .oriented = ref1.oriented,
                };

                for (0..ref1.numVectors()) |i| {
                    _ = self.vms[vm_id1].add(.{ .op = ref1.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i) }, .{
                        .op = ref2.val_ref.op,
                        .idx = @enumFromInt(@intFromEnum(ref2.val_ref.idx) + i),
                    });
                }

                return result;
            } else unreachable;
        }
        pub fn elemMul(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            const vm_id1 = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);
            const vm_id2 = refVmIdx(ref2.oriented, ref2.rows, ref2.columns);

            if (vm_id1 == vm_id2) {
                const result = .{
                    .val_ref = ValueRef{
                        .op = .add,
                        .idx = @enumFromInt(self.vms[vm_id1].data_storage.items.len),
                    },
                    .oriented = ref1.oriented,
                };

                for (0..ref1.numVectors()) |i| {
                    _ = self.vms[vm_id1].mul(.{ .op = ref1.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i) }, .{
                        .op = ref2.val_ref.op,
                        .idx = @enumFromInt(@intFromEnum(ref2.val_ref.idx) + i),
                    });
                }

                return result;
            } else unreachable;
        }
        pub fn sumRows(self: *Self, ref: anytype) ManyRef(ref.rows, 1, true) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, true));

            var vector: @Vector(ref.rows, Scalar) = undefined;
            for (self.getData(ref)) |vec| {
                for (0..ref.rows) |i| {
                    vector[i] = @reduce(.Add, vec);
                }
            }
            // const vm_idx = validateVector(@TypeOf(vector));
            // const vm_ptr = &self.vms[vm_idx];

            // TODO Need to store children for backward
            // TODO Need to store external operator
            //  for external ops, children are not only ValueRefs (probably change ValueManager to reflect this)
            //  children_storage stores 32bit (or 64bit) blocks that can be reinterpreted
            //  we can store pointers to other vms, external operators, sizes, and ValueRefs from other vms

            return self.newColumn(vector);
        }
        pub fn sumColumns(self: *Self, ref: anytype) ManyRef(1, ref.columns, true) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, true));

            var vector: @Vector(ref.columns, Scalar) = undefined;
            for (self.getData(ref)) |vec| {
                for (0..ref.columns) |i| {
                    vector[i] = @reduce(.Add, vec);
                }
            }
            var result = self.newRow(vector);
            result.val_ref.op = .external;

            // TODO Need to store children for backward
            // TODO Need to store external operator

            return result;
        }
    };
}
