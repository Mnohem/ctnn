const std = @import("std");

// power operator is hidden in the unused (_) bits of the Operator enum
// unused bits are interpreted as a signed int to be raised to
// .external means this value came from an operation in ManyValueManager
pub const Operator = enum(i8) { noop = 1, add, mul, max, exp, neg, external_sum, external_max, external_splat, _ };
pub const Idx = enum(u24) { _ };
// Index is crammed with Operator, meaning our index is 24 bit
// Thus we can only store 16,777,215 vectors
pub const ValueRef = packed struct {
    op: Operator,
    idx: Idx,
};

const LOWEST_POWER_SIZE = 4;
pub const op_without_int_size: comptime_int = @ceil(@log2(@as(comptime_float, @floatFromInt(std.meta.fields(Operator).len))));
const size_for_int = 8 - op_without_int_size;
const _ = if (size_for_int < LOWEST_POWER_SIZE) @compileError(std.fmt.comptimePrint("Operator enum is too large to store power int: {d} bits left", .{size_for_int}));
pub const PowInt = std.meta.Int(.signed, size_for_int);
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
        // stores tuple index of this vm in the mvm and other children for external operations
        children_storage: std.ArrayListUnmanaged(u32),
        expr_lists: std.AutoArrayHashMapUnmanaged(ValueRef, []ValueRef),

        const Self = @This();

        pub fn init(a: std.mem.Allocator, capacity: usize) !Self {
            return Self{
                .allocator = a,
                .data_storage = try std.ArrayListUnmanaged(Data).initCapacity(a, capacity),
                .grad_storage = std.ArrayListUnmanaged(Data){},
                .child_idx_map = std.AutoArrayHashMapUnmanaged(Idx, usize){},
                .children_storage = std.ArrayListUnmanaged(u32){},
                .expr_lists = std.AutoArrayHashMapUnmanaged(ValueRef, []ValueRef){},
            };
        }
        pub fn deinit(self: *Self) void {
            self.data_storage.deinit(self.allocator);
            self.grad_storage.deinit(self.allocator);
            self.child_idx_map.deinit(self.allocator);
            self.children_storage.deinit(self.allocator);
            for (self.expr_lists.values()) |val| {
                self.allocator.free(val);
            }
            self.expr_lists.deinit(self.allocator);
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
        // Calling this function is only correct after using backward
        pub fn getGrad(self: *const Self, ref: ValueRef) Data {
            return self.grad_storage.items[@intFromEnum(ref.idx)];
        }
        pub fn newExpr(self: *Self, op: Operator, data: Data, children: []const u32) !ValueRef {
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
        pub fn max(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.newExpr(.add, @max(self.getData(id1), self.getData(id2)), &[_]u32{ @bitCast(id1), @bitCast(id2) }) catch |err| {
                std.debug.panic("Failed to add refs {} and {}: {}", .{ id1, id2, err });
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

        pub fn zeroGrad(self: *Self) void {
            @memset(self.grad_storage.items, std.mem.zeroes(Data));
        }

        // returns the parts of this expression that originate from external operations
        // returns slice that must be freed by caller
        pub fn externalParts(self: *const Self, ref: ValueRef) []ValueRef {
            var externals = std.ArrayList(ValueRef).init(self.allocator);
            var to_travel = std.ArrayList(ValueRef).init(self.allocator);
            defer to_travel.deinit();

            var curr_ref = ref;
            op: switch (curr_ref.op) {
                .external_sum, .external_max, .external_splat => {
                    externals.append(curr_ref) catch unreachable;
                    continue :op .noop;
                },
                .noop => if (to_travel.pop()) |r| {
                    curr_ref = r;
                    continue :op curr_ref.op;
                } else return externals.toOwnedSlice() catch unreachable,

                .add, .mul, .max => {
                    const c = self.children_storage.items[self.child_idx_map.get(curr_ref.idx).?..];
                    to_travel.append(@bitCast(c[1])) catch unreachable;
                    curr_ref = @bitCast(c[0]);
                    continue :op curr_ref.op;
                },
                _ => continue :op .neg,
                .exp, .neg => {
                    const c = self.children_storage.items[self.child_idx_map.get(curr_ref.idx).?..];
                    curr_ref = @bitCast(c[0]);
                    continue :op curr_ref.op;
                },
            }
        }

        pub fn getLocalChildren(self: *Self, ref: ValueRef) []ValueRef {
            const num_children: usize = switch (ref.op) {
                .noop, .external_sum, .external_splat, .external_max => @panic(".noop and external operators have no local children"),
                .add, .mul, .max => 2,
                _ => 1,
                .exp, .neg => 1,
            };
            return @ptrCast(self.children_storage.items[self.child_idx_map.get(ref.idx).?..][0..num_children]);
        }
        pub fn getExternalInfo(self: *const Self, ref: ValueRef) struct { u32, ValueRef, []u32 } {
            switch (ref.op) {
                .external_sum, .external_max, .external_splat => {
                    const children_slice = self.children_storage.items[self.child_idx_map.get(ref.idx).?..];
                    return .{ children_slice[0], @bitCast(children_slice[1]), children_slice[2..] };
                },
                else => @panic("Only external operators have external info"),
            }
        }
        pub fn recalculateData(self: *Self, ref: ValueRef) void {
            std.debug.assert(switch (ref.op) {
                .noop, .external_sum, .external_max, .external_splat => false,
                else => true,
            });
            const c = self.getLocalChildren(ref);
            const data = self.getDataPtr(ref);

            data.* = switch (ref.op) {
                .noop, .external_sum, .external_splat, .external_max => unreachable,
                .add => self.getData(c[0]) + self.getData(c[1]),
                .mul => self.getData(c[0]) * self.getData(c[1]),
                .max => @max(self.getData(c[0]), self.getData(c[1])),
                .exp => @exp(self.getData(c[0])),
                .neg => -self.getData(c[0]),
                _ => blk: {
                    const num = @as(PowInt, @intCast(@intFromEnum(ref.op) >> op_without_int_size));
                    const d = self.getData(c[0]);
                    var result: Data = one;
                    for (0..@abs(num)) |_| {
                        result *= d;
                    }
                    break :blk if (std.math.sign(num) == -1) one / result else result;
                },
            };
        }

        // TODO ability to hold multiple expressions
        pub fn createExprList(self: *Self, ref: ValueRef) !void {
            std.debug.assert(switch (ref.op) {
                .noop, .external_sum, .external_max, .external_splat => false,
                else => true,
            });
            std.debug.assert(self.expr_lists.get(ref) == null);

            var expr_list = std.ArrayList(ValueRef).init(self.allocator);
            try expr_list.append(ref);

            var i: isize = 0;
            while (i < expr_list.items.len) : (i += 1) {
                switch (expr_list.items[@intCast(i)].op) {
                    .noop, .external_sum, .external_max, .external_splat => {
                        _ = expr_list.orderedRemove(@intCast(i));
                        i -= 1;
                    },
                    else => try expr_list.appendSlice(self.getLocalChildren(expr_list.items[@intCast(i)])),
                }
            }

            try self.expr_lists.putNoClobber(self.allocator, ref, try expr_list.toOwnedSlice());
        }

        pub fn forward(self: *Self, ref: ValueRef) !void {
            switch (ref.op) {
                .noop, .external_sum, .external_max, .external_splat => return,
                else => {},
            }

            const expr_list = if (self.expr_lists.get(ref)) |expr_list| expr_list else blk: {
                try self.createExprList(ref);
                break :blk self.expr_lists.get(ref).?;
            };

            for (0..expr_list.len) |i| {
                self.recalculateData(expr_list[expr_list.len - i - 1]);
            }
        }

        pub fn recalculateGrad(self: *Self, ref: ValueRef) void {
            std.debug.assert(switch (ref.op) {
                .noop, .external_sum, .external_max, .external_splat => false,
                else => true,
            });
            const c = self.getLocalChildren(ref);
            const curr_grad = self.getGrad(ref);

            switch (ref.op) {
                .noop, .external_sum, .external_max, .external_splat => unreachable,
                .add => {
                    self.grad_storage.items[@intFromEnum(c[0].idx)] += curr_grad;
                    self.grad_storage.items[@intFromEnum(c[1].idx)] += curr_grad;
                },
                .mul => {
                    self.grad_storage.items[@intFromEnum(c[0].idx)] += self.getData(c[1]) * curr_grad;
                    self.grad_storage.items[@intFromEnum(c[1].idx)] += self.getData(c[0]) * curr_grad;
                },
                .max => {
                    const selected0 = self.getData(c[0]) == self.getData(ref);
                    if (data_is_scalar) {
                        self.grad_storage.items[@intFromEnum(c[0].idx)] += if (selected0) curr_grad else 0;
                        self.grad_storage.items[@intFromEnum(c[1].idx)] += if (!selected0) curr_grad else 0;
                    } else {
                        self.grad_storage.items[@intFromEnum(c[0].idx)] += @select(Scalar, selected0, curr_grad, @as(Data, @splat(0)));
                        self.grad_storage.items[@intFromEnum(c[1].idx)] += @select(Scalar, selected0, @as(Data, @splat(0)), curr_grad);
                    }
                },
                .exp => self.grad_storage.items[@intFromEnum(c[0].idx)] += self.getData(ref) * curr_grad,
                .neg => self.grad_storage.items[@intFromEnum(c[0].idx)] -= curr_grad,
                _ => {
                    const num = @as(PowInt, @intCast(@intFromEnum(ref.op) >> op_without_int_size));
                    const data_num: Data = if (data_is_scalar) @floatFromInt(num) else @splat(@floatFromInt(num));
                    self.grad_storage.items[@intFromEnum(c[0].idx)] += data_num * (self.getData(ref) / self.getData(c[0])) * curr_grad;
                },
            }
        }

        pub fn backwardWithGrad(self: *Self, ref: ValueRef, grad: ?Data) !void {
            switch (ref.op) {
                .noop, .external_sum, .external_max, .external_splat => return,
                else => {},
            }

            if (self.grad_storage.items.len != self.data_storage.items.len) {
                self.grad_storage.resize(self.allocator, self.data_storage.items.len) catch |err| {
                    std.debug.panic("Failed to resize gradient: {}", .{err});
                };
                self.zeroGrad();
            }

            if (grad) |g| {
                self.grad_storage.items[@intFromEnum(ref.idx)] = g;
            }

            const expr_list = if (self.expr_lists.get(ref)) |expr_list| expr_list else blk: {
                try self.createExprList(ref);
                break :blk self.expr_lists.get(ref).?;
            };

            for (expr_list) |val_ref| {
                self.recalculateGrad(val_ref);
            }
        }
        pub fn backward(self: *Self, ref: ValueRef) void {
            self.backwardWithGrad(ref, one) catch unreachable;
        }
    };
}
