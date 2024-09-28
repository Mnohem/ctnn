const std = @import("std");

// Require caller to give a one_value for easy interop with Vectors
pub fn ValueManager(Data: type) type {
    const data_is_scalar = s: switch (@typeInfo(Data)) {
        .float => true,
        .vector => |v| switch (@typeInfo(v.child)) {
            .float => false,
            else => continue :s null,
        },
        else => @compileError(std.fmt.comptimePrint("Expected a float scalar or vector type, found {} instead", .{Data})),
    };
    const one: Data = if (data_is_scalar) 1 else @splat(1);
    // const one_half: Data = one / (one + one);

    // The lowest amount of bits Im willing to give to power
    const LOWEST_POWER_SIZE = 4;
    // power operator is hidden in the unused bits of the Operator enum
    // unused bits are interpreted as a signed int to be raised to
    const Operator = enum(i8) { noop = 0, add, mul, exp, neg, _ };
    const op_without_int_size: comptime_int = @ceil(@log2(@as(comptime_float, @floatFromInt(std.meta.fields(Operator).len))));
    const size_for_int = 8 - op_without_int_size;
    if (size_for_int < LOWEST_POWER_SIZE) @compileError(std.fmt.comptimePrint("Operator enum is too large to store int: {d} bits left", .{size_for_int}));
    const PowInt = std.meta.Int(.signed, size_for_int);

    const Idx = enum(u24) { _ };

    comptime std.debug.assert(@sizeOf(packed struct {
        op: Operator,
        idx: Idx,
    }) == 4);

    return struct {
        allocator: std.mem.Allocator,
        // indexed by Idx Type
        data_storage: std.ArrayListUnmanaged(Data),
        grad_storage: std.ArrayListUnmanaged(Data),
        child_idx_map: std.AutoArrayHashMapUnmanaged(Idx, u24),
        // indexed by u24 returned from child_idx_map
        children_storage: std.ArrayListUnmanaged(ValueRef),

        const Self = @This();
        // Index is crammed with Operator, meaning our index is 24 bit
        // Thus we can only store 16,777,215 values
        pub const ValueRef = packed struct {
            op: Operator,
            idx: Idx,
        };

        pub fn init(a: std.mem.Allocator, capacity: usize) !Self {
            return Self{
                .allocator = a,
                .data_storage = try std.ArrayListUnmanaged(Data).initCapacity(a, capacity),
                .grad_storage = try std.ArrayListUnmanaged(Data).initCapacity(a, capacity),
                .child_idx_map = try std.AutoArrayHashMapUnmanaged(Idx, u24).init(a, &[_]Idx{}, &[_]u24{}),
                .children_storage = try std.ArrayListUnmanaged(ValueRef).initCapacity(a, capacity),
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

            self.grad_storage.append(self.allocator, std.mem.zeroes(Data)) catch |err| {
                std.debug.panic("Failed to zero gradient {}: {}", .{ data, err });
            };

            return .{
                .op = .noop,
                .idx = @enumFromInt(self.data_storage.items.len - 1),
            };
        }
        // Idx is always valid assuming the caller did cast into it
        fn getData(self: *const Self, ref: ValueRef) Data {
            return self.data_storage.items[@intFromEnum(ref.idx)];
        }
        fn getGrad(self: *const Self, ref: ValueRef) Data {
            return self.grad_storage.items[@intFromEnum(ref.idx)];
        }
        fn newExpr(self: *Self, op: Operator, data: Data, children: []const ValueRef) !ValueRef {
            var new_ref = self.new(data);
            errdefer {
                _ = self.data_storage.pop();
                _ = self.grad_storage.pop();
            }
            new_ref.op = op;

            try self.child_idx_map.put(self.allocator, new_ref.idx, @intCast(self.children_storage.items.len));
            errdefer _ = self.child_idx_map.swapRemove(new_ref.idx);
            try self.children_storage.appendSlice(self.allocator, children);

            return new_ref;
        }

        pub fn add(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.newExpr(.add, self.getData(id1) + self.getData(id2), &[_]ValueRef{ id1, id2 }) catch |err| {
                std.debug.panic("Failed to add refs {} and {}: {}", .{ id1, id2, err });
            };
        }
        pub fn mul(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.newExpr(.mul, self.getData(id1) * self.getData(id2), &[_]ValueRef{ id1, id2 }) catch |err| {
                std.debug.panic("Failed to add refs {} and {}: {}", .{ id1, id2, err });
            };
        }
        pub inline fn div(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
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
            return self.newExpr(@enumFromInt(@as(i8, num) << op_without_int_size), result, &[_]ValueRef{ref}) catch |err| {
                std.debug.panic("Failed to raise ref {} to the power of {d}: {}", .{ ref, num, err });
            };
        }
        pub fn exp(self: *Self, ref: ValueRef) ValueRef {
            return self.newExpr(.exp, @exp(self.getData(ref)), &[_]ValueRef{ref}) catch |err| {
                std.debug.panic("Failed to exponentiate ref {}: {}", .{ ref, err });
            };
        }
        pub fn neg(self: *Self, ref: ValueRef) ValueRef {
            return self.newExpr(.neg, -self.getData(ref), &[_]ValueRef{ref}) catch |err| {
                std.debug.panic("Failed to negate ref {}: {}", .{ ref, err });
            };
        }
        pub inline fn sub(self: *Self, id1: ValueRef, id2: ValueRef) ValueRef {
            return self.add(id1, self.neg(id2));
        }

        // pub fn eval(self: *Self, expr: []const u8, variables: anytype) Idx {
        //     return @import("eval.zig").eval(Data, self, expr, variables);
        // }

        // can be optimized to not visit leaves
        pub fn backward(self: *Self, ref: ValueRef) void {
            self.grad_storage.items[@intFromEnum(ref.idx)] = one;

            var child_list = std.ArrayList(ValueRef).init(self.allocator);
            defer child_list.deinit();

            child_list.append(ref) catch |err| {
                std.debug.panic("Failed to calculate backward for {}: {}", .{ ref, err });
            };

            var i: usize = 0;
            while (i < child_list.items.len) : (i += 1) {
                const curr = child_list.items[i];

                if (curr.op != .noop) {
                    const curr_grad = self.getGrad(curr);
                    const child_idx: usize = @intCast(self.child_idx_map.get(curr.idx).?);

                    const to_add = blk: switch (curr.op) {
                        .noop => unreachable,
                        .add => {
                            const children = self.children_storage.items[child_idx .. child_idx + 2];
                            self.grad_storage.items[@intFromEnum(children[0].idx)] += curr_grad;
                            self.grad_storage.items[@intFromEnum(children[1].idx)] += curr_grad;

                            break :blk children;
                        },
                        .mul => {
                            const children = self.children_storage.items[child_idx .. child_idx + 2];
                            self.grad_storage.items[@intFromEnum(children[0].idx)] += self.getData(children[1]) * curr_grad;
                            self.grad_storage.items[@intFromEnum(children[1].idx)] += self.getData(children[0]) * curr_grad;

                            break :blk children;
                        },
                        .exp => {
                            const child = self.children_storage.items[child_idx .. child_idx + 1];
                            self.grad_storage.items[@intFromEnum(child[0].idx)] += self.getData(curr) * curr_grad;

                            break :blk child;
                        },
                        .neg => {
                            const child = self.children_storage.items[child_idx .. child_idx + 1];
                            self.grad_storage.items[@intFromEnum(child[0].idx)] -= curr_grad;

                            break :blk child;
                        },
                        //.pow
                        _ => {
                            const child = self.children_storage.items[child_idx .. child_idx + 1];
                            const num = @as(PowInt, @intCast(@intFromEnum(curr.op) >> op_without_int_size));
                            const data_num: Data = if (data_is_scalar) @floatFromInt(num) else @splat(@floatFromInt(num));
                            self.grad_storage.items[@intFromEnum(child[0].idx)] += data_num * (self.getData(curr) / self.getData(child[0])) * curr_grad;

                            break :blk child;
                        },
                    };
                    child_list.appendSlice(to_add) catch |err| {
                        std.debug.panic("Failed to calculate backward for {} at {}: {}", .{ ref, curr, err });
                    };
                }
            }
        }
    };
}

const APPROX = 0.001;

test "ValueManager Operations and Duplicate Terms Test" {
    var vm = try ValueManager(f32).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(1);
    const b = vm.new(2);
    const c = vm.new(3);

    // f = c * e
    //   = c * ad
    //   = ca(a + b)
    // 9 = a^2c + abc [a := 1, b := 2, c := 3]
    //f'a= 2ac + bc [a := 1, b := 2, c := 3]
    const d = vm.add(a, b);
    try std.testing.expectEqual(vm.getData(c), vm.getData(d));

    const e = vm.mul(a, d);
    try std.testing.expectEqual(vm.getData(c), vm.getData(e));

    const f = vm.mul(c, e);
    vm.backward(f);
    try std.testing.expectApproxEqAbs(9, vm.getData(f), APPROX);

    try std.testing.expectApproxEqAbs(1, vm.getGrad(f), APPROX);
    try std.testing.expectApproxEqAbs(3, vm.getGrad(e), APPROX);
    try std.testing.expectApproxEqAbs(3, vm.getGrad(c), APPROX);
    try std.testing.expectApproxEqAbs(3, vm.getGrad(d), APPROX);
    try std.testing.expectApproxEqAbs(3, vm.getGrad(b), APPROX);
    try std.testing.expectApproxEqAbs(12, vm.getGrad(a), APPROX);
}

test "ValueManager Negation Test" {
    var vm = try ValueManager(f32).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(1);
    const b = vm.new(2);

    const d = vm.neg(b);
    const e = vm.add(a, d);
    try std.testing.expectApproxEqAbs(-1, vm.getData(e), APPROX);

    vm.backward(e);
    try std.testing.expectApproxEqAbs(-1, vm.getGrad(b), APPROX);
}

test "ValueManager Exponentiation Test" {
    var vm = try ValueManager(f32).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(@log(2.0));
    const b = vm.new(@log(2.0));

    const c = vm.add(a, b);
    try std.testing.expectApproxEqAbs(2 * @log(2.0), vm.getData(c), APPROX);

    const d = vm.exp(c);
    try std.testing.expectApproxEqAbs(4, vm.getData(d), APPROX);

    vm.backward(d);
    try std.testing.expectApproxEqAbs(4, vm.getGrad(a), APPROX);
    try std.testing.expectApproxEqAbs(4, vm.getGrad(b), APPROX);
    try std.testing.expectApproxEqAbs(4, vm.getGrad(c), APPROX);
}

test "ValueManager Power Raising Test" {
    var vm = try ValueManager(f32).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(2);
    const b = vm.new(3);

    const c = vm.powi(a, 3);
    try std.testing.expectApproxEqAbs(8, vm.getData(c), APPROX);

    const d = vm.powi(b, 2);
    try std.testing.expectApproxEqAbs(9, vm.getData(d), APPROX);

    const e = vm.add(c, d);
    try std.testing.expectApproxEqAbs(17, vm.getData(e), APPROX);
    vm.backward(e);

    try std.testing.expectApproxEqAbs(12, vm.getGrad(a), APPROX);
    try std.testing.expectApproxEqAbs(6, vm.getGrad(b), APPROX);
}

test "ValueManager Division Test" {
    var vm = try ValueManager(f32).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(2);
    const b = vm.new(3);

    const c = vm.powi(a, -1);
    try std.testing.expectApproxEqAbs(0.5, vm.getData(c), APPROX);

    const d = vm.div(b, a);
    try std.testing.expectApproxEqAbs(1.5, vm.getData(d), APPROX);

    const e = vm.mul(c, d);
    try std.testing.expectApproxEqAbs(0.75, vm.getData(e), APPROX);
    vm.backward(e);

    try std.testing.expectApproxEqAbs(-0.75, vm.getGrad(a), APPROX);
    try std.testing.expectApproxEqAbs(0.25, vm.getGrad(b), APPROX);
}

test "ValueManager Vector Test" {
    var vm = try ValueManager(@Vector(4, f32)).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new([_]f32{ 1, 2, 3, 4 });
    const b = vm.new(@splat(5));
    const c = vm.new([_]f32{ 0.09, 2, 10, 1.1 });

    const d = vm.powi(a, 2);
    try std.testing.expectApproxEqAbs(1, vm.getData(d)[0], APPROX);
    try std.testing.expectApproxEqAbs(4, vm.getData(d)[1], APPROX);
    try std.testing.expectApproxEqAbs(9, vm.getData(d)[2], APPROX);
    try std.testing.expectApproxEqAbs(16, vm.getData(d)[3], APPROX);

    const e = vm.div(d, b);
    try std.testing.expectApproxEqAbs(1.0 / 5.0, vm.getData(e)[0], APPROX);
    try std.testing.expectApproxEqAbs(4.0 / 5.0, vm.getData(e)[1], APPROX);
    try std.testing.expectApproxEqAbs(9.0 / 5.0, vm.getData(e)[2], APPROX);
    try std.testing.expectApproxEqAbs(16.0 / 5.0, vm.getData(e)[3], APPROX);

    const f = vm.add(e, c);
    try std.testing.expectApproxEqAbs(1.0 / 5.0 + 0.09, vm.getData(f)[0], APPROX);
    try std.testing.expectApproxEqAbs(4.0 / 5.0 + 2.0, vm.getData(f)[1], APPROX);
    try std.testing.expectApproxEqAbs(9.0 / 5.0 + 10.0, vm.getData(f)[2], APPROX);
    try std.testing.expectApproxEqAbs(16.0 / 5.0 + 1.1, vm.getData(f)[3], APPROX);

    vm.backward(f);

    try std.testing.expectEqual(@as(@Vector(4, f32), @splat(1)), vm.getGrad(c));

    const a_over_b = vm.getData(a) / vm.getData(b);
    const df_db = -a_over_b * a_over_b;

    try std.testing.expectApproxEqAbs(df_db[0], vm.getGrad(b)[0], APPROX);
    try std.testing.expectApproxEqAbs(df_db[1], vm.getGrad(b)[1], APPROX);
    try std.testing.expectApproxEqAbs(df_db[2], vm.getGrad(b)[2], APPROX);
    try std.testing.expectApproxEqAbs(df_db[3], vm.getGrad(b)[3], APPROX);

    const df_da = a_over_b + a_over_b;

    try std.testing.expectApproxEqAbs(df_da[0], vm.getGrad(a)[0], APPROX);
    try std.testing.expectApproxEqAbs(df_da[1], vm.getGrad(a)[1], APPROX);
    try std.testing.expectApproxEqAbs(df_da[2], vm.getGrad(a)[2], APPROX);
    try std.testing.expectApproxEqAbs(df_da[3], vm.getGrad(a)[3], APPROX);
}

test "ValueManager Square Test" {
    var vm = try ValueManager(f32).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(2);

    const b = vm.powi(a, 2);
    // try std.testing.expectApproxEqAbs(4, vm.getData(b), APPROX);
    try std.testing.expectApproxEqAbs(4, vm.getData(b), APPROX);

    vm.backward(b);

    try std.testing.expectApproxEqAbs(4, vm.getGrad(a), APPROX);
}
