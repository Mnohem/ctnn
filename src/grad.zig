const std = @import("std");

pub fn ValueManager(Data: type, one_value: Data) type {
    const Idx = enum(u32) { _ };
    // power operator is hidden in the unused bits of the Operator enum
    // unused bits are interpreted as a signed int to be raised to
    const Operator = enum(i8) { plus, mul, exp, _ };
    const op_without_int_size: comptime_int = @ceil(@log2(@as(comptime_float, @floatFromInt(std.meta.fields(Operator).len))));
    const size_for_int = 8 - op_without_int_size;
    if (size_for_int <= 4) @compileError(std.fmt.comptimePrint("Operator enum is too large to store int: {d} bits left", .{size_for_int}));
    const PowType = std.meta.Int(.signed, size_for_int);

    return struct {
        allocator: std.mem.Allocator,
        data_storage: std.ArrayListUnmanaged(Data),
        grad_storage: std.ArrayListUnmanaged(Data),
        op_map: std.AutoArrayHashMapUnmanaged(Idx, Operator),
        unary_map: std.AutoArrayHashMapUnmanaged(Idx, Idx),
        binary_map: std.AutoArrayHashMapUnmanaged(Idx, [2]Idx),
        // May need for operations with variable number of args
        // extra_children_storage: std.ArrayListUnmanaged(Idx),

        const Self = @This();

        pub fn init(a: std.mem.Allocator) !Self {
            return Self{
                .allocator = a,
                .data_storage = try std.ArrayListUnmanaged(Data).initCapacity(a, 10),
                .grad_storage = try std.ArrayListUnmanaged(Data).initCapacity(a, 10),
                .op_map = try std.AutoArrayHashMapUnmanaged(Idx, Operator).init(a, &[_]Idx{}, &[_]Operator{}),
                .unary_map = try std.AutoArrayHashMapUnmanaged(Idx, Idx).init(a, &[_]Idx{}, &[_]Idx{}),
                .binary_map = try std.AutoArrayHashMapUnmanaged(Idx, [2]Idx).init(a, &[_]Idx{}, &[_][2]Idx{}),
            };
        }
        pub fn deinit(self: *Self) void {
            self.data_storage.deinit(self.allocator);
            self.grad_storage.deinit(self.allocator);
            self.op_map.deinit(self.allocator);
            self.unary_map.deinit(self.allocator);
            self.binary_map.deinit(self.allocator);
        }
        pub fn new(self: *Self, data: Data) Idx {
            self.data_storage.append(self.allocator, data) catch |err| {
                std.debug.panic("Failed to store data {}: {}", .{ data, err });
            };
            errdefer _ = self.data_storage.pop();

            self.grad_storage.append(self.allocator, std.mem.zeroes(Data)) catch |err| {
                std.debug.panic("Failed to zero gradient {}: {}", .{ data, err });
            };
            errdefer _ = self.grad_storage.pop();

            return @enumFromInt(self.data_storage.items.len - 1);
        }
        // Idx is always valid assuming the caller did cast into it
        fn getData(self: *const Self, idx: Idx) Data {
            return self.data_storage.items[@intFromEnum(idx)];
        }
        fn getGrad(self: *const Self, idx: Idx) Data {
            return self.grad_storage.items[@intFromEnum(idx)];
        }
        fn bin(self: *Self, id1: Idx, id2: Idx, data: Data, op: Operator) !Idx {
            const new_idx = self.new(data);
            errdefer {
                _ = self.data_storage.pop();
                _ = self.grad_storage.pop();
            }

            try self.op_map.put(self.allocator, new_idx, op);
            errdefer _ = self.op_map.swapRemove(new_idx);
            try self.binary_map.put(self.allocator, new_idx, [_]Idx{ id1, id2 });

            return new_idx;
        }
        fn un(self: *Self, idx: Idx, data: Data, op: Operator) !Idx {
            const new_idx = self.new(data);
            errdefer {
                _ = self.data_storage.pop();
                _ = self.grad_storage.pop();
            }

            try self.op_map.put(self.allocator, new_idx, op);
            errdefer _ = self.op_map.swapRemove(new_idx);
            try self.unary_map.put(self.allocator, new_idx, idx);
            errdefer _ = self.binary_map.swapRemove(new_idx);

            return new_idx;
        }

        pub fn add(self: *Self, id1: Idx, id2: Idx) Idx {
            return self.bin(id1, id2, self.getData(id1) + self.getData(id2), .plus) catch |err| {
                std.debug.panic("Failed to add indices {d} and {d}: {}", .{ id1, id2, err });
            };
        }
        pub fn mul(self: *Self, id1: Idx, id2: Idx) Idx {
            return self.bin(id1, id2, self.getData(id1) * self.getData(id2), .mul) catch |err| {
                std.debug.panic("Failed to multiply indices {d} and {d}: {}", .{ id1, id2, err });
            };
        }
        pub fn div(self: *Self, id1: Idx, id2: Idx) Idx {
            const recip = self.pow(id2, -1);

            return self.mul(id1, recip);
        }
        // pow cannot be raised to a Value power because this changes derivs, we will have more complications
        // Though we still need to hold the num for backward, so we cram it into the Operator enum
        // PowType is type ix where x is the number of unused bits in Operator
        // We cant store PowType as a Value since it can't be a vector
        pub fn pow(self: *Self, idx: Idx, num: PowType) Idx {
            const data = self.getData(idx);
            var result: Data = data;
            for (0..@abs(num) - 1) |_| {
                result *= data;
            }
            result = if (std.math.sign(num) == -1) one_value / result else if (num == 0) one_value else result;
            return self.un(idx, result, @enumFromInt(num << op_without_int_size)) catch |err| {
                std.debug.panic("Failed to raise index {d} to the power of {d}: {}", .{ idx, num, err });
            };
        }
        pub fn exp(self: *Self, idx: Idx) Idx {
            return self.un(idx, @exp(self.getData(idx)), .exp) catch |err| {
                std.debug.panic("Failed to exponentiate index {d}: {}", .{ idx, err });
            };
        }

        pub fn backward(self: *Self, idx: Idx) void {
            self.grad_storage.items[@intFromEnum(idx)] = one_value;

            var child_list = std.ArrayList(Idx).init(self.allocator);
            defer child_list.deinit();

            child_list.append(idx) catch |err| {
                std.debug.panic("Failed to calculate backward for {d}: {}", .{ idx, err });
            };

            var i: usize = 0;
            while (i < child_list.items.len) : (i += 1) {
                const curr = child_list.items[i];

                if (self.op_map.get(curr)) |op| {
                    const curr_grad = self.grad_storage.items[@intFromEnum(curr)];
                    const to_add = blk: switch (op) {
                        .plus => {
                            const children = self.binary_map.get(curr).?;
                            self.grad_storage.items[@intFromEnum(children[0])] += curr_grad;
                            self.grad_storage.items[@intFromEnum(children[1])] += curr_grad;

                            break :blk &children;
                        },
                        .mul => {
                            const children = self.binary_map.get(curr).?;
                            const d0 = &self.data_storage.items[@intFromEnum(children[0])];
                            const d1 = &self.data_storage.items[@intFromEnum(children[1])];

                            self.grad_storage.items[@intFromEnum(children[0])] += d1.* * curr_grad;
                            self.grad_storage.items[@intFromEnum(children[1])] += d0.* * curr_grad;

                            break :blk &children;
                        },
                        .exp => {
                            const child = self.unary_map.get(curr).?;
                            self.grad_storage.items[@intFromEnum(child)] += self.getData(curr) * curr_grad;

                            break :blk &[1]Idx{child};
                        },
                        //.pow
                        _ => {
                            const child = self.unary_map.get(curr).?;
                            const num = @as(i8, @intFromEnum(op)) >> op_without_int_size;
                            const data_num: Data = if (Data == f32 or Data == f64) @floatFromInt(num) else @splat(@floatFromInt(num));
                            self.grad_storage.items[@intFromEnum(child)] += data_num * (self.getData(curr) / self.getData(child)) * curr_grad;

                            break :blk &[1]Idx{child};
                        },
                    };
                    child_list.appendSlice(to_add) catch |err| {
                        std.debug.panic("Failed to calculate backward for {d} on idx {d}: {}", .{ idx, curr, err });
                    };
                }
            }
        }
    };
}

const APPROX = 0.001;

test "ValueManager Operations and Duplicate Terms Test" {
    var vm = try ValueManager(f32, 1).init(std.testing.allocator);
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

test "ValueManager Exponentiation Test" {
    var vm = try ValueManager(f32, 1).init(std.testing.allocator);
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
    var vm = try ValueManager(f32, 1).init(std.testing.allocator);
    defer vm.deinit();

    const a = vm.new(2);
    const b = vm.new(3);

    const c = vm.pow(a, 3);
    try std.testing.expectApproxEqAbs(8, vm.getData(c), APPROX);

    const d = vm.pow(b, 2);
    try std.testing.expectApproxEqAbs(9, vm.getData(d), APPROX);

    const e = vm.add(c, d);
    try std.testing.expectApproxEqAbs(17, vm.getData(e), APPROX);
    vm.backward(e);

    try std.testing.expectApproxEqAbs(12, vm.getGrad(a), APPROX);
    try std.testing.expectApproxEqAbs(6, vm.getGrad(b), APPROX);
}

test "ValueManager Division Test" {
    var vm = try ValueManager(f32, 1).init(std.testing.allocator);
    defer vm.deinit();

    const a = vm.new(2);
    const b = vm.new(3);

    const c = vm.pow(a, -1);
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
    var vm = try ValueManager(@Vector(4, f32), @splat(1)).init(std.testing.allocator);
    defer vm.deinit();

    const a = vm.new([_]f32{ 1, 2, 3, 4 });
    const b = vm.new(@splat(5));
    const c = vm.new([_]f32{ 0.09, 2, 10, 1.1 });

    const d = vm.pow(a, 2);
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
