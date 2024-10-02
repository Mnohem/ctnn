const std = @import("std");
const ManyValueManager = @import("grad.zig").ManyValueManager;
const ValueManager = @import("grad.zig").ValueManager;

const APPROX = 0.001;

test "ValueManager Operations and Duplicate Terms Test" {
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(1);
    const b = vm.new(2);
    const c = vm.new(3);
    try std.testing.expectEqual(.noop, a.op);
    try std.testing.expectEqual(.noop, b.op);
    try std.testing.expectEqual(.noop, c.op);
    try std.testing.expectEqual(0b00000000, @as(i8, @intFromEnum(a.op)));
    try std.testing.expectEqual(0b00000000, @as(i8, @intFromEnum(b.op)));
    try std.testing.expectEqual(0b00000000, @as(i8, @intFromEnum(c.op)));
    // f = c * e
    //   = c * ad
    //   = ca(a + b)
    // 9 = a^2c + abc [a := 1, b := 2, c := 3]
    //f'a= 2ac + bc [a := 1, b := 2, c := 3]
    const d = vm.add(a, b);
    try std.testing.expectEqual(vm.getData(c), vm.getData(d));
    try std.testing.expectEqual(.add, d.op);
    try std.testing.expectEqual(0b00000001, @as(i8, @intFromEnum(d.op)));

    const e = vm.mul(a, d);
    try std.testing.expectEqual(.mul, e.op);
    try std.testing.expectEqual(0b00000010, @as(i8, @intFromEnum(e.op)));
    try std.testing.expectEqual(vm.getData(c), vm.getData(e));

    const f = vm.mul(c, e);
    try std.testing.expectEqual(.mul, f.op);
    try std.testing.expectEqual(0b00000010, @as(i8, @intFromEnum(f.op)));

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
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
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
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
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
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
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
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
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
    var vm = try ValueManager(f32, 4).init(std.testing.allocator, 10);
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
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(2);

    const b = vm.powi(a, 2);
    // try std.testing.expectApproxEqAbs(4, vm.getData(b), APPROX);
    try std.testing.expectApproxEqAbs(4, vm.getData(b), APPROX);

    vm.backward(b);

    try std.testing.expectApproxEqAbs(4, vm.getGrad(a), APPROX);
}

test "ValueManager Power of Zero Test" {
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(2);

    const b = vm.powi(a, 0);
    // try std.testing.expectApproxEqAbs(4, vm.getData(b), APPROX);
    try std.testing.expectApproxEqAbs(1, vm.getData(b), APPROX);

    vm.backward(b);

    try std.testing.expectApproxEqAbs(0, vm.getGrad(a), APPROX);
}

test "ValueManager Recalculate Test" {
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
    defer vm.deinit();

    const a = vm.new(2);
    const b = vm.new(3);

    const c = vm.powi(a, -1);
    try std.testing.expectApproxEqAbs(0.5, vm.getData(c), APPROX);

    const d = vm.div(b, a);
    try std.testing.expectApproxEqAbs(1.5, vm.getData(d), APPROX);

    const e = vm.mul(c, d);
    try std.testing.expectApproxEqAbs(0.75, vm.getData(e), APPROX);

    vm.forward(e);

    try std.testing.expectApproxEqAbs(0.5, vm.getData(c), APPROX);
    try std.testing.expectApproxEqAbs(1.5, vm.getData(d), APPROX);
    try std.testing.expectApproxEqAbs(0.75, vm.getData(e), APPROX);

    vm.backward(e);
    try std.testing.expectApproxEqAbs(-0.75, vm.getGrad(a), APPROX);
    try std.testing.expectApproxEqAbs(0.25, vm.getGrad(b), APPROX);
}

test "ValueManager Forward Test" {
    var vm = try ValueManager(f32, 0).init(std.testing.allocator, 10);
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

    vm.getDataPtr(a).* = 1;
    vm.zeroGrad();

    vm.forward(e);
    try std.testing.expectApproxEqAbs(1, vm.getData(c), APPROX);
    try std.testing.expectApproxEqAbs(3, vm.getData(d), APPROX);
    try std.testing.expectApproxEqAbs(3, vm.getData(e), APPROX);

    vm.backward(e);
    try std.testing.expectApproxEqAbs(-6, vm.getGrad(a), APPROX);
    try std.testing.expectApproxEqAbs(1, vm.getGrad(b), APPROX);
}

test "ManyValueManager Addition" {
    const SIZE = 4;
    const V = @Vector(SIZE, f32);
    var mvm = try ManyValueManager(f32, &[_]comptime_int{SIZE}).init(std.testing.allocator, 10);
    defer mvm.deinit();

    const a = mvm.newRow(@as(V, @splat(3)));
    for (0..SIZE) |i| {
        try std.testing.expectApproxEqAbs(3, mvm.getData(a)[0][i], APPROX);
    }
    const b = mvm.newRow(@as(V, @splat(1)));
    for (0..SIZE) |i| {
        try std.testing.expectApproxEqAbs(1, mvm.getData(b)[0][i], APPROX);
    }

    const c = mvm.add(a, b);
    for (0..SIZE) |i| {
        try std.testing.expectApproxEqAbs(4, mvm.getData(c)[0][i], APPROX);
    }

    const rows = SIZE;
    const columns = 3;
    const x = mvm.manyNewColumns([1]V{@as(V, .{ 1, 2, 3, 4 })} ** columns);
    for (0..rows) |row| {
        for (0..columns) |column| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(row)) + 1, mvm.getData(x)[column][row], APPROX);
        }
    }
    const y = mvm.manyNewColumns([1]V{@as(V, .{ 4, 3, 2, 1 })} ** columns);
    for (0..rows) |row| {
        for (0..columns) |column| {
            try std.testing.expectApproxEqAbs(4 - @as(f32, @floatFromInt(row)), mvm.getData(y)[column][row], APPROX);
        }
    }

    const z = mvm.add(x, y);
    for (0..rows) |row| {
        for (0..columns) |column| {
            try std.testing.expectApproxEqAbs(5, mvm.getData(z)[column][row], APPROX);
        }
    }
}
