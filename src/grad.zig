const std = @import("std");

pub fn Value(Data: type) type {
    return struct {
        data: Data,
        op: ?Operator = null,

        const Self = @This();

        const Operator = union(enum) {
            Plus: [2]*const Self,
            Mul: [2]*const Self,
            Minus: [2]*const Self,
            Div: [2]*const Self,
        };

        fn new(d: Data, o: Operator) Self {
            return Self{ .data = d, .op = o };
        }

        pub fn add(self: *const Self, other: *const Self) Self {
            return new(self.data + other.data, .{ .Plus = [_]*const Self{ self, other } });
        }

        pub fn mul(self: *const Self, other: *const Self) Self {
            return new(self.data * other.data, .{ .Mul = [_]*const Self{ self, other } });
        }
    };
}

test "Value Operations Test" {
    var a: Value(f32) = .{ .data = 1 };
    const b: Value(f32) = .{ .data = 2 };
    const c: Value(f32) = .{ .data = 3 };

    try std.testing.expectEqual(c.data, a.add(&b).data);

    try std.testing.expectEqual(c.mul(&a).data, c.data);

    try std.testing.expectEqual(a.add(&a).add(&a).add(&a).add(&a).add(&a).data, b.mul(&c).data);
    try std.testing.expectEqual(b.add(&b).add(&b).add(&b).data, b.mul(&c.add(&a)).data);
}
