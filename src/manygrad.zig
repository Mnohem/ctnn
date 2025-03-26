const std = @import("std");
const mode = @import("builtin").mode;
const grad = @import("grad.zig");
const ValueManager = grad.ValueManager;
const ValueRef = grad.ValueRef;
const Operator = grad.Operator;
const Idx = grad.Idx;
const PowInt = grad.PowInt;
const op_without_int_size = grad.op_without_int_size;

pub const Orientation = enum(u1) { by_row, by_column };
pub fn ManyRef(m: comptime_int, n: comptime_int, o: Orientation) type {
    return struct {
        val_ref: ValueRef,
        // if oriented by row, we store m vectors of size n
        // if oriented by column, we store n vectors of size m
        comptime oriented: Orientation = o,
        comptime rows: usize = m,
        comptime columns: usize = n,
        fn numVectors() comptime_int {
            comptime return switch (o) {
                .by_row => m,
                .by_column => n,
            };
        }
        fn vectorSize() comptime_int {
            comptime return switch (o) {
                .by_row => n,
                .by_column => m,
            };
        }
        pub fn transpose(self: @This()) ManyRef(n, m, switch (o) {
            .by_column => .by_row,
            .by_row => .by_column,
        }) {
            return .{ .val_ref = self.val_ref };
        }
    };
}
test "Ensure ManyRef Size" {
    std.debug.assert(@sizeOf(ManyRef(0, 0, .by_column)) == 4);
}
fn reconstruct(comptime num_vectors: usize, comptime vector_size: usize, comptime oriented: Orientation, val_ref: ValueRef) ManyRef(switch (oriented) {
    .by_row => num_vectors,
    .by_column => vector_size,
}, switch (oriented) {
    .by_row => vector_size,
    .by_column => num_vectors,
}, oriented) {
    return .{ .val_ref = val_ref };
}
// Assume each value in vector_size is unique
pub fn ManyValueManager(NumType: type, vector_sizes: []const comptime_int) type {
    switch (@typeInfo(NumType)) {
        .float => {},
        else => @compileError("Scalar type must be float"),
    }
    comptime var vector_list: [vector_sizes.len]type = undefined;
    inline for (0..vector_sizes.len) |i| {
        vector_list[i] = @Vector(vector_sizes[i], NumType);
    }

    comptime var vm_list: [vector_sizes.len]type = undefined;
    inline for (0..vector_sizes.len) |i| {
        vm_list[i] = ValueManager(NumType, vector_sizes[i]);
    }
    const Vms = std.meta.Tuple(&vm_list);

    return struct {
        allocator: std.mem.Allocator,
        vms: Vms,
        expr_graph: std.AutoArrayHashMap(TermId, []ValueRef),

        const Self = @This();
        const TermId = struct { u32, ValueRef };
        const _ = std.debug.assert(@sizeOf(TermId) == 8);
        pub const Scalar = NumType;

        fn nanBP(vector: anytype) void {
            if (mode != .Debug) return;
            const nan_cond = vector != vector;
            const inf_cond = blk: {
                const is_inf = @abs(vector) == @as(@TypeOf(vector), @splat(std.math.inf(Scalar)));
                const num_infs = @reduce(.Add, @select(Scalar, is_inf, @as(@TypeOf(vector), @splat(1)), @as(@TypeOf(vector), @splat(0))));
                break :blk num_infs > 0;
            };
            if (@reduce(.Or, nan_cond) or inf_cond) {
                std.debug.print("{} contains nonnormal values", .{vector});
                @breakpoint();
            }
        }
        // returns the idx for which vm this vector is in
        fn validateVector(Vec: type) comptime_int {
            inline for (vector_list, 0..) |Vector, i| {
                if (Vec == Vector) {
                    return i;
                }
            } else {
                @compileError(std.fmt.comptimePrint("There is no ValueManager for vector {}", .{Vec}));
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

        pub fn getData(self: *const Self, ref: anytype) switch (ref.oriented) {
            .by_row => [ref.rows]@Vector(ref.columns, Scalar),
            .by_column => [ref.columns]@Vector(ref.rows, Scalar),
        } {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            return self.vms[vm_idx].data_storage.items[@intFromEnum(ref.val_ref.idx)..][0..@TypeOf(ref).numVectors()].*;
        }
        pub fn getGrad(self: *const Self, ref: anytype) switch (ref.oriented) {
            .by_row => [ref.rows]@Vector(ref.columns, Scalar),
            .by_column => [ref.columns]@Vector(ref.rows, Scalar),
        } {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const vm_grad_storage = self.vms[vm_idx].grad_storage;
            const start = @intFromEnum(ref.val_ref.idx);
            const len = @TypeOf(ref).numVectors();
            return if (vm_grad_storage.items.len >= start + len)
                vm_grad_storage.items[start..][0..len].*
            else
                @splat(@splat(0.0));
        }
        pub fn getDataPtr(self: *Self, ref: anytype) switch (ref.oriented) {
            .by_row => *[ref.rows]@Vector(ref.columns, Scalar),
            .by_column => *[ref.columns]@Vector(ref.rows, Scalar),
        } {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            return self.vms[vm_idx].data_storage.items[@intFromEnum(ref.val_ref.idx)..][0..@TypeOf(ref).numVectors()];
        }

        pub fn init(a: std.mem.Allocator, capacity: usize) !Self {
            var vms: Vms = undefined;
            inline for (&vms) |*vm| {
                vm.* = try @TypeOf(vm.*).init(a, capacity);
            }
            return .{ .allocator = a, .vms = vms, .expr_graph = std.AutoArrayHashMap(TermId, []ValueRef).init(a) };
        }
        pub fn deinit(self: *Self) void {
            inline for (&self.vms) |*vm| {
                vm.deinit();
            }
            for (self.expr_graph.values()) |val| {
                self.allocator.free(val);
            }
            self.expr_graph.deinit();
        }
        pub fn valuesWithShapeOf(self: *Self, Shape: type, value: Scalar) Shape {
            var ref: Shape = .{
                .val_ref = ValueRef{
                    .op = .noop,
                    .idx = @enumFromInt(0),
                },
            };
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            ref.val_ref.idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len);

            self.vms[vm_idx].data_storage.appendSlice(self.allocator, &[1]@Vector(@TypeOf(ref).vectorSize(), Scalar){@splat(value)} ** @TypeOf(ref).numVectors()) catch |err| {
                std.debug.panic("Failed to store value {} with shape of {any}: {}", .{ value, ref, err });
            };

            return ref;
        }
        pub fn newRow(self: *Self, row: anytype) ManyRef(1, vectorSize(@TypeOf(row)), .by_row) {
            const vm_idx = validateVector(@TypeOf(row));

            return .{ .val_ref = self.vms[vm_idx].new(row), .oriented = .by_row };
        }
        pub fn manyNewRows(self: *Self, rows: anytype) ManyRef(rows.len, vectorSize(@TypeOf(rows[0])), .by_row) {
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
        pub fn newColumn(self: *Self, column: anytype) ManyRef(vectorSize(@TypeOf(column)), 1, .by_column) {
            const vm_idx = validateVector(@TypeOf(column));

            return .{ .val_ref = self.vms[vm_idx].new(column), .oriented = .by_column };
        }
        pub fn manyNewColumns(self: *Self, columns: anytype) ManyRef(vectorSize(@TypeOf(columns[0])), columns.len, .by_column) {
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

        pub fn splatIntoRows(self: *Self, row_length: comptime_int, ref: anytype) ManyRef(ref.rows, row_length, .by_row) {
            if (ref.columns != 1)
                @compileError(std.fmt.comptimePrint("ref: {} must refer to a column vector", .{@TypeOf(ref)}));
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            inline for (vector_sizes, 0..) |size, parent_vm_idx| {
                if (size == row_length) {
                    const result = ValueRef{ .op = .external_splat, .idx = @enumFromInt(self.vms[parent_vm_idx].data_storage.items.len) };

                    if (@TypeOf(ref).numVectors() == 1) {
                        const vector = self.vms[vm_idx].getData(ref.val_ref);

                        inline for (0..@TypeOf(ref).vectorSize()) |i| {
                            _ = self.vms[parent_vm_idx].newExpr(.external_splat, @splat(vector[i]), &([_]u32{ vm_idx, @bitCast(ref.val_ref), i })) catch |err| {
                                std.debug.panic("Could not splat into row from {}: {}", .{ ref, err });
                            };
                        }
                    } else {
                        // TODO this branch requires testing, this runs when we are given a noncontiguous vector to splat
                        // should work with backward too but needs tests
                        // inline for (0..@TypeOf(ref).vectorSize()) |i| {
                        //     const val_ref = ValueRef{ .op = ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) };
                        //     _ = self.vms[parent_vm_idx].newExpr(.external_splat, @splat(self.vms[vm_idx].getData(val_ref)[0]), &([_]u32{ vm_idx, @bitCast(val_ref), 0 })) catch |err| {
                        //         std.debug.panic("Could not splat into row from {}: {}", .{ ref, err });
                        //     };
                        // }
                        unreachable;
                    }

                    return .{ .val_ref = result, .oriented = .by_row };
                }
            }
        }
        pub fn splatIntoColumns(self: *Self, column_length: comptime_int, ref: anytype) ManyRef(column_length, ref.columns, .by_column) {
            if (ref.rows != 1)
                @compileError(std.fmt.comptimePrint("ref: {} must refer to a row vector", .{@TypeOf(ref)}));
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            inline for (vector_sizes, 0..) |size, parent_vm_idx| {
                if (size == column_length) {
                    const result = ValueRef{ .op = .external_splat, .idx = @enumFromInt(self.vms[parent_vm_idx].data_storage.items.len) };

                    if (@TypeOf(ref).numVectors() == 1) {
                        const vector = self.vms[vm_idx].getData(ref.val_ref);

                        inline for (0..@TypeOf(ref).vectorSize()) |i| {
                            _ = self.vms[parent_vm_idx].newExpr(.external_splat, @splat(vector[i]), &([_]u32{ vm_idx, @bitCast(ref.val_ref), i })) catch |err| {
                                std.debug.panic("Could not splat into column from {}: {}", .{ ref, err });
                            };
                        }
                    } else {
                        // TODO this branch requires testing, this runs when we are given a noncontiguous vector to splat
                        // should work with backward too but needs tests
                        // inline for (0..@TypeOf(ref).vectorSize()) |i| {
                        //     const val_ref = ValueRef{ .op = ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) };
                        //     _ = self.vms[parent_vm_idx].newExpr(.external_splat, @splat(self.vms[vm_idx].getData(val_ref)[0]), &([_]u32{ vm_idx, @bitCast(val_ref), 0 })) catch |err| {
                        //         std.debug.panic("Could not splat into column from {}: {}", .{ ref, err });
                        //     };
                        // }
                        unreachable;
                    }

                    return .{ .val_ref = result, .oriented = .by_column };
                }
            }
        }

        // User facing functions are type checked but types must have the same orientation
        // This can be ensured by the user using self.reorient
        fn sameRefTypes(ref1: anytype, ref2: anytype) void {
            if (ref1.rows != ref2.rows or ref1.columns != ref2.columns) {
                @compileError(std.fmt.comptimePrint("{} and {} are not the same shape", .{ @TypeOf(ref1), @TypeOf(ref2) }));
            } else if (ref1.oriented != ref2.oriented) {
                @compileError(std.fmt.comptimePrint("{} and {} are not oriented the same way", .{ @TypeOf(ref1), @TypeOf(ref2) }));
            }
        }
        pub fn add(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            const vm_idx = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);

            const result: @TypeOf(ref1) = .{
                .val_ref = ValueRef{
                    .op = .add,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref1.oriented,
            };

            for (0..@TypeOf(ref1).numVectors()) |i| {
                _ = self.vms[vm_idx].add(.{ .op = ref1.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i) }, .{
                    .op = ref2.val_ref.op,
                    .idx = @enumFromInt(@intFromEnum(ref2.val_ref.idx) + i),
                });
            }

            return result;
        }
        pub fn elemMul(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            const vm_idx = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);

            const result: @TypeOf(ref1) = .{
                .val_ref = ValueRef{
                    .op = .mul,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref1.oriented,
            };

            for (0..@TypeOf(ref1).numVectors()) |i| {
                _ = self.vms[vm_idx].mul(.{ .op = ref1.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i) }, .{
                    .op = ref2.val_ref.op,
                    .idx = @enumFromInt(@intFromEnum(ref2.val_ref.idx) + i),
                });
            }

            return result;
        }
        pub fn elemMax(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            const vm_idx = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);

            const result: @TypeOf(ref1) = .{
                .val_ref = ValueRef{
                    .op = .max,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref1.oriented,
            };

            for (0..@TypeOf(ref1).numVectors()) |i| {
                _ = self.vms[vm_idx].max(.{ .op = ref1.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i) }, .{
                    .op = ref2.val_ref.op,
                    .idx = @enumFromInt(@intFromEnum(ref2.val_ref.idx) + i),
                });
            }

            return result;
        }

        pub fn sub(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            return self.add(ref1, self.neg(ref2));
        }
        pub fn elemDiv(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            return self.elemMul(ref1, self.elemPowi(ref2, -1));
        }

        pub fn elemPowi(self: *Self, ref: anytype, power: PowInt) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result: @TypeOf(ref) = .{
                .val_ref = ValueRef{
                    .op = @enumFromInt(@as(i8, power) << op_without_int_size),
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..@TypeOf(ref).numVectors()) |i| {
                _ = self.vms[vm_idx].powi(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) }, power);
            }

            return result;
        }
        pub fn elemExp(self: *Self, ref: anytype) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result: @TypeOf(ref) = .{
                .val_ref = ValueRef{
                    .op = .exp,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..@TypeOf(ref).numVectors()) |i| {
                _ = self.vms[vm_idx].exp(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) });
            }

            return result;
        }
        pub fn log(self: *Self, ref: anytype) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result: @TypeOf(ref) = .{
                .val_ref = ValueRef{
                    .op = .log,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..@TypeOf(ref).numVectors()) |i| {
                _ = self.vms[vm_idx].log(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) });
            }

            return result;
        }
        pub fn sqrt(self: *Self, ref: anytype) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result: @TypeOf(ref) = .{
                .val_ref = ValueRef{
                    .op = .sqrt,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..@TypeOf(ref).numVectors()) |i| {
                _ = self.vms[vm_idx].sqrt(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) });
            }

            return result;
        }

        pub fn neg(self: *Self, ref: anytype) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result: @TypeOf(ref) = .{
                .val_ref = ValueRef{
                    .op = .neg,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..@TypeOf(ref).numVectors()) |i| {
                _ = self.vms[vm_idx].neg(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) });
            }

            return result;
        }
        fn matMulTyped(m: comptime_int, n: comptime_int, p: comptime_int, _: ManyRef(m, n, .by_row), _: ManyRef(n, p, .by_column)) void {}
        pub fn matMul(self: *Self, ref1: anytype, ref2: anytype) ManyRef(ref1.rows, ref2.columns, .by_row) {
            matMulTyped(ref1.rows, ref1.columns, ref2.columns, ref1, ref2);
            const vm_id1 = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);
            const vm_id2 = refVmIdx(ref2.oriented, ref2.rows, ref2.columns);

            if (vm_id1 == vm_id2) {
                var idxs: [ref1.rows]Idx = undefined;
                for (0..ref1.rows) |i| {
                    idxs[i] = @enumFromInt(self.vms[vm_id1].data_storage.items.len);
                    for (0..ref2.columns) |j| {
                        const ref1_idx: Idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i);
                        const ref2_idx: Idx = @enumFromInt(@intFromEnum(ref2.val_ref.idx) + j);

                        _ = self.vms[vm_id1].mul(.{ .op = ref1.val_ref.op, .idx = ref1_idx }, .{
                            .op = ref2.val_ref.op,
                            .idx = ref2_idx,
                        });
                    }
                }

                var intermediate = ManyRef(ref1.columns, ref2.columns, .by_column){
                    .val_ref = ValueRef{
                        .op = .mul,
                        .idx = idxs[0],
                    },
                };
                const start = self.sumColumns(intermediate);
                for (1..ref1.rows) |i| {
                    intermediate.val_ref.idx = idxs[i];
                    _ = self.sumColumns(intermediate);
                }

                return .{ .val_ref = start.val_ref };
            } else unreachable;
        }
        pub fn matVecMul(self: *Self, ref1: anytype, ref2: ManyRef(ref1.columns, 1, .by_column)) ManyRef(ref1.rows, 1, .by_column) {
            matMulTyped(ref1.rows, ref1.columns, ref2.columns, ref1, ref2);
            const vm_id1 = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);
            const vm_id2 = refVmIdx(ref2.oriented, ref2.rows, ref2.columns);

            if (vm_id1 == vm_id2) {
                const start_idx: Idx = @enumFromInt(self.vms[vm_id1].data_storage.items.len);
                for (0..ref1.rows) |i| {
                    const ref1_idx: Idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i);

                    _ = self.vms[vm_id1].mul(.{ .op = ref1.val_ref.op, .idx = ref1_idx }, ref2.val_ref);
                }

                return self.sumRows(@TypeOf(ref1){
                    .val_ref = ValueRef{
                        .op = .mul,
                        .idx = start_idx,
                    },
                });
            } else unreachable;
        }
        pub fn sumRows(self: *Self, ref: anytype) ManyRef(ref.rows, 1, .by_column) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, .by_row));

            var vector: @Vector(ref.rows, Scalar) = undefined;
            for (self.getData(ref), 0..ref.rows) |vec, i| {
                vector[i] = @reduce(.Add, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_sum, vector, &([_]u32{ child_vm_idx, @bitCast(ref.val_ref) })) catch |err| {
                std.debug.panic("Could not sum rows from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_column };
        }
        pub fn sumColumns(self: *Self, ref: anytype) ManyRef(1, ref.columns, .by_row) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, .by_column));

            var vector: @Vector(ref.columns, Scalar) = undefined;
            for (self.getData(ref), 0..ref.columns) |vec, i| {
                vector[i] = @reduce(.Add, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_sum, vector, &([_]u32{ child_vm_idx, @bitCast(ref.val_ref) })) catch |err| {
                std.debug.panic("Could not sum columns from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_row };
        }
        pub fn maxRows(self: *Self, ref: anytype) ManyRef(ref.rows, 1, .by_column) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, .by_row));

            var vector: @Vector(ref.rows, Scalar) = undefined;
            for (self.getData(ref), 0..ref.rows) |vec, i| {
                vector[i] = @reduce(.Max, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_max, vector, &([_]u32{ child_vm_idx, @bitCast(ref.val_ref) })) catch |err| {
                std.debug.panic("Could not max rows from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_column };
        }
        pub fn maxColumns(self: *Self, ref: anytype) ManyRef(1, ref.columns, .by_row) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, .by_column));

            var vector: @Vector(ref.columns, Scalar) = undefined;
            for (self.getData(ref), 0..ref.columns) |vec, i| {
                vector[i] = @reduce(.Max, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_max, vector, &([_]u32{ child_vm_idx, @bitCast(ref.val_ref) })) catch |err| {
                std.debug.panic("Could not max columns from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_row };
        }

        fn externalRecalculate(self: *Self, ref: anytype) void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const child_vm_idx, const child, const extra = self.vms[vmi].getExternalInfo(ref.val_ref);

            switch (child_vm_idx) {
                inline 0...vector_sizes.len - 1 => |cvm_idx| {
                    const cvm = &self.vms[cvm_idx];
                    switch (ref.val_ref.op) {
                        .external_sum => {
                            for (cvm.data_storage.items[@intFromEnum(child.idx)..][0..@TypeOf(ref).vectorSize()], 0..) |vec, i| {
                                self.vms[vmi].getDataPtr(ref.val_ref)[i] = @reduce(.Add, vec);
                            }
                        },
                        .external_max => {
                            for (cvm.data_storage.items[@intFromEnum(child.idx)..][0..@TypeOf(ref).vectorSize()], 0..) |vec, i| {
                                self.vms[vmi].getDataPtr(ref.val_ref)[i] = @reduce(.Max, vec);
                            }
                        },
                        .external_splat => {
                            const vector = cvm.data_storage.items[@intFromEnum(child.idx)];
                            self.getDataPtr(ref)[0] = @splat(vector[extra[0]]);
                        },
                        else => unreachable,
                    }
                },
                else => unreachable,
            }
        }

        pub fn zeroGrad(self: *Self) void {
            inline for (&self.vms) |*vm| {
                vm.zeroGrad();
            }
        }
        pub fn createExprGraph(self: *Self, ref: anytype) !void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);

            for (self.expr_graph.values()) |val| {
                self.allocator.free(val);
            }
            self.expr_graph.clearRetainingCapacity();
            // go over the span of ValueRefs that this ManyRef holds
            for (0..@TypeOf(ref).numVectors()) |i| {
                const val_ref: ValueRef = .{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) };
                try self.expr_graph.putNoClobber(.{ vmi, val_ref }, self.vms[vmi].externalParts(val_ref));
            }

            var prev_index: usize = 0;
            while (prev_index != self.expr_graph.count()) {
                var index = self.expr_graph.count();
                var expr_graph_idx = prev_index;
                while (expr_graph_idx < index) : (expr_graph_idx += 1) {
                    const head_vm_idx, _ = self.expr_graph.keys()[expr_graph_idx];
                    const vm_local_externals = self.expr_graph.values()[expr_graph_idx];

                    for (vm_local_externals) |requested| {
                        switch (head_vm_idx) {
                            inline 0...vector_sizes.len - 1 => |vm_idx| {
                                const vm = &self.vms[vm_idx];
                                const child_vm_idx, const child, _ = vm.getExternalInfo(requested);

                                switch (child_vm_idx) {
                                    inline 0...vector_sizes.len - 1 => |cvm_idx| {
                                        const cvm = &self.vms[cvm_idx];
                                        switch (requested.op) {
                                            .external_sum, .external_max => for (0..vector_sizes[vm_idx]) |i| {
                                                const child_val_ref: ValueRef = .{ .op = child.op, .idx = @enumFromInt(@intFromEnum(child.idx) + i) };
                                                const key = .{ cvm_idx, child_val_ref };
                                                const value = if (self.expr_graph.getIndex(key)) |idx| blk: {
                                                    if (idx < index) index -= 1;
                                                    if (idx < expr_graph_idx) expr_graph_idx -= 1;
                                                    if (idx == expr_graph_idx) unreachable; // circular dependency
                                                    break :blk self.expr_graph.fetchOrderedRemove(key).?.value;
                                                } else cvm.externalParts(child_val_ref);
                                                try self.expr_graph.putNoClobber(key, value);
                                            },
                                            .external_splat => {
                                                const key = .{ cvm_idx, child };
                                                const value = if (self.expr_graph.getIndex(key)) |idx| blk: {
                                                    if (idx < index) index -= 1;
                                                    if (idx < expr_graph_idx) expr_graph_idx -= 1;
                                                    if (idx == expr_graph_idx) unreachable; // circular dependency
                                                    break :blk self.expr_graph.fetchOrderedRemove(key).?.value;
                                                } else cvm.externalParts(child);
                                                try self.expr_graph.putNoClobber(key, value);
                                            },
                                            else => unreachable,
                                        }
                                    },
                                    else => unreachable,
                                }
                            },
                            else => unreachable,
                        }
                    }
                }
                prev_index = index;
            }
        }
        // fn constructGraphRec(ref: anytype) void {
        //     const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);
        //     const vm_local_externals = self.vms[vmi].externalParts(ref);
        // }

        pub fn forward(self: *Self, ref: anytype) !void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const end = if (self.expr_graph.getIndex(.{ vmi, ref.val_ref })) |idx| idx else blk: {
                try self.createExprGraph(ref);
                break :blk 0;
            };
            const len = self.expr_graph.count();
            for (end..len) |i| {
                const head_vm_idx, const expr_head = self.expr_graph.keys()[len - i - 1];
                const vm_local_externals = self.expr_graph.values()[len - i - 1];
                switch (head_vm_idx) {
                    inline 0...vector_sizes.len - 1 => |vm_idx| {
                        const vm = &self.vms[vm_idx];
                        for (vm_local_externals) |external| {
                            self.externalRecalculate(ManyRef(vector_sizes[vm_idx], 1, .by_column){ .val_ref = external });
                        }

                        try vm.forward(expr_head);
                    },
                    else => unreachable,
                }
            }
        }
        pub fn backwardWithGrad(self: *Self, ref: anytype, init_g: Scalar) !void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);

            if (self.vms[vmi].grad_storage.items.len != self.vms[vmi].data_storage.items.len) {
                self.vms[vmi].grad_storage.resize(self.allocator, self.vms[vmi].data_storage.items.len) catch |err| {
                    std.debug.panic("Failed to resize gradient: {}", .{err});
                };
                self.vms[vmi].zeroGrad();
            }
            for (0..@TypeOf(ref).numVectors()) |i| {
                self.vms[vmi].grad_storage.items[@intFromEnum(ref.val_ref.idx)..][i] = @splat(init_g);
            }

            if (!self.expr_graph.contains(.{ vmi, ref.val_ref })) {
                try self.createExprGraph(ref);
            }
            for (self.expr_graph.keys(), self.expr_graph.values()) |key, value| {
                const head_vm_idx, const expr_head = key;
                const vm_local_externals = value;
                switch (head_vm_idx) {
                    inline 0...vector_sizes.len - 1 => |vm_idx| {
                        const vm = &self.vms[vm_idx];
                        try vm.backwardWithGrad(expr_head, null);

                        for (vm_local_externals) |external| {
                            const child_vm_idx, const child, const extra = vm.getExternalInfo(external);
                            const curr_grad = vm.getGrad(external);
                            // if (@reduce(.And, curr_grad == @as(@TypeOf(curr_grad), @splat(0)))) @breakpoint();

                            switch (child_vm_idx) {
                                inline 0...vector_sizes.len - 1 => |cvm_idx| {
                                    const cvm = &self.vms[cvm_idx];
                                    if (cvm.grad_storage.items.len != cvm.data_storage.items.len) {
                                        cvm.grad_storage.resize(self.allocator, cvm.data_storage.items.len) catch |err| {
                                            std.debug.panic("Failed to resize gradient: {}", .{err});
                                        };
                                        cvm.zeroGrad();
                                    }

                                    switch (external.op) {
                                        .external_sum => for (cvm.grad_storage.items[@intFromEnum(child.idx)..][0..vector_sizes[vm_idx]], 0..) |*g, i| {
                                            g.* += @splat(curr_grad[i]);
                                            nanBP(g.*);
                                        },
                                        .external_max => for (cvm.grad_storage.items[@intFromEnum(child.idx)..][0..vector_sizes[vm_idx]], cvm.data_storage.items[@intFromEnum(child.idx)..][0..vector_sizes[vm_idx]], 0..) |*g, d, i| {
                                            // This is not mathematical as max is a discrete function
                                            // Right now it is the naive solution, the max num passes along its gradient, other nums have no effect
                                            // If there are multiple equal maxes, we balance the proportion linearly between them
                                            const curr_max = vm.getData(external)[i];
                                            const which_max = d == @as(@TypeOf(d), @splat(curr_max));
                                            if (!@reduce(.Or, which_max)) @breakpoint();
                                            // const num_max = @reduce(.Add, @select(Scalar, which_max, @as(@TypeOf(d), @splat(1)), @as(@TypeOf(d), @splat(0))));
                                            const unbalanced = @select(Scalar, which_max, @as(@TypeOf(d), @splat(curr_grad[i])), @as(@TypeOf(d), @splat(0)));
                                            g.* += unbalanced; // / @as(@TypeOf(d), @splat(num_max));
                                            nanBP(g.*);
                                        },
                                        .external_splat => {
                                            const child_grad_ptr = &cvm.grad_storage.items[@intFromEnum(child.idx)];
                                            child_grad_ptr[extra[0]] += @reduce(.Add, curr_grad);
                                            nanBP(child_grad_ptr.*);
                                        },
                                        else => unreachable,
                                    }
                                },
                                else => unreachable,
                            }
                        }
                    },
                    else => unreachable,
                }
            }
        }

        pub fn debugRef(self: *const Self, ref: anytype) void {
            std.debug.print("Shape: {}\n", .{@TypeOf(ref)});
            std.debug.print("Operation: {}\n", .{ref.val_ref.op});
            std.debug.print("Data: {any}\n", .{self.getData(ref)});
            std.debug.print("Gradient: {any}\n", .{self.getGrad(ref)});
        }
        pub fn treePrint(self: *const Self, ref: anytype, depth: usize) void {
            for (0..depth) |_| {
                std.debug.print("-", .{});
            }
            std.debug.print(" {}, {}({d})\n", .{ @TypeOf(ref), ref.val_ref.op, @intFromEnum(ref.val_ref.op) >> op_without_int_size });
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);
            var vm = self.vms[vmi];
            op: switch (ref.val_ref.op) {
                .noop => {},
                inline .external_splat, .external_sum, .external_max => |op| {
                    const child_vm_idx, const child, _ = vm.getExternalInfo(ref.val_ref);
                    const child_orient = ref.transpose().oriented;
                    const child_num_vectors = if (op == .external_splat) 1 else @TypeOf(ref).vectorSize();
                    switch (child_vm_idx) {
                        inline 0...vector_sizes.len - 1 => |idx| {
                            const child_vector_size = vector_sizes[idx];
                            const child_many_ref = reconstruct(child_num_vectors, child_vector_size, child_orient, child);
                            self.treePrint(child_many_ref, depth + 1);
                        },
                        else => unreachable,
                    }
                },
                .add, .mul, .max => {
                    const children = vm.getLocalChildren(ref.val_ref);
                    self.treePrint(@TypeOf(ref){ .val_ref = children[0] }, depth + 1);
                    self.treePrint(@TypeOf(ref){ .val_ref = children[1] }, depth + 1);
                },
                _ => continue :op .neg,
                .exp, .neg, .log, .sqrt => {
                    const children = vm.getLocalChildren(ref.val_ref);
                    self.treePrint(@TypeOf(ref){ .val_ref = children[0] }, depth + 1);
                },
            }
        }
        pub fn backward(self: *Self, ref: anytype) !void {
            return self.backwardWithGrad(ref, 1);
        }
    };
}
