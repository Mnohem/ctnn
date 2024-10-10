const std = @import("std");
const grad = @import("grad.zig");
const ValueManager = grad.ValueManager;
const ValueRef = grad.ValueRef;
const Operator = grad.Operator;
const Idx = grad.Idx;
const PowInt = grad.PowInt;
const op_without_int_size = grad.op_without_int_size;

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
pub fn ManyRefOrient(m: comptime_int, n: comptime_int, orientation: Orientation) type {
    return ManyRef(m, n, (m > n and orientation == .by_column) or (m <= n and orientation == .by_row));
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
        expr_graph: std.AutoArrayHashMap(TermId, []ValueRef),

        const Self = @This();
        const TermId = struct { u32, ValueRef };
        const _ = std.debug.assert(@sizeOf(TermId) == 8);

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
        pub fn getGrad(self: *Self, ref: anytype) switch (ref.oriented) {
            .by_row => [ref.rows]@Vector(ref.columns, Scalar),
            .by_column => [ref.columns]@Vector(ref.rows, Scalar),
        } {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            return self.vms[vm_idx].grad_storage.items[@intFromEnum(ref.val_ref.idx)..][0..ref.numVectors()].*;
        }
        pub fn getDataPtr(self: *Self, ref: anytype) switch (ref.oriented) {
            .by_row => *[ref.rows]@Vector(ref.columns, Scalar),
            .by_column => *[ref.columns]@Vector(ref.rows, Scalar),
        } {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            return self.vms[vm_idx].data_storage.items[@intFromEnum(ref.val_ref.idx)..][0..ref.numVectors()];
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
        pub fn newRow(self: *Self, row: anytype) ManyRef(1, vectorSize(@TypeOf(row)), true) {
            const vm_idx = validateVector(@TypeOf(row));

            return .{ .val_ref = self.vms[vm_idx].new(row), .oriented = .by_row };
        }
        pub fn manyNewRows(self: *Self, rows: anytype) ManyRefOrient(rows.len, vectorSize(@TypeOf(rows[0])), .by_row) {
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
        pub fn manyNewColumns(self: *Self, columns: anytype) ManyRefOrient(vectorSize(@TypeOf(columns[0])), columns.len, .by_column) {
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
                        .op = .mul,
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

        pub fn elemDiv(self: *Self, ref1: anytype, ref2: anytype) @TypeOf(ref1) {
            sameRefTypes(ref1, ref2);
            return self.elemMul(ref1, self.elemPowi(ref2, -1));
        }

        pub fn elemPowi(self: *Self, ref: anytype, power: PowInt) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result = .{
                .val_ref = ValueRef{
                    .op = @enumFromInt(@as(i8, power) << op_without_int_size),
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..ref.numVectors()) |i| {
                _ = self.vms[vm_idx].powi(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) }, power);
            }

            return result;
        }
        pub fn elemExp(self: *Self, ref: anytype) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result = .{
                .val_ref = ValueRef{
                    .op = .exp,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..ref.numVectors()) |i| {
                _ = self.vms[vm_idx].exp(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) });
            }

            return result;
        }

        pub fn neg(self: *Self, ref: anytype) @TypeOf(ref) {
            const vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);

            const result = .{
                .val_ref = ValueRef{
                    .op = .neg,
                    .idx = @enumFromInt(self.vms[vm_idx].data_storage.items.len),
                },
                .oriented = ref.oriented,
            };

            for (0..ref.numVectors()) |i| {
                _ = self.vms[vm_idx].neg(.{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) });
            }

            return result;
        }
        fn matMulTyped(m: comptime_int, n: comptime_int, p: comptime_int, _: ManyRefOrient(m, n, .by_row), _: ManyRefOrient(n, p, .by_column)) void {}
        pub fn matMul(self: *Self, ref1: anytype, ref2: anytype) ManyRefOrient(ref1.rows, ref2.columns, .by_row) {
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

                var intermediate = ManyRefOrient(ref1.columns, ref2.columns, .by_column){
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
        pub fn matVecMul(self: *Self, ref1: anytype, ref2: ManyRef(ref1.columns, 1, true)) ManyRef(ref1.rows, 1, true) {
            matMulTyped(ref1.rows, ref1.columns, ref2.columns, ref1, ref2);
            const vm_id1 = refVmIdx(ref1.oriented, ref1.rows, ref1.columns);
            const vm_id2 = refVmIdx(ref2.oriented, ref2.rows, ref2.columns);

            if (vm_id1 == vm_id2) {
                const start_idx: Idx = @enumFromInt(self.vms[vm_id1].data_storage.items.len);
                for (0..ref1.rows) |i| {
                    const ref1_idx: Idx = @enumFromInt(@intFromEnum(ref1.val_ref.idx) + i);

                    _ = self.vms[vm_id1].mul(.{ .op = ref1.val_ref.op, .idx = ref1_idx }, ref2.val_ref);
                }

                return self.sumRows(ManyRefOrient(ref1.rows, ref2.rows, .by_row){
                    .val_ref = ValueRef{
                        .op = .mul,
                        .idx = start_idx,
                    },
                });
            } else unreachable;
        }
        pub fn sumRows(self: *Self, ref: anytype) ManyRef(ref.rows, 1, true) {
            std.debug.assert(@TypeOf(ref) == ManyRefOrient(ref.rows, ref.columns, .by_row));

            var vector: @Vector(ref.rows, Scalar) = undefined;
            for (self.getData(ref), 0..ref.rows) |vec, i| {
                vector[i] = @reduce(.Add, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_sum, vector, &([_]u32{ child_vm_idx, @bitCast(ref.val_ref) })) catch |err| {
                std.debug.panic("Could not sum columns from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_column };
        }
        pub fn sumColumns(self: *Self, ref: anytype) ManyRef(1, ref.columns, true) {
            std.debug.assert(@TypeOf(ref) == ManyRefOrient(ref.rows, ref.columns, .by_column));

            var vector: @Vector(ref.columns, Scalar) = undefined;
            for (self.getData(ref), 0..ref.columns) |vec, i| {
                vector[i] = @reduce(.Add, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            // children stored: tuple index to vm of children, ValueRef to start of columns
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_sum, vector, &([_]u32{ child_vm_idx, @bitCast(ref.val_ref) })) catch |err| {
                std.debug.panic("Could not sum columns from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_row };
        }

        fn externalRecalculate(self: *Self, ref: anytype) void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const child_vm_idx, const child = self.vms[vmi].getExternalInfo(ref.val_ref);

            var vector: @Vector(vector_sizes[vmi], Scalar) = undefined;
            inline for (&self.vms, 0..) |*cvm, cvm_idx| {
                if (cvm_idx == child_vm_idx) {
                    for (cvm.data_storage.items[@intFromEnum(child.idx)..][0..ref.vectorSize()], 0..) |vec, i| {
                        vector[i] = @reduce(.Add, vec);
                    }
                    break;
                }
            }
            self.vms[vmi].getDataPtr(ref.val_ref).* = vector;
        }

        pub fn zeroGrad(self: *Self) void {
            inline for (&self.vms) |*vm| {
                vm.zeroGrad();
            }
        }
        // forward and backward is valid to call on a ref only after calling createExprGraph on that ref or any of its parents
        pub fn createExprGraph(self: *Self, ref: anytype) !void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);

            for (self.expr_graph.values()) |val| {
                self.allocator.free(val);
            }
            self.expr_graph.clearRetainingCapacity();
            // need to go over the span of ValueRefs that this ManyRef can hold
            for (0..ref.numVectors()) |i| {
                const val_ref: ValueRef = .{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) };
                try self.expr_graph.putNoClobber(.{ vmi, val_ref }, self.vms[vmi].externalParts(val_ref));
            }

            var prev_index: usize = 0;
            while (prev_index != self.expr_graph.count()) {
                const index = self.expr_graph.count();
                for (self.expr_graph.keys()[prev_index..index], self.expr_graph.values()[prev_index..index]) |head_info, vm_local_externals| {
                    const head_vm_idx, _ = head_info;
                    for (vm_local_externals) |requested| {
                        inline for (&self.vms, 0..) |*vm, vm_idx| {
                            if (vm_idx == head_vm_idx) {
                                const child_vm_idx, const child = vm.getExternalInfo(requested);

                                inline for (&self.vms, 0..) |*cvm, cvm_idx| {
                                    if (cvm_idx == child_vm_idx) {
                                        for (0..vector_sizes[vm_idx]) |i| {
                                            const child_val_ref: ValueRef = .{ .op = child.op, .idx = @enumFromInt(@intFromEnum(child.idx) + i) };
                                            try self.expr_graph.putNoClobber(.{ child_vm_idx, child_val_ref }, cvm.externalParts(child_val_ref));
                                        }
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
                prev_index = index;
            }
        }

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
                inline for (&self.vms, 0..) |*vm, vm_idx| {
                    if (vm_idx == head_vm_idx) {
                        for (vm_local_externals) |external| {
                            self.externalRecalculate(ManyRef(vector_sizes[vm_idx], 1, true){ .val_ref = external });
                        }

                        try vm.forward(expr_head);
                    }
                }
            }
        }
        pub fn backward(self: *Self, ref: anytype) !void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);

            if (self.vms[vmi].grad_storage.items.len != self.vms[vmi].data_storage.items.len) {
                self.vms[vmi].grad_storage.resize(self.allocator, self.vms[vmi].data_storage.items.len) catch |err| {
                    std.debug.panic("Failed to resize gradient: {}", .{err});
                };
                self.vms[vmi].zeroGrad();
            }
            for (0..ref.numVectors()) |i| {
                const val_ref: ValueRef = .{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) };
                self.vms[vmi].grad_storage.items[@intFromEnum(val_ref.idx)] = @splat(1);
            }

            const start = if (self.expr_graph.getIndex(.{ vmi, ref.val_ref })) |idx| idx else blk: {
                try self.createExprGraph(ref);
                break :blk 0;
            };
            for (self.expr_graph.keys()[start..], self.expr_graph.values()[start..]) |key, value| {
                const head_vm_idx, const expr_head = key;
                const vm_local_externals = value;
                inline for (&self.vms, 0..) |*vm, vm_idx| {
                    if (vm_idx == head_vm_idx) {
                        try vm.backwardWithGrad(expr_head, null);
                        for (vm_local_externals) |external| {
                            const child_vm_idx, const child = vm.getExternalInfo(external);
                            const vector = vm.getGrad(external);

                            inline for (&self.vms, 0..) |*cvm, cvm_idx| {
                                if (cvm_idx == child_vm_idx) {
                                    if (cvm.grad_storage.items.len != cvm.data_storage.items.len) {
                                        cvm.grad_storage.resize(self.allocator, cvm.data_storage.items.len) catch |err| {
                                            std.debug.panic("Failed to resize gradient: {}", .{err});
                                        };
                                        cvm.zeroGrad();
                                    }

                                    for (cvm.grad_storage.items[@intFromEnum(child.idx)..][0..vector_sizes[vm_idx]], 0..) |*g, i| {
                                        g.* += @splat(vector[i]);
                                    }
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    };
}
