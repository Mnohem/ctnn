const std = @import("std");
const grad = @import("grad.zig");
const ValueManager = grad.ValueManager;
const ValueRef = grad.ValueRef;
const Operator = grad.Operator;
const Idx = grad.Idx;
const u32s_in_usize = grad.u32s_in_usize;

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
        pub fn sumRows(self: *Self, ref: anytype) ManyRef(ref.rows, 1, true) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, true));

            var vector: @Vector(ref.rows, Scalar) = undefined;
            for (self.getData(ref), 0..ref.rows) |vec, i| {
                vector[i] = @reduce(.Add, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const vm_ptr: *ValueManager(Scalar, vector_sizes[child_vm_idx]) = &self.vms[child_vm_idx];
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_sum, vector, &(@as([u32s_in_usize]u32, @bitCast(@intFromPtr(vm_ptr))) ++ [_]u32{@bitCast(ref.val_ref)})) catch |err| {
                std.debug.panic("Could not sum rows from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_column };
        }
        pub fn sumColumns(self: *Self, ref: anytype) ManyRef(1, ref.columns, true) {
            std.debug.assert(@TypeOf(ref) == ManyRef(ref.rows, ref.columns, true));

            var vector: @Vector(ref.columns, Scalar) = undefined;
            for (self.getData(ref), 0..ref.columns) |vec, i| {
                vector[i] = @reduce(.Add, vec);
            }
            const parent_vm_idx = validateVector(@TypeOf(vector));
            const child_vm_idx = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const vm_ptr: *ValueManager(Scalar, vector_sizes[child_vm_idx]) = &self.vms[child_vm_idx];
            // children stored: pointer to vm of children, ValueRef to start of columns
            const val_ref = self.vms[parent_vm_idx].newExpr(.external_sum, vector, &(@as([u32s_in_usize]u32, @bitCast(@intFromPtr(vm_ptr))) ++ [_]u32{@bitCast(ref.val_ref)})) catch |err| {
                std.debug.panic("Could not sum columns from {}: {}", .{ ref, err });
            };

            return .{ .val_ref = val_ref, .oriented = .by_row };
        }

        fn externalRecalculate(self: *Self, ref: anytype) void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);
            const erased_child_vm_ptr, const child = self.vms[vmi].getExternalInfo(ref.val_ref);

            var vector: @Vector(vector_sizes[vmi], Scalar) = undefined;
            inline for (&self.vms) |*cvm| {
                if (@as(*anyopaque, @ptrCast(cvm)) == erased_child_vm_ptr) {
                    for (cvm.data_storage.items[@intFromEnum(child.idx)..][0..ref.vectorSize()], 0..) |vec, i| {
                        vector[i] = @reduce(.Add, vec);
                    }
                    break;
                }
            }
            self.vms[vmi].getDataPtr(ref.val_ref).* = vector;
        }

        pub fn forward(self: *Self, ref: anytype) !void {
            const vmi = refVmIdx(ref.oriented, ref.rows, ref.columns);

            var externals = std.AutoArrayHashMap(struct { *anyopaque, ValueRef }, []ValueRef).init(self.allocator);
            defer {
                for (externals.values()) |val| {
                    self.allocator.free(val);
                }
                externals.deinit();
            }
            // need to go over the span of ValueRefs that this ManyRef can hold
            for (0..ref.numVectors()) |i| {
                const val_ref: ValueRef = .{ .op = ref.val_ref.op, .idx = @enumFromInt(@intFromEnum(ref.val_ref.idx) + i) };
                try externals.putNoClobber(.{ @ptrCast(&self.vms[vmi]), val_ref }, self.vms[vmi].requestExternalForwards(val_ref));
            }

            var prev_index: usize = 0;
            while (prev_index != externals.count()) {
                const index = externals.count();
                for (externals.keys()[prev_index..index], externals.values()[prev_index..index]) |head_info, vm_local_externals| {
                    const erased_vm_ptr, _ = head_info;
                    for (vm_local_externals) |requested| {
                        inline for (&self.vms, 0..) |*vm, vm_idx| {
                            if (@as(*anyopaque, @ptrCast(vm)) == erased_vm_ptr) {
                                const erased_child_vm_ptr, const child = vm.getExternalInfo(requested);

                                inline for (&self.vms) |*cvm| {
                                    if (@as(*anyopaque, @ptrCast(cvm)) == erased_child_vm_ptr) {
                                        for (0..vector_sizes[vm_idx]) |i| {
                                            const child_val_ref: ValueRef = .{ .op = child.op, .idx = @enumFromInt(@intFromEnum(child.idx) + i) };
                                            try externals.putNoClobber(.{ erased_child_vm_ptr, child_val_ref }, cvm.requestExternalForwards(child_val_ref));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                prev_index = index;
            }

            // this list could even be cached for faster forward
            // we would need a hashmap to contain the graph for each expr head
            // similar could be done for backward
            const len = externals.count();
            for (0..len) |i| {
                const erased_vm_ptr, const expr_head = externals.keys()[len - i - 1];
                const vm_local_externals = externals.values()[len - i - 1];
                inline for (&self.vms, 0..) |*vm, vm_idx| {
                    if (@as(*anyopaque, @ptrCast(vm)) == erased_vm_ptr) {
                        for (vm_local_externals) |external| {
                            const erased_child_vm_ptr, _ = vm.getExternalInfo(external);
                            inline for (&self.vms) |*cvm| {
                                if (@as(*anyopaque, @ptrCast(cvm)) == erased_child_vm_ptr) {
                                    // TODO impl external recalculate, should be really easy
                                    self.externalRecalculate(ManyRef(vector_sizes[vm_idx], 1, true){ .val_ref = external });
                                }
                            }
                        }

                        try vm.forward(expr_head);
                    }
                }
            }
        }
    };
}
