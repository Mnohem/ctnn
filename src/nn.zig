const std = @import("std");
const manygrad = @import("manygrad.zig");
const ManyRef = manygrad.ManyRef;

pub fn SimpleLayer(
    comptime Mvm: type, // Type of the ManyValueManager we use
    comptime NType: type, // type of numbers to use
    comptime input_size: usize,
    comptime output_size: usize,
    comptime activation: *const fn (type, type, *Mvm, ManyRef(output_size, 1, .by_column)) ManyRef(output_size, 1, .by_column),
) type {
    const InputRef = ManyRef(input_size, 1, .by_column);
    const OutputRef = ManyRef(output_size, 1, .by_column);
    const WeightRef = ManyRef(output_size, input_size, .by_row);

    return struct {
        pub const Input = InputRef;
        pub const Output = OutputRef;
        pub const NumType = NType;
        pub const in_size = input_size;
        pub const out_size = output_size;

        weights: WeightRef,
        biases: OutputRef,

        const Self = @This();

        pub fn calculateOutputs(self: *const Self, mvm: *Mvm, inputs: Input) Output {
            const weighted_input = mvm.add(self.biases, mvm.matVecMul(self.weights, inputs));

            const activations = activation(Mvm, Output, mvm, weighted_input);

            return activations;
        }

        pub fn init(mvm: *Mvm, rng: std.Random) Self {
            var weights: [out_size]@Vector(in_size, NumType) = undefined;

            for (&weights) |*w| {
                w.* = randFloatVec(rng, in_size, NumType);
            }

            return .{
                .weights = mvm.manyNewRows(weights),
                .biases = mvm.newColumn(@as(@Vector(out_size, NumType), @splat(0.0))),
            };
        }

        pub fn applyGradients(self: *Self, mvm: *Mvm, learn_rate: NumType) void {
            mvm.getDataPtr(self.biases)[0] -= mvm.getGrad(self.biases)[0] * @as(@Vector(out_size, NumType), @splat(learn_rate));

            for (mvm.getDataPtr(self.weights), mvm.getGrad(self.weights)) |*weight, cost_w| {
                weight.* -= cost_w * @as(@Vector(in_size, NumType), @splat(learn_rate));
            }
        }
    };
}

// LayerTypes is a list of differently sized SimpleLayers
pub fn Model(comptime max_batch_size: usize, comptime LayerTypes: []const type) type {
    const END = LayerTypes.len - 1;
    const NumType = LayerTypes[0].NumType;
    const Input = LayerTypes[0].Input;
    const Output = LayerTypes[END].Output;

    comptime var unique_sizes = [1]comptime_int{0} ** (LayerTypes.len + 1);
    unique_sizes[0] = LayerTypes[0].in_size;
    comptime var num_unique_sizes: usize = 1;

    inline for (LayerTypes[0..END], LayerTypes[1..], 0..) |PrevLayerTy, LayerTy, idx| {
        if (PrevLayerTy.Output != LayerTy.Input) @compileError(std.fmt.comptimePrint("Layers {d} and {d} do not agree in size", .{ idx, idx + 1 }));

        const unique = for (unique_sizes[0..num_unique_sizes]) |size| {
            if (size == LayerTy.in_size) break false;
        } else true;
        if (unique) {
            unique_sizes[num_unique_sizes] = LayerTy.in_size;
            num_unique_sizes += 1;
        }
    }

    const unique = for (unique_sizes[0..num_unique_sizes]) |size| {
        if (size == LayerTypes[END].out_size) break false;
    } else true;
    if (unique) {
        unique_sizes[num_unique_sizes] = LayerTypes[END].out_size;
        num_unique_sizes += 1;
    }

    const uniques = unique_sizes[0..num_unique_sizes].* ++ [1]comptime_int{1};
    const Mvm = manygrad.ManyValueManager(LayerTypes[0].NumType, &uniques);

    return struct {
        layers: std.meta.Tuple(LayerTypes),
        mvm: Mvm,
        input_batch_reserve: []@Vector(LayerTypes[0].in_size, NumType),
        expected_batch_reserve: []@Vector(LayerTypes[END].out_size, NumType),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, rand: std.Random) Self {
            var layers: std.meta.Tuple(LayerTypes) = undefined;
            var mvm = Mvm.init(allocator, 1) catch unreachable;

            inline for (0..LayerTypes.len) |idx| {
                layers[idx] = LayerTypes[idx].init(&mvm, rand);
            }

            const inputs = mvm.manyNewColumns(std.mem.zeroes([max_batch_size]@Vector(LayerTypes[0].in_size, NumType)));
            const expecteds = mvm.manyNewColumns(std.mem.zeroes([max_batch_size]@Vector(LayerTypes[END].out_size, NumType)));

            return Self{
                .layers = layers,
                .mvm = mvm,
                .input_batch_reserve = mvm.getDataPtr(inputs),
                .expected_batch_reserve = mvm.getDataPtr(expecteds),
            };
        }

        fn inputReserveIdxtoRef(idx: u24) ManyRef(LayerTypes[0].in_size, 1, .by_column) {
            return .{ .val_ref = .{ .op = .noop, .idx = @enumFromInt(idx) } };
        }
        fn expectedReserveIdxtoRef(idx: u24) ManyRef(LayerTypes[END].out_size, 1, .by_column) {
            const reserve_start = if (LayerTypes[0].in_size == LayerTypes[END].out_size) max_batch_size else 0;
            return .{ .val_ref = .{ .op = .noop, .idx = @enumFromInt(idx + reserve_start) } };
        }

        inline fn recCalculateLayer(self: *Self, comptime layer_idx: usize, input: Input) LayerTypes[layer_idx].Output {
            return if (layer_idx == 0)
                self.layers[0].calculateOutputs(&self.mvm, input)
            else
                self.layers[layer_idx].calculateOutputs(&self.mvm, self.recCalculateLayer(layer_idx - 1, input));
        }

        pub fn calculateOutputs(self: *Self, input: Input) Output {
            return self.recCalculateLayer(END, input);
        }

        pub fn singleLoss(self: *Self, input: Input, expected: Output) Output {
            const output = self.calculateOutputs(input);

            return cost(Mvm, Output, &self.mvm, output, expected);
        }

        pub fn loss(self: *Self, batch_size: usize) Output {
            var total_loss = self.singleLoss(inputReserveIdxtoRef(0), expectedReserveIdxtoRef(0));
            for (1..batch_size) |i| {
                const sing_loss = self.singleLoss(inputReserveIdxtoRef(@intCast(i)), expectedReserveIdxtoRef(@intCast(i)));
                total_loss = self.mvm.add(total_loss, sing_loss);
            }
            return total_loss;
        }

        // To train model, must call loss once before looping over gradient descent
        pub fn batchGradDescent(self: *Self, rng: std.Random, loss_expr: ManyRef(LayerTypes[END].out_size, 1, .by_column), batch_size: usize, input_vectors: []const @Vector(LayerTypes[0].in_size, NumType), expected_vectors: []const @Vector(LayerTypes[END].out_size, NumType), learn_rate: NumType) !void {
            if (batch_size > max_batch_size or max_batch_size < 1) @panic("Batch size is too large");

            const index = rng.intRangeLessThan(usize, 0, input_vectors.len - batch_size);
            @memcpy(self.input_batch_reserve[0..batch_size], input_vectors[index..][0..batch_size]);
            @memcpy(self.expected_batch_reserve[0..batch_size], expected_vectors[index..][0..batch_size]);

            try self.mvm.forward(loss_expr);
            try self.mvm.backward(loss_expr);
            inline for (&self.layers) |*layer| {
                layer.applyGradients(&self.mvm, learn_rate);
            }
            self.mvm.zeroGrad();
        }
    };
}

pub fn cost(Mvm: type, Ref: type, mvm: *Mvm, output_activation: Ref, expected_output: Ref) Ref {
    const diff = mvm.sub(output_activation, expected_output);
    return mvm.elemPowi(diff, 2);
}

fn randFloatVec(rng: std.Random, comptime size: usize, comptime FType: type) @Vector(size, FType) {
    var result: @Vector(size, FType) = undefined;
    for (0..size) |i| {
        result[i] = rng.float(FType);
    }
    return result;
}

// These activation functions take a ManyRef(N, 1, .by_column)
pub fn id(Mvm: type, Ref: type, _: Mvm, ref: Ref) Ref {
    return ref;
}
pub fn sigmoid(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return mvm.elemPowi(mvm.add(mvm.valuesWithShapeOf(@TypeOf(ref), 1.0), mvm.elemExp(mvm.neg(ref))), -1);
}
pub fn relu(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return mvm.elemMax(mvm.valuesWithShapeOf(@TypeOf(ref), 0.0), ref);
}
pub fn softmax(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    const norm_x = mvm.sub(ref, mvm.splatIntoColumns(ref.rows, mvm.maxColumns(ref)));
    const ex = mvm.elemExp(norm_x);
    const sum_ex = mvm.sumColumns(ex);
    return mvm.elemDiv(ex, mvm.splatIntoColumns(ref.rows, sum_ex));
}
// pub fn softmax(comptime VecTy: type, x: VecTy) VecTy {
//     const norm_x = x - @as(VecTy, @splat(@reduce(.Max, x)));
//     const ex = @exp(norm_x);
//     const sum_ex = @reduce(.Add, ex);
//     return ex / @as(VecTy, @splat(sum_ex));
// }
