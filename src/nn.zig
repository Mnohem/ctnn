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
        const LEARN_CLAMP: NumType = 1;

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
                // w.* = initWeightVecs(NumType, in_size, out_size, rng);
            }

            return .{
                .weights = mvm.manyNewRows(weights),
                .biases = mvm.newColumn(@as(@Vector(out_size, NumType), @splat(0.0))),
            };
        }

        pub fn applyGradients(self: *Self, mvm: *Mvm) void {
            mvm.getDataPtr(self.biases)[0] -= mvm.getGrad(self.biases)[0];

            for (mvm.getDataPtr(self.weights), mvm.getGrad(self.weights)) |*weight, cost_w| {
                weight.* -= cost_w;
            }
        }
        pub fn debugPrint(self: *const Self, mvm: *const Mvm) void {
            std.debug.print("Weights: ", .{});
            mvm.debugRef(self.weights);
            std.debug.print("Biases: ", .{});
            mvm.debugRef(self.biases);
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
        input_batch_reserve: ManyRef(LayerTypes[0].in_size, max_batch_size, .by_column),
        expected_batch_reserve: ManyRef(LayerTypes[END].out_size, max_batch_size, .by_column),

        const Self = @This();

        pub fn debugPrint(self: *const Self) void {
            inline for (self.layers, 1..) |layer, i| {
                std.debug.print("Layer {}: \n", .{i});
                layer.debugPrint(&self.mvm);
                std.debug.print("\n", .{});
            }
        }

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
                .input_batch_reserve = inputs,
                .expected_batch_reserve = expecteds,
            };
        }

        // batch reserve is made after weights, so this is incorrect
        fn inputReserveIdxtoRef(self: *const Self, idx: u24) ManyRef(LayerTypes[0].in_size, 1, .by_column) {
            return .{ .val_ref = .{ .op = .noop, .idx = @enumFromInt(@intFromEnum(self.input_batch_reserve.val_ref.idx) + idx) } };
        }
        fn expectedReserveIdxtoRef(self: *const Self, idx: u24) ManyRef(LayerTypes[END].out_size, 1, .by_column) {
            return .{ .val_ref = .{ .op = .noop, .idx = @enumFromInt(@intFromEnum(self.expected_batch_reserve.val_ref.idx) + idx) } };
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

        pub fn loss(self: *Self, batch_size: usize) ManyRef(1, 1, .by_row) {
            var total_loss = self.singleLoss(self.inputReserveIdxtoRef(0), self.expectedReserveIdxtoRef(0));
            for (1..batch_size) |i| {
                const sing_loss = self.singleLoss(self.inputReserveIdxtoRef(@intCast(i)), self.expectedReserveIdxtoRef(@intCast(i)));
                total_loss = self.mvm.add(total_loss, sing_loss);
            }
            return self.mvm.sumColumns(total_loss);
        }

        pub fn batchNormalLoss(self: *Self, comptime batch_size: usize) ManyRef(1, 1, .by_row) {
            const FirstLayerOut = LayerTypes[0].Output;

            var first_layer_output: [batch_size]FirstLayerOut = @splat(self.layers[0].calculateOutputs(&self.mvm, self.inputReserveIdxtoRef(0)));
            for (1..batch_size) |i| {
                first_layer_output[i] = self.layers[0].calculateOutputs(&self.mvm, self.inputReserveIdxtoRef(@intCast(i)));
            }

            var first_layer_output_sum: FirstLayerOut = first_layer_output[0];
            for (first_layer_output[1..batch_size]) |ref| {
                first_layer_output_sum = self.mvm.add(first_layer_output_sum, ref);
            }
            const batch_size_splat = self.mvm.valuesWithShapeOf(@TypeOf(first_layer_output_sum), batch_size);
            const neg_mean = self.mvm.neg(self.mvm.elemDiv(first_layer_output_sum, batch_size_splat));

            var variance = self.mvm.elemPowi(self.mvm.add(first_layer_output[0], neg_mean), 2);
            for (first_layer_output[1..batch_size]) |ref| {
                const diff = self.mvm.add(ref, neg_mean);
                variance = self.mvm.add(variance, self.mvm.elemPowi(diff, 2));
            }
            variance = self.mvm.elemDiv(variance, batch_size_splat);

            const a = self.mvm.sqrt(self.mvm.add(variance, self.mvm.valuesWithShapeOf(@TypeOf(variance), 1e-9)));
            for (0..batch_size) |i| {
                first_layer_output[i] = self.mvm.elemDiv(self.mvm.add(first_layer_output[i], neg_mean), a);
            }

            const batched_output = self.recCalculateBatchLayer(batch_size, END, first_layer_output);

            var loss_sum = cost(Mvm, Output, &self.mvm, batched_output[0], self.expectedReserveIdxtoRef(0));
            for (batched_output[1..], 1..) |output, i| {
                loss_sum = self.mvm.add(loss_sum, cost(Mvm, Output, &self.mvm, output, self.expectedReserveIdxtoRef(@intCast(i))));
            }
            return self.mvm.sumColumns(loss_sum);
        }

        fn recCalculateBatchLayer(self: *Self, comptime batch_size: usize, comptime layer_idx: usize, first_layer_output: [batch_size]LayerTypes[0].Output) [batch_size]LayerTypes[layer_idx].Output {
            var result: [batch_size]LayerTypes[layer_idx].Output = undefined;
            return if (layer_idx == 1) blk: {
                for (&result, first_layer_output) |*output, input| {
                    output.* = self.layers[1].calculateOutputs(&self.mvm, input);
                }
                break :blk result;
            } else blk: {
                const batched_output = self.recCalculateBatchLayer(batch_size, layer_idx - 1, first_layer_output);
                for (&result, batched_output) |*res, output| {
                    res.* = self.layers[layer_idx].calculateOutputs(&self.mvm, output);
                }
                break :blk result;
            };
        }

        // To train model, must call loss once before looping over gradient descent
        pub fn batchGradDescent(self: *Self, rng: std.Random, loss_expr: ManyRef(1, 1, .by_row), batch_size: usize, input_vectors: []const @Vector(LayerTypes[0].in_size, NumType), expected_vectors: []const @Vector(LayerTypes[END].out_size, NumType), learn_rate: NumType) !void {
            if (batch_size > max_batch_size) @panic("Batch size is too large");

            const index = rng.intRangeLessThan(usize, 0, input_vectors.len / batch_size);
            @memcpy(self.mvm.getDataPtr(self.input_batch_reserve)[0..batch_size], input_vectors[index * batch_size ..][0..batch_size]);
            @memcpy(self.mvm.getDataPtr(self.expected_batch_reserve)[0..batch_size], expected_vectors[index * batch_size ..][0..batch_size]);

            try self.mvm.forward(loss_expr);
            self.mvm.zeroGrad();
            try self.mvm.backwardWithGrad(loss_expr, learn_rate);
            inline for (&self.layers) |*layer| {
                layer.applyGradients(&self.mvm); // / @as(NumType, @floatFromInt(batch_size)));
            }
        }
    };
}

pub fn cost(Mvm: type, Ref: type, mvm: *Mvm, output_activation: Ref, expected_output: Ref) Ref {
    // return square_loss(Mvm, Ref, mvm, output_activation, expected_output);
    return cross_entropy_loss(Mvm, Ref, mvm, output_activation, expected_output);
}
pub fn square_loss(Mvm: type, Ref: type, mvm: *Mvm, output_activation: Ref, expected_output: Ref) Ref {
    const diff = mvm.sub(output_activation, expected_output);
    return mvm.elemPowi(diff, 2);
}
pub fn cross_entropy_loss(Mvm: type, Ref: type, mvm: *Mvm, output_activation: Ref, expected_output: Ref) Ref {
    return mvm.neg(mvm.elemMul(expected_output, mvm.log(output_activation)));
}

fn randFloatVec(rng: std.Random, comptime size: usize, comptime FType: type) @Vector(size, FType) {
    var result: @Vector(size, FType) = undefined;
    for (0..size) |i| {
        result[i] = rng.float(FType) * 2 - 1;
    }
    return result;
}
fn initWeightVecs(FType: type, comptime neuron_size: usize, comptime neuron_amount: usize, rng: std.Random) @Vector(neuron_size, FType) {
    var result: @Vector(neuron_size, FType) = undefined;
    const mean: FType = @max(-1, -8.0 / @as(FType, @floatFromInt(neuron_amount)));
    for (0..neuron_size) |i| {
        result[i] = rng.floatNorm(FType) + mean;
    }
    return result;
}

// These activation functions take a ManyRef(N, 1, .by_column)
pub fn id(Mvm: type, Ref: type, _: Mvm, ref: Ref) Ref {
    return ref;
}
pub fn sigmoid(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return mvm.elemPowi(mvm.add(mvm.valuesWithShapeOf(Ref, 1.0), mvm.elemExp(mvm.neg(ref))), -1);
}
pub fn sigmoidSafe(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    const zero = mvm.valuesWithShapeOf(Ref, 0.0);
    const one = mvm.valuesWithShapeOf(Ref, 1.0);
    const positive_x = mvm.elemMax(ref, zero);
    const negative_x = mvm.neg(mvm.elemMax(mvm.neg(ref), zero));
    const e_x = mvm.elemExp(negative_x);
    const e_neg_x = mvm.elemExp(mvm.neg(positive_x));
    const positive_term = mvm.elemPowi(mvm.add(one, e_neg_x), -1);
    const negative_term = mvm.elemDiv(e_x, mvm.add(one, e_x));
    const result = mvm.add(positive_term, negative_term);
    return mvm.sub(result, mvm.valuesWithShapeOf(Ref, 0.5));
}
pub fn logit(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return mvm.log(mvm.elemDiv(ref, mvm.sub(mvm.valuesWithShapeOf(Ref, 1.0), ref)));
}
pub fn relu(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return mvm.elemMax(mvm.valuesWithShapeOf(Ref, 1e-12), ref);
}
pub fn reluClip(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return clip(Mvm, Ref, mvm, ref, 0.0, 1e10);
}
pub fn relu6(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return mvm.elemMax(mvm.valuesWithShapeOf(Ref, 6.0), ref);
}
pub fn reluHalfClip(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return clip(Mvm, Ref, mvm, ref, 0.5, 1.0);
}
pub fn relu1(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    return mvm.elemMax(mvm.valuesWithShapeOf(Ref, 1.0), ref);
}
pub fn softmaxSafe(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    const max_input = mvm.maxColumns(ref);
    const norm_x = mvm.sub(ref, mvm.splatIntoColumns(ref.rows, max_input));
    const ex = mvm.elemExp(norm_x);
    const sum_ex = mvm.sumColumns(ex);
    return mvm.elemDiv(ex, mvm.splatIntoColumns(ref.rows, sum_ex));
}
pub fn softmaxSafeClip(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    const result = softmaxSafe(*Mvm, Ref, mvm, ref);
    return clip(Mvm, Ref, mvm, result, 1e-12, 1.0);
}
pub fn softmaxSafeNonZero(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    const result = softmaxSafe(*Mvm, Ref, mvm, ref);
    return mvm.add(result, mvm.valuesWithShapeOf(Ref, 1e-12));
}
pub fn softmaxClip(Mvm: type, Ref: type, mvm: Mvm, ref: Ref) Ref {
    const ex = mvm.elemExp(ref);
    const sum_ex = mvm.sumColumns(ex);
    const result = mvm.elemDiv(ex, mvm.splatIntoColumns(ref.rows, sum_ex));
    return clip(Mvm, Ref, mvm, result, 1e-12, 1.0);
}
fn clip(Mvm: type, Ref: type, mvm: *Mvm, ref: Ref, lower: comptime_float, upper: comptime_float) Ref {
    return mvm.neg(mvm.elemMax(mvm.valuesWithShapeOf(Ref, -upper), mvm.neg(mvm.elemMax(mvm.valuesWithShapeOf(Ref, lower), ref))));
}
