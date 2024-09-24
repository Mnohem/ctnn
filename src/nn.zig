const std = @import("std");

pub fn SimpleLayer(
    comptime input_size: usize,
    comptime output_size: usize,
    comptime NType: type, // type of numbers to use
    comptime activation: *const fn (type, @Vector(output_size, NType)) @Vector(output_size, NType),
) type {
    const InputVector = @Vector(input_size, NType);
    const OutputVector = @Vector(output_size, NType);

    return struct {
        pub const Input = InputVector;
        pub const Output = OutputVector;
        pub const NumType = NType;
        pub const in_size = input_size;
        pub const out_size = output_size;

        cost_grad_w: [output_size]InputVector,
        cost_grad_b: OutputVector,
        weights: [output_size]InputVector,
        biases: OutputVector,

        pub fn calculateOutputs(self: *const @This(), inputs: InputVector) OutputVector {
            var weighted_input = self.biases;

            for (self.weights, 0..) |weight, i| {
                weighted_input[i] += @reduce(.Add, weight * inputs);
            }

            const activations = activation(OutputVector, weighted_input);

            return activations;
        }

        pub fn init(rng: std.Random) @This() {
            var weights: [output_size]InputVector = undefined;

            for (&weights) |*w| {
                w.* = randFloatVec(rng, input_size, NumType);
            }

            return .{
                .cost_grad_w = .{@as(InputVector, @splat(0.0))} ** output_size,
                .cost_grad_b = @splat(0.0),
                .weights = weights,
                .biases = @splat(0.0),
            };
        }

        pub fn applyGradients(self: *@This(), learn_rate: NumType) void {
            self.*.biases -= self.cost_grad_b * @as(OutputVector, @splat(learn_rate));

            for (&self.weights, self.cost_grad_w) |*weight, cost_w| {
                weight.* -= cost_w * @as(InputVector, @splat(learn_rate));
            }
        }
    };
}

// LayerTypes is a list of differently sized SimpleLayers
pub fn Model(comptime LayerTypes: []const type) type {
    const END = LayerTypes.len - 1;
    const NumType = LayerTypes[0].NumType;
    const InputVector = LayerTypes[0].Input;
    const OutputVector = LayerTypes[END].Output;

    inline for (LayerTypes[0..END], LayerTypes[1..], 0..) |PrevLayerTy, LayerTy, idx| {
        if (PrevLayerTy.Output != LayerTy.Input) @compileError(std.fmt.comptimePrint("Layers {d} and {d} do not agree in size", .{ idx, idx + 1 }));
    }

    return struct {
        layers: std.meta.Tuple(LayerTypes),

        pub fn init(rand: std.Random) @This() {
            var self: @This() = undefined;
            inline for (0..LayerTypes.len) |idx| {
                self.layers[idx] = LayerTypes[idx].init(rand);
            }
            return self;
        }

        inline fn recCalculateLayer(self: *const @This(), comptime layer_idx: usize, input: InputVector) LayerTypes[layer_idx].Output {
            return if (layer_idx == 0)
                self.layers[0].calculateOutputs(input)
            else
                self.layers[layer_idx].calculateOutputs(self.recCalculateLayer(layer_idx - 1, input));
        }

        pub fn calculateOutputs(self: *const @This(), input: InputVector) OutputVector {
            return self.recCalculateLayer(END, input);
        }

        pub fn singleLoss(self: *const @This(), input: InputVector, expected: OutputVector) NumType {
            const output = self.calculateOutputs(input);

            const result = @reduce(.Add, cost(OutputVector, output, expected));

            return result;
        }

        pub fn loss(self: *const @This(), inputs: []const InputVector, expecteds: []const OutputVector) NumType {
            var total_loss: NumType = 0;
            for (inputs, expecteds) |input, expected| {
                const sing_loss = self.singleLoss(input, expected);
                total_loss += sing_loss;
            }
            return total_loss;
        }

        pub fn gradDescent(self: *@This(), inputs: []const InputVector, expecteds: []const OutputVector, learn_rate: NumType) void {
            const h = 0.001;
            const original_cost = self.loss(inputs, expecteds);

            inline for (&self.layers) |*layer| {
                for (&layer.weights, &layer.cost_grad_w) |*weights, *cost_w| {
                    for (0..@TypeOf(layer.*).in_size) |idx| {
                        weights[idx] += h;
                        const delta_cost = self.loss(inputs, expecteds) - original_cost;
                        weights[idx] -= h;
                        cost_w[idx] = delta_cost / h;
                    }
                }
                for (0..@TypeOf(layer.*).out_size) |idx| {
                    layer.biases[idx] += h;
                    const delta_cost = self.loss(inputs, expecteds) - original_cost;
                    layer.biases[idx] -= h;
                    layer.cost_grad_b[idx] = delta_cost / h;
                }
            }

            inline for (&self.layers) |*layer| {
                layer.applyGradients(learn_rate);
            }
        }

        pub fn batchGradDescent(self: *@This(), rng: std.Random, comptime batch_size: usize, inputs: []const InputVector, expecteds: []const OutputVector, learn_rate: NumType) void {
            var batched_inputs: [batch_size]InputVector = undefined;
            var batched_expecteds: [batch_size]OutputVector = undefined;

            for (0..batch_size) |i| {
                const index = rng.intRangeLessThan(usize, 0, inputs.len);
                batched_inputs[i] = inputs[index];
                batched_expecteds[i] = expecteds[index];
            }

            self.gradDescent(&batched_inputs, &batched_expecteds, learn_rate);
        }
    };
}

pub fn cost(VecTy: type, output_activation: VecTy, expected_output: VecTy) VecTy {
    const diff = output_activation - expected_output;
    return diff * diff;
}

fn randFloatVec(rng: std.Random, comptime size: usize, comptime FType: type) @Vector(size, FType) {
    var result: @Vector(size, FType) = undefined;
    for (0..size) |i| {
        result[i] = rng.float(FType);
    }
    return result;
}

pub fn id(comptime VecTy: type, x: VecTy) VecTy {
    return x;
}
pub fn sigmoid(comptime VecTy: type, x: VecTy) VecTy {
    return @as(VecTy, @splat(1.0)) / (@as(VecTy, @splat(1.0)) + @exp(-x));
}
pub fn relu(comptime VecTy: type, x: VecTy) VecTy {
    return @max(@as(VecTy, @splat(0.0)), x);
}
pub fn softmax(comptime VecTy: type, x: VecTy) VecTy {
    const norm_x = x - @as(VecTy, @splat(@reduce(.Max, x)));
    const ex = @exp(norm_x);
    const sum_ex = @reduce(.Add, ex);
    return ex / @as(VecTy, @splat(sum_ex));
}

// const NanFound = error{
//     InVector,
//     Alone,
// };
//
// fn errorIfNanInVec(vec: anytype) NanFound!@TypeOf(vec) {
//     return if (std.math.isNan(@reduce(.Add, vec)))
//         error.InVector
//     else
//         vec;
// }
// fn errorIfNan(float: anytype) NanFound!@TypeOf(float) {
//     return if (std.math.isNan(float))
//         error.Alone
//     else
//         float;
// }
