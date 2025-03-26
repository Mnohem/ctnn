const std = @import("std");
const nn = @import("nn.zig");
const mnist = @import("idx.zig");
const manygrad = @import("manygrad.zig");

const F = f32;

const MNIST_IMAGE_SIZE = 28 * 28;
const EPSILON: F = 1e-9;

const Mvm = manygrad.ManyValueManager(F, &.{ MNIST_IMAGE_SIZE, 32, 10, 1 });

const LayerTypes = [_]type{
    nn.SimpleLayer(Mvm, F, MNIST_IMAGE_SIZE, 32, nn.relu),
    nn.SimpleLayer(Mvm, F, 32, 10, nn.softmaxSafeNonZero),
};

const Model = nn.Model(10, &LayerTypes);

const InputVector = @Vector(LayerTypes[0].in_size, F);
const OutputVector = @Vector(LayerTypes[LayerTypes.len - 1].out_size, F);

pub fn test_run() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const train_image_size, const train_image_data = comptime mnist.openIdxFile("training/train-images-idx3-ubyte");
    _, const train_labels = comptime mnist.openIdxFile("training/train-labels-idx1-ubyte");

    const steps = 5000;
    const learn_rate = 0.001;
    const batch_size = 10;
    const training_size: usize = @intCast(train_image_size);

    var inputs = try allocator.alloc(InputVector, training_size);
    var expecteds = try allocator.alloc(OutputVector, training_size);

    for (0..training_size) |i| {
        const offset = MNIST_IMAGE_SIZE * i;
        for (train_image_data[offset .. MNIST_IMAGE_SIZE + offset], 0..) |pixel, j| {
            inputs[i][j] = @as(F, @floatFromInt(pixel)) / 255.0;
        }
        if (i % batch_size == batch_size - 1) {
            var sum: InputVector = @splat(0);
            for (inputs[i + 1 - batch_size .. i + 1]) |vec| {
                sum += vec;
            }
            const mean = sum / @as(InputVector, @splat(batch_size));

            var variance: InputVector = @splat(0);
            for (inputs[i + 1 - batch_size .. i + 1]) |vec| {
                const diff = vec - mean;
                variance += diff * diff;
            }
            variance /= @as(InputVector, @splat(batch_size));

            for (i + 1 - batch_size..i + 1) |j| {
                inputs[j] = (inputs[j] - mean) / @sqrt(variance + @as(InputVector, @splat(EPSILON)));
            }
        }

        expecteds[i] = @splat(0);
        expecteds[i][@as(usize, train_labels[i])] = 1;
    }

    const seed: u128 = @bitCast(std.time.nanoTimestamp());
    var prng = std.Random.DefaultPrng.init(@truncate(seed));
    const rng = prng.random();

    var model = Model.init(allocator, rng);

    const loss = model.batchNormalLoss(batch_size);
    // model.mvm.treePrint(loss, 0);

    std.debug.print("-- Running Gradient Descent for {d} batches of size {d}\n\n", .{ steps, batch_size });
    for (0..steps) |i| {
        try model.batchGradDescent(rng, loss, batch_size, inputs, expecteds, learn_rate);

        if (i % 100 == 0) {
            model.layers[0].debugPrint(&model.mvm);
            model.layers[1].debugPrint(&model.mvm);
        }
    }
    std.debug.print("-- Finished {} steps of Gradient Descent\n", .{steps});

    const test_image_size, const test_image_data = comptime mnist.openIdxFile("testing/t10k-images-idx3-ubyte");
    _, const test_labels = comptime mnist.openIdxFile("testing/t10k-labels-idx1-ubyte");

    const testing_size: usize = @intCast(test_image_size);

    var test_inputs = try allocator.alloc(InputVector, testing_size);
    var test_expecteds = try allocator.alloc(OutputVector, testing_size);

    for (0..testing_size) |i| {
        const offset = MNIST_IMAGE_SIZE * i;
        for (test_image_data[offset .. MNIST_IMAGE_SIZE + offset], 0..) |pixel, j| {
            test_inputs[i][j] = @as(F, @floatFromInt(pixel)) / 255;
        }

        // one hot encoding
        test_expecteds[i] = @splat(0);
        test_expecteds[i][@as(usize, test_labels[i])] = 1;
    }

    var correct: usize = 0;
    for (0..testing_size) |i| {
        const input = model.mvm.newColumn(test_inputs[i]);
        const output = model.mvm.getData(model.calculateOutputs(input))[0];
        if (std.math.isNan(@reduce(.Add, output))) @panic("Nan in output during testing");

        const max_output = @reduce(.Max, output);
        const more_than_one_max = @reduce(.Add, @select(f32, output == @as(@TypeOf(output), @splat(max_output)), @as(@TypeOf(output), @splat(1)), @as(@TypeOf(output), @splat(0)))) > 1;

        const is_correct = if (more_than_one_max) false else max_output == @reduce(.Max, output * test_expecteds[i]);

        correct += @intFromBool(is_correct);

        if ((i + 1) % 100 == 0) {
            std.debug.print("First {} Accuracy: {d:.2}%\n", .{ i + 1, 100 * (@as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(i + 1))) });
        }
    }
}

pub fn main() !void {
    // @setFloatMode(.optimized);

    return test_run();
}
