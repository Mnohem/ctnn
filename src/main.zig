const std = @import("std");
const nn = @import("nn.zig");
const mnist = @import("idx.zig");

const F = f32;

const MNIST_IMAGE_SIZE = 28 * 28;

const LayerTypes = [_]type{
    nn.SimpleLayer(MNIST_IMAGE_SIZE, 32, F, nn.sigmoid),
    nn.SimpleLayer(32, 10, F, nn.softmax),
};

const Model = nn.Model(&LayerTypes);

const InputVector = LayerTypes[0].Input;
const OutputVector = LayerTypes[LayerTypes.len - 1].Output;

pub fn test_run() !void {
    const TRAINING_SIZE = 60000;
    _, const train_image_data = mnist.openIdxFile("training/train-images-idx3-ubyte");
    _, const train_labels = mnist.openIdxFile("training/train-labels-idx1-ubyte");

    var inputs: [TRAINING_SIZE]InputVector = undefined;
    var expecteds = std.mem.zeroes([TRAINING_SIZE]OutputVector);

    for (0..TRAINING_SIZE) |i| {
        const offset = MNIST_IMAGE_SIZE * i;
        for (train_image_data[offset .. MNIST_IMAGE_SIZE + offset], 0..) |pixel, j| {
            inputs[i][j] = @as(F, @floatFromInt(pixel)) / 255;
        }

        expecteds[i][@as(usize, train_labels[i])] = 1;
    }

    const seed: u128 = @bitCast(std.time.nanoTimestamp());
    var prng = std.Random.DefaultPrng.init(@truncate(seed));

    var model = Model.init(prng.random());

    const steps = 1000;
    const learn_rate = 0.1;
    const batch_size = 10;

    std.debug.print("-- Running Gradient Descent for {d} batches of size {d}\n\n", .{ steps, batch_size });
    for (0..steps) |i| {
        model.batchGradDescent(prng.random(), batch_size, &inputs, &expecteds, learn_rate);
        if (i % 100 == 0) {
            std.debug.print("Iteration {}\n", .{i});
            std.debug.print("-- Output 0 is {}\n", .{model.calculateOutputs(inputs[0])});
            std.debug.print("-- Expected 0 is {}\n", .{expecteds[0]});
            std.debug.print("-- Model loss is {}\n", .{model.loss(&inputs, &expecteds)});
        }
    }
    std.debug.print("-- Finished {} steps of Gradient Descent\n", .{steps});

    const test_image_size, const test_image_data = mnist.openIdxFile("testing/t10k-images-idx3-ubyte");
    _, const test_labels = mnist.openIdxFile("testing/t10k-labels-idx1-ubyte");

    var test_inputs: [TRAINING_SIZE]InputVector = undefined;
    var test_expecteds = std.mem.zeroes([TRAINING_SIZE]OutputVector);
    for (0..@intCast(test_image_size[0])) |i| {
        const offset = MNIST_IMAGE_SIZE * i;
        for (test_image_data[offset .. MNIST_IMAGE_SIZE + offset], 0..) |pixel, j| {
            test_inputs[i][j] = @as(F, @floatFromInt(pixel)) / 255;
        }

        test_expecteds[i][@as(usize, test_labels[i])] = 1;
    }

    var correct: usize = 0;
    for (0..@intCast(test_image_size[0])) |i| {
        const output = model.calculateOutputs(test_inputs[i]);

        const is_correct = @reduce(.And, @round(output) == test_expecteds[i]);

        correct += @intFromBool(is_correct);

        if (i % 100 == 0) {
            std.debug.print("Accuracy: {:.2}%\n", .{100 * @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(test_image_size[0]))});
        }
    }
}

pub fn main() !void {
    // Expand stack size, we run out by default
    // We might need to use libc for cross compat later
    std.posix.setrlimit(.STACK, .{
        .cur = 0x0000000020000000,
        .max = 0x0000000020000000,
    }) catch |err| {
        std.debug.panic("Could not set stack size: {}", .{err});
    };

    return test_run();
}
