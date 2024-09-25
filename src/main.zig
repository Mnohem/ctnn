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
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var alc = arena.allocator();

    const train_image_size, const train_image_data = comptime mnist.openIdxFile("training/train-images-idx3-ubyte");
    _, const train_labels = comptime mnist.openIdxFile("training/train-labels-idx1-ubyte");

    const training_size: usize = @intCast(train_image_size);

    var inputs = try alc.alloc(InputVector, training_size);
    var expecteds = try alc.alloc(OutputVector, training_size);

    for (0..training_size) |i| {
        const offset = MNIST_IMAGE_SIZE * i;
        for (train_image_data[offset .. MNIST_IMAGE_SIZE + offset], 0..) |pixel, j| {
            inputs[i][j] = @as(F, @floatFromInt(pixel)) / 255;
        }

        expecteds[i] = @splat(0);
        expecteds[i][@as(usize, train_labels[i])] = 1;
    }

    const seed: u128 = @bitCast(std.time.nanoTimestamp());
    var prng = std.Random.DefaultPrng.init(@truncate(seed));
    const rng = prng.random();

    var model = Model.init(rng);

    const steps = 1000;
    const learn_rate = 0.1;
    const batch_size = 10;

    std.debug.print("-- Running Gradient Descent for {d} batches of size {d}\n\n", .{ steps, batch_size });
    for (0..steps) |i| {
        model.batchGradDescent(rng, batch_size, inputs, expecteds, learn_rate);
        if (i % 100 == 0) {
            std.debug.print("Iteration {}\n", .{i});
            std.debug.print("-- Output 0 is {}\n", .{model.calculateOutputs(&inputs[0])});
            std.debug.print("-- Expected 0 is {}\n", .{expecteds[0]});
            std.debug.print("-- Model loss is {}\n", .{model.loss(inputs, expecteds)});
        }
    }
    std.debug.print("-- Finished {} steps of Gradient Descent\n", .{steps});

    const test_image_size, const test_image_data = comptime mnist.openIdxFile("testing/t10k-images-idx3-ubyte");
    _, const test_labels = comptime mnist.openIdxFile("testing/t10k-labels-idx1-ubyte");

    const testing_size: usize = @intCast(test_image_size);

    var test_inputs = try alc.alloc(InputVector, testing_size);
    var test_expecteds = try alc.alloc(OutputVector, testing_size);

    for (0..testing_size) |i| {
        const offset = MNIST_IMAGE_SIZE * i;
        for (test_image_data[offset .. MNIST_IMAGE_SIZE + offset], 0..) |pixel, j| {
            test_inputs[i][j] = @as(F, @floatFromInt(pixel)) / 255;
        }

        test_expecteds[i] = @splat(0);
        test_expecteds[i][@as(usize, test_labels[i])] = 1;
    }

    var correct: usize = 0;
    for (0..testing_size) |i| {
        const output = model.calculateOutputs(&test_inputs[i]);

        const is_correct = @reduce(.And, @round(output) == test_expecteds[i]);

        correct += @intFromBool(is_correct);

        if (i % 100 == 0) {
            std.debug.print("Accuracy: {:.2}%\n", .{100 * @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(testing_size))});
        }
    }
}

pub fn main() !void {
    // Expand stack size, we run out by default
    // We might need to use libc for cross compat later
    // std.posix.setrlimit(.STACK, .{
    //     .cur = 0x0000000020000000,
    //     .max = 0x0000000020000000,
    // }) catch |err| {
    //     std.debug.panic("Could not set stack size: {}", .{err});
    // };

    @setFloatMode(.optimized);

    return test_run();
}
