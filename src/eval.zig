const std = @import("std");
const grad = @import("grad.zig");
const ascii = std.ascii;

const CharToken = enum(u8) {
    lparen = '(',
    rparen = ')',
    lbracket = '[',
    rbracket = ']',
    slash = '/',
    minus = '-',
    plus = '+',
    star = '*',
    caret = '^',
};

pub fn charParse(char: u8) ?CharToken {
    return switch (char) {
        '(' => .lparen,
        ')' => .rparen,
        '[' => .lbracket,
        ']' => .rbracket,
        '/' => .slash,
        '-' => .minus,
        '+' => .plus,
        '*' => .star,
        '^' => .caret,
        else => null,
    };
}
const Token = union(enum) {
    char: CharToken,
    ident: []const u8,
    number: f64,
};

const TokenIter = struct {
    expr: []const u8,
    curr_idx: usize = 0,

    fn next(self: *@This()) ?Token {
        if (self.curr_idx >= self.expr.len) return null;

        const local_state: enum { in_space, in_ident, in_num } = .in_space;
        var local_idx: usize = 0;

        state: switch (local_state) {
            .in_space => {
                if (self.curr_idx >= self.expr.len) {
                    return null;
                } else if (charParse(self.expr[self.curr_idx])) |ct| {
                    self.curr_idx += 1;
                    return .{ .char = ct };
                } else if (ascii.isAlphabetic(self.expr[self.curr_idx])) {
                    local_idx = self.curr_idx;
                    self.curr_idx += 1;
                    continue :state .in_ident;
                } else if (ascii.isDigit(self.expr[self.curr_idx])) {
                    local_idx = self.curr_idx;
                    self.curr_idx += 1;
                    continue :state .in_num;
                } else if (ascii.isWhitespace(self.expr[self.curr_idx])) {
                    self.curr_idx += 1;
                    continue :state .in_space;
                } else {
                    std.debug.panic("Invalid character: {}", .{self.expr[self.curr_idx]});
                }
            },
            .in_ident => {
                if (self.curr_idx + 1 >= self.expr.len) {
                    self.curr_idx += 1;
                    return .{ .ident = self.expr[local_idx..] };
                } else if (charParse(self.expr[self.curr_idx]) != null or ascii.isWhitespace(self.expr[self.curr_idx])) {
                    return .{ .ident = self.expr[local_idx..self.curr_idx] };
                } else if (ascii.isAlphanumeric(self.expr[self.curr_idx])) {
                    self.curr_idx += 1;
                    continue :state .in_ident;
                } else {
                    std.debug.panic("Invalid character: {}", .{self.expr[self.curr_idx]});
                }
            },
            .in_num => {
                if (self.curr_idx + 1 >= self.expr.len) {
                    self.curr_idx += 1;
                    return .{ .number = std.fmt.parseFloat(f64, self.expr[local_idx..]) catch |err| {
                        std.debug.panic("Invalid Expression: {}, {s}", .{ err, self.expr[local_idx..] });
                    } };
                } else if (charParse(self.expr[self.curr_idx]) != null or ascii.isWhitespace(self.expr[self.curr_idx]) or self.curr_idx + 1 == self.expr.len) {
                    return .{ .number = std.fmt.parseFloat(f64, self.expr[local_idx..self.curr_idx]) catch |err| {
                        std.debug.panic("Invalid Expression: {}, {s}", .{ err, self.expr[local_idx..self.curr_idx] });
                    } };
                } else if (ascii.isDigit(self.expr[self.curr_idx]) or self.expr[self.curr_idx] == '.') {
                    self.curr_idx += 1;
                    continue :state .in_num;
                } else {
                    std.debug.panic("Invalid character: {}", .{self.expr[self.curr_idx]});
                }
            },
        }
    }
};

const Op = enum {
    add,
    mul,
    sub,
    div,
    pow,
};
fn power(self: Op) [2]u8 {
    return switch (self) {
        .add, .sub => [2]u8{ 1, 2 },
        .mul, .div => [2]u8{ 3, 4 },
        .pow => [2]u8{ 6, 5 },
    };
}
fn fromCharToken(ct: CharToken) !Op {
    return switch (ct) {
        .plus => .add,
        .star => .mul,
        .minus => .sub,
        .slash => .div,
        .caret => .pow,
        else => error.NotOperator,
    };
}

pub fn VarRefs(VarStruct: type, ValueRef: type) type {
    if (VarStruct == @TypeOf(.{})) return ValueRef;
    var var_struct = @typeInfo(VarStruct).@"struct";
    var_struct.is_tuple = true;
    for (var_struct.fields, 1..) |*field, i| {
        field.type = ValueRef;
        field.alignment = @alignOf(ValueRef);
        field.name = std.fmt.comptimePrint("{d}", .{i});
    }
    var_struct.fields = .{.{ .name = "0", .type = ValueRef, .is_comptime = false, .alignment = @alignOf(ValueRef), .default_value = null }} ++ var_struct.fields;
    return @Type(.{ .@"struct" = var_struct });
}

pub fn parse(Data: type, vm: *grad.ValueManager(Data), comptime expr: []const u8, variables: anytype) !VarRefs(@TypeOf(variables), grad.ValueManager(Data).ValueRef) {
    const ExprAndVars = VarRefs(@TypeOf(variables), grad.ValueManager(Data).ValueRef);
    const data_len = switch (@typeInfo(Data)) {
        .float => 1,
        .vector => |v| v.len,
        else => unreachable,
    };

    comptime var tokens_var: []const Token = &[_]Token{};
    comptime var token_iter = TokenIter{ .expr = expr };
    inline while (comptime token_iter.next()) |token| {
        tokens_var = tokens_var ++ &[1]Token{token};
        const contains_ident = switch (token) {
            .ident => |ident| for (std.meta.fields(@TypeOf(variables))) |field| {
                if (std.mem.eql(ident, field.name)) {
                    break true;
                }
            } else false,
            else => continue,
        };
        if (!contains_ident) {
            @compileError(std.fmt.comptimePrint("Free variable \"{}\"", token.ident));
        }
    }

    var result: ExprAndVars = undefined;

    if (@TypeOf(variables) == @TypeOf(.{})) {
        var tokens = tokens_var;
        return parse_bp(vm, data_len, &tokens, 0, &[0][]const u8{}, &[0]grad.ValueManager(Data).ValueRef{});
    }

    var var_names: [std.meta.fields(variables).len][]const u8 = undefined;
    var var_value_refs: [std.meta.fields(variables).len]grad.ValueManager(Data).ValueRef = undefined;
    for (std.meta.fields(variables), 0..) |field, i| {
        var_names[i] = field.name;
        if (field.type == Data) {
            var_value_refs[i] = vm.new(@field(variables, field.name));
        } else if (field.type == grad.ValueManager(Data).ValueRef) {
            var_value_refs[i] = @field(variables, field.name);
        } else @compileError(std.fmt.comptimePrint("Expected type {} or {} for variable \"{s}\"", .{ Data, grad.ValueManager(Data).ValueRef, field.name }));
    }

    var tokens = tokens_var;
    result[0] = try parse_bp(vm, data_len, &tokens, 0, &var_names, &var_value_refs);
}

fn parse_bp(vm: anytype, data_len: comptime_int, tokens: *[]const Token, min_bp: u8, var_names: []const []const u8, var_value_refs: []const @TypeOf(vm.*).ValueRef) !@TypeOf(vm.*).ValueRef {
    if (tokens.len != 0) {
        var lhs = switch (tokens.*[0]) {
            .number => |v| if (data_len == 1) vm.new(@floatCast(v)) else return error.Unimplemented,
            .ident => |id| for (var_names, var_value_refs) |name, ref| {
                if (std.mem.eql(u8, name, id)) break ref;
            } else unreachable,
            else => return error.Unimplemented,
        };
        tokens.* = tokens.*[1..];

        for (tokens.*) |token| {
            const op = try fromCharToken(token.char);

            const bp = power(op);
            if (bp[0] < min_bp) break;

            tokens.* = tokens.*[1..];
            const rhs = try parse_bp(vm, data_len, tokens, bp[1], var_names, var_value_refs);

            lhs = switch (op) {
                .add => vm.add(lhs, rhs),
                .mul => vm.mul(lhs, rhs),
                .sub => vm.sub(lhs, rhs),
                .div => vm.div(lhs, rhs),
                .pow => return error.Unimplemented, // we will make pow right associative so that we can parse the power
            };
        }

        return lhs;
    }
    return error.Unimplemented;
}

test "Basic Eval Test" {
    var vm = try grad.ValueManager(f32).init(std.testing.allocator, 10);
    defer vm.deinit();

    const expr = "1+1 * 2/4";
    var iter = TokenIter{ .expr = expr };
    const result = try vm.eval(expr, .{});

    try std.testing.expectEqual(Token{ .number = 1 }, iter.next());
    try std.testing.expectEqual(Token{ .char = .plus }, iter.next());
    try std.testing.expectEqual(Token{ .number = 1 }, iter.next());
    try std.testing.expectEqual(Token{ .char = .star }, iter.next());
    try std.testing.expectEqual(Token{ .number = 2 }, iter.next());
    try std.testing.expectEqual(Token{ .char = .slash }, iter.next());
    try std.testing.expectEqual(Token{ .number = 4 }, iter.next());
    try std.testing.expectEqual(null, iter.next());

    try std.testing.expectApproxEqAbs(1.5, vm.forward(result), 0.001);
}
test "Basic Tokenizer Test" {
    const expr = "1+1 * (2^x) + identifier / 0.0";

    var iter = TokenIter{ .expr = expr };

    try std.testing.expectEqual(Token{ .number = 1 }, iter.next());
    try std.testing.expectEqual(Token{ .char = .plus }, iter.next());
    try std.testing.expectEqual(Token{ .number = 1 }, iter.next());
    try std.testing.expectEqual(Token{ .char = .star }, iter.next());
    try std.testing.expectEqual(Token{ .char = .lparen }, iter.next());
    try std.testing.expectEqual(Token{ .number = 2 }, iter.next());
    try std.testing.expectEqual(Token{ .char = .caret }, iter.next());
    try std.testing.expectEqualDeep(Token{ .ident = "x" }, iter.next());
    try std.testing.expectEqual(Token{ .char = .rparen }, iter.next());
    try std.testing.expectEqual(Token{ .char = .plus }, iter.next());
    try std.testing.expectEqualDeep(Token{ .ident = "identifier" }, iter.next());
    try std.testing.expectEqual(Token{ .char = .slash }, iter.next());
    try std.testing.expectEqual(Token{ .number = 0 }, iter.next());
    try std.testing.expectEqual(null, iter.next());
}
test "Tokenizer Whitespace Test" {
    const expr = "\t\n\t * (2^x)      + \tidentifier / 0.0\t  ";

    var iter = TokenIter{ .expr = expr };

    try std.testing.expectEqual(Token{ .char = .star }, iter.next());
    try std.testing.expectEqual(Token{ .char = .lparen }, iter.next());
    try std.testing.expectEqual(Token{ .number = 2 }, iter.next());
    try std.testing.expectEqual(Token{ .char = .caret }, iter.next());
    try std.testing.expectEqualDeep(Token{ .ident = "x" }, iter.next());
    try std.testing.expectEqual(Token{ .char = .rparen }, iter.next());
    try std.testing.expectEqual(Token{ .char = .plus }, iter.next());
    try std.testing.expectEqualDeep(Token{ .ident = "identifier" }, iter.next());
    try std.testing.expectEqual(Token{ .char = .slash }, iter.next());
    try std.testing.expectEqual(Token{ .number = 0 }, iter.next());
    try std.testing.expectEqual(null, iter.next());
}
