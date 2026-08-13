#include "nn/cli.h"
#include "test.h"

TEST(cli_unknown_option) {
    nn::CommandSpec spec;
    spec.name = "inspect";
    spec.flags = {{"summary", '\0', false, "", ""}};
    auto r = nn::parse_command_args({"--foo"}, spec);
    CHECK(!r);
    CHECK(r.error().code() == nn::ErrorCode::UnknownOption);
}

TEST(cli_value_flag) {
    nn::CommandSpec spec;
    spec.flags = {{"op", '\0', true, "TYPE", ""}};
    auto r = nn::parse_command_args({"--op", "Conv", "model.onnx"}, spec);
    CHECK(r);
    CHECK(nn::flag_value(r.value(), "op").value_or("") == "Conv");
    CHECK(r.value().positionals.size() == 1);
}

TEST(cli_repeatable_flag) {
    nn::CommandSpec spec;
    spec.flags = {{"input", '\0', true, "NAME=FILE", ""}};
    auto r = nn::parse_command_args({"--input", "a.npy", "--input", "b.npy"}, spec);
    CHECK(r);
    auto vals = nn::flag_values(r.value(), "input");
    CHECK(vals.size() == 2);
    CHECK(vals[0] == "a.npy");
    CHECK(vals[1] == "b.npy");
    CHECK(nn::flag_value(r.value(), "input").value_or("") == "b.npy");
}
