#include "nn/json.h"
#include "test.h"

TEST(json_roundtrip) {
    auto j = nn::parse_json(R"({"a": 1, "b": [true, false, null, "x"]})");
    CHECK(j);
    CHECK(j.value().is_object());
    CHECK(j.value().at("a").as_number() == 1.0);
    CHECK(j.value().at("b").as_array().size() == 4);
    const std::string d = j.value().dump(false);
    auto j2 = nn::parse_json(d);
    CHECK(j2);
}

TEST(json_error) {
    auto j = nn::parse_json("{");
    CHECK(!j);
}
