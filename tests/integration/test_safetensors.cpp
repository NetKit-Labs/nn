#include "nn/format.h"
#include "nn/json.h"
#include "test.h"

#include <fstream>

TEST(safetensors_header) {
    nn::Json header = nn::Json::object();
    nn::Json t = nn::Json::object();
    t["dtype"] = "F32";
    nn::Json shape = nn::Json::array();
    shape.push(2);
    t["shape"] = shape;
    nn::Json off = nn::Json::array();
    off.push(0);
    off.push(8);
    t["data_offsets"] = off;
    header["w"] = t;
    const std::string h = header.dump(false);
    const auto path = std::filesystem::temp_directory_path() / "nn_tiny.safetensors";
    std::ofstream out(path, std::ios::binary);
    uint64_t n = h.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    float vals[2] = {1.f, 2.f};
    out.write(reinterpret_cast<const char*>(vals), 8);
    out.close();
    auto m = nn::load_model(path);
    CHECK(m);
    CHECK(m.value().source_format == "safetensors");
    const nn::Graph* g = nn::primary_graph(m.value());
    CHECK(g);
    CHECK(g->tensors.size() == 1);
    CHECK(g->tensors[0].name == "w");
}
