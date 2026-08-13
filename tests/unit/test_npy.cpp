#include "test.h"
#include "util/npy.h"
#include "runtime/tensor_io.h"

#include <cstring>
#include <filesystem>

TEST(npy_roundtrip) {
    nn::RuntimeTensor t;
    t.name = "x";
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints({2, 2});
    float v[4] = {1, 2, 3, 4};
    t.bytes.resize(sizeof(v));
    std::memcpy(t.bytes.data(), v, sizeof(v));
    const auto path = std::filesystem::temp_directory_path() / "nn_test.npy";
    auto st = nn::save_npy(path, t);
    CHECK(st);
    auto loaded = nn::load_npy(path);
    CHECK(loaded);
    CHECK(loaded.value().dtype == nn::DataType::Float32);
    CHECK(loaded.value().bytes.size() == sizeof(v));
}

TEST(npz_roundtrip) {
    nn::RuntimeTensor a;
    a.name = "input0";
    a.dtype = nn::DataType::Float32;
    a.shape = nn::shape_from_ints({2});
    float va[2] = {1, 2};
    a.bytes.resize(sizeof(va));
    std::memcpy(a.bytes.data(), va, sizeof(va));
    nn::RuntimeTensor b;
    b.name = "input1";
    b.dtype = nn::DataType::Float32;
    b.shape = nn::shape_from_ints({2});
    float vb[2] = {3, 4};
    b.bytes.resize(sizeof(vb));
    std::memcpy(b.bytes.data(), vb, sizeof(vb));
    const auto path = std::filesystem::temp_directory_path() / "nn_test.npz";
    CHECK(nn::save_npz(path, {{"input0", a}, {"input1", b}}));
    auto loaded = nn::load_npz(path);
    CHECK(loaded);
    CHECK(loaded.value().size() == 2);
    CHECK(loaded.value().count("input0") == 1);
    CHECK(nn::as_f64(loaded.value().at("input1"))[1] == 4);
}
