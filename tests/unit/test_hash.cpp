#include "nn/hash.h"
#include "test.h"

TEST(sha256_abc) {
    const std::string s = "abc";
    auto h = nn::sha256_hex(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(s.data()), s.size()));
    CHECK(h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(sha256_empty) {
    auto h = nn::sha256_hex({});
    CHECK(h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}
