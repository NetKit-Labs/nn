#include "nn/hash.h"

#include "nn/error.h"
#include "nn/result.h"

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace nn {
namespace {

constexpr uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }

}  // namespace

Sha256::Sha256() {
    state_[0] = 0x6a09e667u;
    state_[1] = 0xbb67ae85u;
    state_[2] = 0x3c6ef372u;
    state_[3] = 0xa54ff53au;
    state_[4] = 0x510e527fu;
    state_[5] = 0x9b05688cu;
    state_[6] = 0x1f83d9abu;
    state_[7] = 0x5be0cd19u;
}

void Sha256::transform(const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + S1 + ch + kSha256K[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(std::span<const uint8_t> data) {
    bit_len_ += static_cast<uint64_t>(data.size()) * 8u;
    std::size_t i = 0;
    if (buffer_len_ != 0) {
        while (i < data.size() && buffer_len_ < 64) {
            buffer_[buffer_len_++] = data[i++];
        }
        if (buffer_len_ == 64) {
            transform(buffer_);
            buffer_len_ = 0;
        }
    }
    while (i + 64 <= data.size()) {
        transform(data.data() + i);
        i += 64;
    }
    while (i < data.size()) {
        buffer_[buffer_len_++] = data[i++];
    }
}

void Sha256::update(std::string_view text) {
    update(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

std::array<uint8_t, 32> Sha256::digest() {
    uint8_t last[128];
    std::size_t n = buffer_len_;
    std::memcpy(last, buffer_, n);
    last[n++] = 0x80;
    while ((n % 64) != 56) {
        last[n++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        last[n++] = static_cast<uint8_t>((bit_len_ >> (8 * i)) & 0xFFu);
    }
    transform(last);
    if (n > 64) {
        transform(last + 64);
    }

    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[static_cast<std::size_t>(i * 4)] = static_cast<uint8_t>((state_[i] >> 24) & 0xFFu);
        out[static_cast<std::size_t>(i * 4 + 1)] = static_cast<uint8_t>((state_[i] >> 16) & 0xFFu);
        out[static_cast<std::size_t>(i * 4 + 2)] = static_cast<uint8_t>((state_[i] >> 8) & 0xFFu);
        out[static_cast<std::size_t>(i * 4 + 3)] = static_cast<uint8_t>(state_[i] & 0xFFu);
    }
    return out;
}

std::string Sha256::hex_digest() {
    const auto d = digest();
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (uint8_t b : d) {
        os << std::setw(2) << static_cast<int>(b);
    }
    return os.str();
}

std::string sha256_hex(std::span<const uint8_t> data) {
    Sha256 h;
    h.update(data);
    return h.hex_digest();
}

std::string sha256_hex_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return {};
    }
    Sha256 h;
    uint8_t buf[1 << 16];
    while (true) {
        const std::size_t n = std::fread(buf, 1, sizeof(buf), f);
        if (n == 0) {
            break;
        }
        h.update({buf, n});
    }
    std::fclose(f);
    return h.hex_digest();
}

}  // namespace nn
