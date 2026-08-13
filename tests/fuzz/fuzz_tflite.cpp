#include "nn/format.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("nn_fuzz_tflite_" + std::to_string(reinterpret_cast<uintptr_t>(data)));
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    nn::LoadOptions opt;
    opt.max_allocation_bytes = 1 << 20;
    (void)nn::load_model(path, opt);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}
