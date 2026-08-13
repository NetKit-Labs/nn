#include "nn/target.h"

#include "nn/analysis.h"
#include "nn/json.h"

#include <algorithm>
#include <fstream>

namespace nn {
namespace {

Target make_target(std::string name, std::string cpu, uint64_t ram, uint64_t flash, int cores,
                   bool fpu, const char* simd = "") {
    Target t;
    t.name = std::move(name);
    t.cpu = std::move(cpu);
    t.ram_bytes = ram;
    t.storage_bytes = flash;
    t.cores = cores;
    t.native_types = {DataType::Int8, DataType::Int16, DataType::Int32};
    if (fpu) {
        t.native_types.push_back(DataType::Float32);
        t.capabilities.fpu = true;
        t.capabilities.fpu_name = "FPv4-SP";
    }
    t.capabilities.simd = simd[0] != '\0';
    t.capabilities.simd_name = simd;
    t.performance.has_estimate = false;
    return t;
}

}  // namespace

std::vector<Target> builtin_targets() {
    return {
        make_target("cortex-m0+", "Cortex-M0+", 16ull << 10, 128ull << 10, 1, false),
        make_target("cortex-m3", "Cortex-M3", 64ull << 10, 256ull << 10, 1, false),
        make_target("cortex-m4", "Cortex-M4", 128ull << 10, 512ull << 10, 1, false),
        make_target("cortex-m4f", "Cortex-M4F", 256ull << 10, 1ull << 20, 1, true, "DSP"),
        make_target("cortex-m7", "Cortex-M7", 512ull << 10, 2ull << 20, 1, true, "DSP"),
        make_target("cortex-m33", "Cortex-M33", 256ull << 10, 1ull << 20, 1, true),
        make_target("cortex-m55", "Cortex-M55", 512ull << 10, 2ull << 20, 1, true, "Helium"),
        make_target("cortex-m85", "Cortex-M85", 1ull << 20, 4ull << 20, 1, true, "Helium"),
        make_target("cortex-a53", "Cortex-A53", 1ull << 30, 8ull << 30, 4, true, "NEON"),
        make_target("cortex-a55", "Cortex-A55", 2ull << 30, 16ull << 30, 4, true, "NEON"),
        make_target("cortex-a72", "Cortex-A72", 4ull << 30, 32ull << 30, 4, true, "NEON"),
        make_target("cortex-a76", "Cortex-A76", 8ull << 30, 64ull << 30, 4, true, "NEON"),
        make_target("riscv-mcu", "generic RISC-V MCU", 64ull << 10, 256ull << 10, 1, false),
        make_target("riscv-linux", "generic RISC-V Linux", 1ull << 30, 8ull << 30, 4, true),
        make_target("esp32", "ESP32", 520ull << 10, 4ull << 20, 2, true),
        make_target("esp32-s3", "ESP32-S3", 512ull << 10, 8ull << 20, 2, true, "vector"),
        make_target("apple-silicon", "Apple Silicon generic", 8ull << 30, 256ull << 30, 8, true,
                    "NEON"),
        make_target("x86-64", "x86-64 generic", 8ull << 30, 256ull << 30, 8, true, "AVX"),
    };
}

const Target* find_builtin_target(std::string_view name) {
    static const auto k = builtin_targets();
    for (const auto& t : k) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

Result<Target> load_target_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return error(ErrorCode::FileNotFound, "cannot open target file: " + path.string());
    }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto j = parse_json(s);
    if (!j) {
        return j.error();
    }
    Target t;
    t.name = j.value().at("name").as_string();
    t.cpu = j.value().at("cpu").as_string();
    t.performance.clock_hz = static_cast<uint64_t>(j.value().at("clock_hz").as_number());
    t.ram_bytes = static_cast<uint64_t>(j.value().at("ram_bytes").as_number());
    t.storage_bytes = static_cast<uint64_t>(j.value().contains("flash_bytes")
                                                ? j.value().at("flash_bytes").as_number()
                                                : j.value().at("storage_bytes").as_number());
    t.capabilities.simd_name = j.value().at("simd").as_string();
    t.capabilities.fpu_name = j.value().at("fpu").as_string();
    t.capabilities.simd = !t.capabilities.simd_name.empty();
    t.capabilities.fpu = !t.capabilities.fpu_name.empty();
    t.native_types = {DataType::Int8, DataType::Int32};
    if (t.capabilities.fpu) {
        t.native_types.push_back(DataType::Float32);
    }
    return t;
}

Result<Accelerator> load_accelerator_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return error(ErrorCode::FileNotFound, "cannot open accelerator file: " + path.string());
    }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto j = parse_json(s);
    if (!j) {
        return j.error();
    }
    Accelerator a;
    a.name = j.value().at("name").as_string();
    if (j.value().contains("supported_ops") && j.value().at("supported_ops").is_array()) {
        for (const auto& o : j.value().at("supported_ops").as_array()) {
            if (o.is_string()) {
                a.supported_ops.push_back(o.as_string());
            }
        }
    }
    if (j.value().contains("data_types") && j.value().at("data_types").is_array()) {
        for (const auto& o : j.value().at("data_types").as_array()) {
            if (o.is_string()) {
                if (auto d = datatype_from_name(o.as_string())) {
                    a.data_types.push_back(*d);
                }
            }
        }
    }
    a.macs_per_cycle = static_cast<uint64_t>(j.value().at("macs_per_cycle").as_number());
    a.clock_hz = static_cast<uint64_t>(j.value().at("clock_hz").as_number());
    a.local_memory_bytes = static_cast<uint64_t>(j.value().at("local_memory_bytes").as_number());
    a.alignment = static_cast<uint64_t>(j.value().at("alignment").as_number());
    if (a.alignment == 0) {
        a.alignment = 1;
    }
    return a;
}

}  // namespace nn
