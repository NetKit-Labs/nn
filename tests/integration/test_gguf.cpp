#include "nn/format.h"
#include "test.h"

#include <fstream>

TEST(gguf_magic) {
    const auto path = std::filesystem::temp_directory_path() / "nn_tiny.gguf";
    std::ofstream out(path, std::ios::binary);
    const char mag[] = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
    out.write(mag, 8);
    uint64_t zero = 0;
    out.write(reinterpret_cast<const char*>(&zero), 8);  // tensor_count
    out.write(reinterpret_cast<const char*>(&zero), 8);  // kv_count
    out.close();
    auto m = nn::load_model(path);
    CHECK(m);
    CHECK(m.value().source_format == "gguf");
}
