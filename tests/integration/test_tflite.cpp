#include "nn/format.h"
#include "test.h"

#include <fstream>

TEST(tflite_probe_rejects_random) {
    const auto path = std::filesystem::temp_directory_path() / "nn_not.tflite";
    std::ofstream out(path, std::ios::binary);
    out << "not a tflite file";
    out.close();
    auto files = nn::FileSet::from_path(path);
    CHECK(files);
    auto* r = nn::default_format_registry().find_by_name("tflite");
    CHECK(r);
    CHECK(!r->probe(files.value()));
}

#ifdef NN_TEST_TFLITE_ADD
TEST(tflite_add_loads) {
    auto m = nn::load_model(NN_TEST_TFLITE_ADD);
    CHECK(m);
    CHECK(m.value().source_format == "tflite");
    CHECK(!m.value().graphs.empty());
    CHECK(!m.value().graphs.front().nodes.empty());
}

#if defined(NN_HAS_LITERT)
TEST(tflite_formats_execute_yes) {
    auto* r = nn::default_format_registry().find_by_name("tflite");
    CHECK(r);
    CHECK(r->capabilities().execute);
}
#endif
#endif

