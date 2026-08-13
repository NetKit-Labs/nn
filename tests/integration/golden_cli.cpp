#include "formats/onnx/writer.h"
#include "nn/json.h"
#include "test.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifdef _WIN32
#define NN_POPEN _popen
#define NN_PCLOSE _pclose
#else
#include <sys/wait.h>
#define NN_POPEN popen
#define NN_PCLOSE pclose
#endif

namespace {

#ifndef NN_TEST_NN_EXE
#define NN_TEST_NN_EXE ""
#endif

std::string quote(const std::string& s) { return "\"" + s + "\""; }

struct CmdOut {
    int rc = -1;
    std::string out;
};

CmdOut run_nn(const std::string& args) {
    CmdOut r;
    const std::string exe = NN_TEST_NN_EXE;
    if (exe.empty()) {
        return r;
    }
    const std::string cmd = quote(exe) + " " + args;
    FILE* p = NN_POPEN(cmd.c_str(), "r");
    if (!p) {
        return r;
    }
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), p) != nullptr) {
        r.out.append(buf);
    }
    r.rc = NN_PCLOSE(p);
#if defined(WIFEXITED)
    if (WIFEXITED(r.rc)) {
        r.rc = WEXITSTATUS(r.rc);
    }
#elif defined(_WIN32)
    // _pclose returns the process exit code
#endif
    return r;
}

}  // namespace

TEST(golden_nn_exe_present) {
    CHECK(!std::string(NN_TEST_NN_EXE).empty());
    CHECK(std::filesystem::exists(NN_TEST_NN_EXE));
}

TEST(golden_formats_json) {
    auto r = run_nn("--json formats");
    CHECK(r.rc == 0);
    auto j = nn::parse_json(r.out);
    CHECK(j);
    CHECK(j.value().contains("schema_version"));
    CHECK(j.value().at("schema_version").as_number() == 1);
    CHECK(j.value().contains("formats"));
    CHECK(j.value().at("formats").is_array());
    bool onnx = false;
    for (const auto& f : j.value().at("formats").as_array()) {
        if (f.contains("name") && f.at("name").is_string() && f.at("name").as_string() == "onnx") {
            onnx = true;
        }
    }
    CHECK(onnx);
}

TEST(golden_version) {
    auto r = run_nn("version");
    CHECK(r.rc == 0);
    CHECK(r.out.find("nn ") == 0);
}

TEST(golden_help_inspect) {
    auto r = run_nn("help inspect");
    CHECK(r.rc == 0);
    CHECK(r.out.find("inspect") != std::string::npos);
}

TEST(golden_hash_json_schema) {
    const auto path = std::filesystem::temp_directory_path() / "nn_golden.onnx";
    CHECK(nn::write_simple_onnx(path, {}));
    auto r = run_nn("--json hash " + quote(path.string()));
    CHECK(r.rc == 0);
    auto j = nn::parse_json(r.out);
    CHECK(j);
    CHECK(j.value().contains("schema_version"));
    CHECK(j.value().at("schema_version").as_number() == 1);
    CHECK(j.value().contains("artifact"));
    CHECK(j.value().at("artifact").is_string());
    CHECK(j.value().at("artifact").as_string().size() == 64);
}

TEST(golden_porcelain_hash) {
    const auto path = std::filesystem::temp_directory_path() / "nn_golden_p.onnx";
    CHECK(nn::write_simple_onnx(path, {}));
    auto r = run_nn("--porcelain hash " + quote(path.string()));
    CHECK(r.rc == 0);
    CHECK(r.out.find("artifact:") != std::string::npos);
    CHECK(r.out.find('\t') != std::string::npos);
}
