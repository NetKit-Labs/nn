#include "cli/commands.h"
#include "commands/common.h"
#include "nn/json.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#endif

namespace nn {
namespace {

std::string shell_quote(const std::string& s) {
#ifdef _WIN32
    std::string o = "\"";
    for (char c : s) {
        if (c == '"') {
            o += "\\\"";
        } else {
            o += c;
        }
    }
    o += '"';
    return o;
#else
    std::string o = "'";
    for (char c : s) {
        if (c == '\'') {
            o += "'\\''";
        } else {
            o += c;
        }
    }
    o += "'";
    return o;
#endif
}

std::string replace_placeholder(std::string cmd, const std::string& path) {
    const auto pos = cmd.find("{}");
    if (pos != std::string::npos) {
        cmd.replace(pos, 2, shell_quote(path));
    }
    return cmd;
}

int run_test_command(const std::string& cmd) {
    const int rc = std::system(cmd.c_str());
    if (rc == -1) {
        return -1;
    }
#ifdef _WIN32
    return rc;
#else
    if (WIFEXITED(rc)) {
        return WEXITSTATUS(rc);
    }
    return 1;
#endif
}

Result<std::vector<std::string>> load_artifact_sequence(const std::filesystem::path& manifest,
                                                       const std::string& good,
                                                       const std::string& bad) {
    std::vector<std::string> seq;
    if (!manifest.empty()) {
        std::ifstream in(manifest);
        if (!in) {
            return error(ErrorCode::FileNotFound, "cannot open " + manifest.string());
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto j = parse_json(s);
        if (!j) {
            return j.error();
        }
        const Json* arr = nullptr;
        if (j.value().contains("artifacts") && j.value().at("artifacts").is_array()) {
            arr = &j.value().at("artifacts");
        } else if (j.value().contains("sequence") && j.value().at("sequence").is_array()) {
            arr = &j.value().at("sequence");
        }
        if (!arr) {
            return error(ErrorCode::InvalidFormat, "manifest missing artifacts/sequence array");
        }
        const auto root = manifest.parent_path();
        for (const auto& v : arr->as_array()) {
            if (!v.is_string()) {
                continue;
            }
            std::filesystem::path p(v.as_string());
            if (!p.is_absolute()) {
                p = root / p;
            }
            seq.push_back(p.string());
        }
    }
    if (seq.empty()) {
        if (good.empty() || bad.empty()) {
            return error(ErrorCode::MissingArgument, "--good and --bad are required without --manifest");
        }
        seq.push_back(good);
        seq.push_back(bad);
    }
    return seq;
}

int first_bad_index(Printer& p, const std::vector<std::string>& seq, const std::string& test) {
    int lo = 0;                       // last known good
    int hi = static_cast<int>(seq.size()) - 1;  // known bad
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        const std::string cmd = replace_placeholder(test, seq[static_cast<std::size_t>(mid)]);
        p.println("try [" + std::to_string(mid) + "] " + seq[static_cast<std::size_t>(mid)]);
        const int rc = run_test_command(cmd);
        if (rc == -1 || rc == 127) {
            p.errln("nn: test command failed to execute");
            return -1;
        }
        if (rc == 0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return hi;
}

}  // namespace

int cmd_bisect(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "bisect");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto test = flag_value(args.value(), "test");
    if (!test) {
        return p.usage_error("nn: --test is required", "nn bisect --good FILE --bad FILE --test CMD");
    }
    p.println("Safety: nn does not modify the caller's working tree.");

    if (flag_set(args.value(), "git")) {
        auto good = flag_value(args.value(), "good");
        auto bad = flag_value(args.value(), "bad");
        auto model = flag_value(args.value(), "model");
        if (!good || !bad || !model) {
            return p.usage_error("nn: git mode requires --good --bad --model --test",
                                 "nn bisect --git --good COMMIT --bad COMMIT --model PATH --test CMD");
        }
        const auto tmp = std::filesystem::temp_directory_path() / "nn-bisect-worktree";
        std::error_code ec;
        std::filesystem::remove_all(tmp, ec);
        auto git_commits = [&]() -> Result<std::vector<std::string>> {
            const std::string list_cmd = "git rev-list --reverse " + shell_quote(*good) + ".." +
                                         shell_quote(*bad);
            const auto list_file = std::filesystem::temp_directory_path() / "nn-bisect-revs.txt";
            const int rc = run_test_command(list_cmd + " > " + shell_quote(list_file.string()));
            if (rc != 0) {
                return error(ErrorCode::FileError, "git rev-list failed; run from a git repository");
            }
            std::ifstream in(list_file);
            std::vector<std::string> revs;
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty()) {
                    revs.push_back(line);
                }
            }
            if (revs.empty()) {
                revs.push_back(*bad);
            }
            return revs;
        };
        auto revs = git_commits();
        if (!revs) {
            return cmd_fail(p, revs.error());
        }
        int lo = -1;
        int hi = static_cast<int>(revs.value().size()) - 1;
        auto test_rev = [&](int idx) -> int {
            const std::string rev = revs.value()[static_cast<std::size_t>(idx)];
            std::filesystem::remove_all(tmp, ec);
            const std::string add = "git worktree add --detach " + shell_quote(tmp.string()) + " " +
                                    shell_quote(rev);
            if (run_test_command(add) != 0) {
                return 127;
            }
            const auto artifact = tmp / *model;
            const int rc = run_test_command(replace_placeholder(*test, artifact.string()));
            run_test_command("git worktree remove --force " + shell_quote(tmp.string()));
            return rc;
        };
        while (lo + 1 < hi) {
            const int mid = lo + (hi - lo) / 2;
            p.println("try git [" + std::to_string(mid) + "] " +
                      revs.value()[static_cast<std::size_t>(mid)]);
            const int rc = test_rev(mid);
            if (rc == -1 || rc == 127) {
                std::filesystem::remove_all(tmp, ec);
                p.errln("nn: git worktree test failed");
                return kExitValidation;
            }
            if (rc == 0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        std::filesystem::remove_all(tmp, ec);
        p.println("first bad: " + revs.value()[static_cast<std::size_t>(hi)]);
        return kExitOk;
    }

    auto good = flag_value(args.value(), "good").value_or("");
    auto bad = flag_value(args.value(), "bad").value_or("");
    auto manifest = flag_value(args.value(), "manifest").value_or("");
    auto seq = load_artifact_sequence(manifest, good, bad);
    if (!seq) {
        return cmd_fail(p, seq.error());
    }
    if (seq.value().size() < 2) {
        p.errln("nn: need at least two artifacts to bisect");
        return kExitUsage;
    }
    const int idx = first_bad_index(p, seq.value(), *test);
    if (idx < 0) {
        return kExitValidation;
    }
    p.println("first bad: " + seq.value()[static_cast<std::size_t>(idx)]);
    return kExitOk;
}
}  // namespace nn
