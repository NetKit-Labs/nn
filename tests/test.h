#ifndef NN_TEST_H
#define NN_TEST_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace nn_test {

struct Case {
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& cases() {
    static std::vector<Case> c;
    return c;
}

struct Reg {
    Reg(const char* name, void (*fn)()) { cases().push_back({name, fn}); }
};

inline int failures = 0;

inline void check(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::cerr << "FAIL " << file << ":" << line << "  " << expr << "\n";
        ++failures;
    }
}

inline int run() {
    for (const auto& c : cases()) {
        const int before = failures;
        c.fn();
        if (failures == before) {
            std::cout << "ok   " << c.name << "\n";
        } else {
            std::cout << "FAIL " << c.name << "\n";
        }
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << cases().size() << " tests passed\n";
    return 0;
}

}  // namespace nn_test

#define TEST(name)                                      \
    static void test_##name();                          \
    static nn_test::Reg reg_##name(#name, test_##name); \
    static void test_##name()

#define CHECK(expr) nn_test::check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

#endif
