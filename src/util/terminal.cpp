#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <io.h>
#define NN_ISATTY _isatty
#define NN_FILENO _fileno
#else
#include <unistd.h>
#define NN_ISATTY isatty
#define NN_FILENO fileno
#endif

namespace nn {

bool stdout_is_tty() { return NN_ISATTY(NN_FILENO(stdout)) != 0; }
bool stderr_is_tty() { return NN_ISATTY(NN_FILENO(stderr)) != 0; }

bool want_color(bool always, bool never, bool auto_detect) {
    if (never) {
        return false;
    }
    if (always) {
        return true;
    }
    if (!auto_detect) {
        return false;
    }
    const char* no_color = std::getenv("NO_COLOR");
    if (no_color && no_color[0] != '\0') {
        return false;
    }
    return stdout_is_tty();
}

}  // namespace nn
