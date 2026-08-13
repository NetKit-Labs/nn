#ifndef NN_TERMINAL_H
#define NN_TERMINAL_H

namespace nn {

bool stdout_is_tty();
bool stderr_is_tty();
bool want_color(bool always, bool never, bool auto_detect);

}  // namespace nn

#endif
