#ifndef NN_COMMANDS_H
#define NN_COMMANDS_H

#include "nn/cli.h"

namespace nn {

int cmd_inspect(const GlobalOptions& g);
int cmd_ops(const GlobalOptions& g);
int cmd_tensors(const GlobalOptions& g);
int cmd_io(const GlobalOptions& g);
int cmd_metadata(const GlobalOptions& g);
int cmd_hash(const GlobalOptions& g);
int cmd_diff(const GlobalOptions& g);
int cmd_graph(const GlobalOptions& g);
int cmd_memory(const GlobalOptions& g);
int cmd_compute(const GlobalOptions& g);
int cmd_quant(const GlobalOptions& g);
int cmd_sparsity(const GlobalOptions& g);
int cmd_lint(const GlobalOptions& g);
int cmd_run(const GlobalOptions& g);
int cmd_compare(const GlobalOptions& g);
int cmd_test(const GlobalOptions& g);
int cmd_validate(const GlobalOptions& g);
int cmd_regression(const GlobalOptions& g);
int cmd_benchmark(const GlobalOptions& g);
int cmd_profile(const GlobalOptions& g);
int cmd_compat(const GlobalOptions& g);
int cmd_target(const GlobalOptions& g);
int cmd_partition(const GlobalOptions& g);
int cmd_convert(const GlobalOptions& g);
int cmd_optimize(const GlobalOptions& g);
int cmd_extract(const GlobalOptions& g);
int cmd_canonicalize(const GlobalOptions& g);
int cmd_bisect(const GlobalOptions& g);
int cmd_doctor(const GlobalOptions& g);
int cmd_formats(const GlobalOptions& g);
int cmd_backends(const GlobalOptions& g);
int cmd_targets(const GlobalOptions& g);
int cmd_config(const GlobalOptions& g);
int cmd_version(const GlobalOptions& g);
int cmd_help(const GlobalOptions& g);

}  // namespace nn

#endif
