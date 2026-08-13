#ifndef NN_REF_OPS_H
#define NN_REF_OPS_H

#include "nn/runtime.h"

#include <map>

namespace nn {

bool reference_op_supported(const Node& node);
Result<RuntimeTensor> reference_exec_node(const Graph& graph, const Node& node,
                                          const std::map<TensorId, RuntimeTensor>& vals);

}  // namespace nn

#endif
