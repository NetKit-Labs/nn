#ifndef NN_OPTIMIZE_H
#define NN_OPTIMIZE_H

#include "nn/model.h"
#include "nn/result.h"

#include <string>
#include <string_view>
#include <vector>

namespace nn {

struct OptimizeReport {
    std::vector<std::string> changes;
    ModelIR model;
};

OptimizeReport optimize_model(ModelIR model);
Result<ModelIR> extract_subgraph(const ModelIR& model, std::string_view from, std::string_view to);

}  // namespace nn

#endif
