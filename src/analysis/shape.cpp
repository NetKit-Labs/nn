#include "nn/analysis.h"

namespace nn {

// Shape inference beyond declared tensor shapes is format-specific. This
// module currently reports declared shapes only.
Status infer_shapes(ModelIR& model) {
    (void)model;
    return Status::ok();
}

}  // namespace nn
