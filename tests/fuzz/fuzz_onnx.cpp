#include "formats/onnx/proto.h"
#include "nn/format.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    nn::LoadOptions opt;
    opt.max_allocation_bytes = 1 << 20;
    (void)nn::onnx::parse_model_proto({data, size}, opt, {});
    return 0;
}
