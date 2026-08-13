#ifndef NN_QUANTIZATION_H
#define NN_QUANTIZATION_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nn {

struct QuantizationInfo {
    bool quantized = false;
    std::vector<double> scales;
    std::vector<int64_t> zero_points;
    std::optional<int> axis;
    int bits = 0;
    bool per_channel = false;
    std::string scheme;  // "affine", "symmetric", "pow2", ...

    bool valid() const;
    std::string describe() const;
};

}  // namespace nn

#endif
