#include "nn/quantization.h"

#include <sstream>

namespace nn {

bool QuantizationInfo::valid() const {
    if (!quantized) {
        return true;
    }
    if (scales.empty()) {
        return false;
    }
    if (!zero_points.empty() && zero_points.size() != scales.size()) {
        return false;
    }
    if (per_channel && !axis) {
        return false;
    }
    for (double s : scales) {
        if (!(s > 0.0)) {
            return false;
        }
    }
    return true;
}

std::string QuantizationInfo::describe() const {
    if (!quantized) {
        return "none";
    }
    std::ostringstream os;
    if (bits > 0) {
        os << "int" << bits << ' ';
    }
    os << (per_channel ? "per-channel" : "per-tensor");
    if (!scales.empty()) {
        os << " scale=" << scales.front();
        if (scales.size() > 1) {
            os << " [" << scales.size() << "]";
        }
    }
    if (!zero_points.empty()) {
        os << " zp=" << zero_points.front();
    }
    return os.str();
}

}  // namespace nn
