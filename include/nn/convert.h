#ifndef NN_CONVERT_H
#define NN_CONVERT_H

#include "nn/result.h"

#include <string>
#include <vector>

namespace nn {

struct ConversionRoute {
    std::string from;
    std::string to;
    bool available = false;
    std::string notes;
};

std::vector<ConversionRoute> conversion_routes();

}  // namespace nn

#endif
