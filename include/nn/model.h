#ifndef NN_MODEL_H
#define NN_MODEL_H

#include "nn/graph.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nn {

using Metadata = std::map<std::string, std::string>;

struct ModelIR {
    std::string source_format;
    std::string source_format_version;
    std::string producer;
    std::string framework_version;
    std::string domain;
    int64_t model_version = 0;
    std::string doc;
    std::vector<Graph> graphs;
    Metadata metadata;
    std::filesystem::path source_path;
    uint64_t file_size = 0;
    std::string sha256;
    std::vector<std::string> warnings;
};

const Graph* primary_graph(const ModelIR& model);
Graph* primary_graph(ModelIR& model);

}  // namespace nn

#endif
