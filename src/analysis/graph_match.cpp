#include "nn/analysis.h"

#include <algorithm>
#include <map>

namespace nn {

std::vector<MatchedNode> match_graphs(const Graph& a, const Graph& b) {
    std::vector<MatchedNode> out;
    std::map<std::string, const Node*> by_name;
    for (const auto& n : b.nodes) {
        if (!n.name.empty()) {
            by_name[n.name] = &n;
        }
    }
    std::vector<bool> used(b.nodes.size(), false);
    for (const auto& n : a.nodes) {
        MatchedNode m;
        m.a = &n;
        if (!n.name.empty()) {
            auto it = by_name.find(n.name);
            if (it != by_name.end()) {
                m.b = it->second;
                const auto idx = static_cast<std::size_t>(m.b - b.nodes.data());
                if (idx < used.size()) {
                    used[idx] = true;
                }
            }
        }
        if (!m.b) {
            for (std::size_t i = 0; i < b.nodes.size(); ++i) {
                if (used[i]) {
                    continue;
                }
                if (b.nodes[i].op_type == n.op_type) {
                    m.b = &b.nodes[i];
                    used[i] = true;
                    break;
                }
            }
        }
        out.push_back(m);
    }
    for (std::size_t i = 0; i < b.nodes.size(); ++i) {
        if (!used[i]) {
            MatchedNode m;
            m.b = &b.nodes[i];
            out.push_back(m);
        }
    }
    return out;
}

}  // namespace nn
