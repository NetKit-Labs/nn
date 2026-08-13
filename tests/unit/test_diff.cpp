#include "nn/diff.h"
#include "test.h"

TEST(diff_identical) {
    nn::ModelIR a;
    nn::Graph g;
    g.name = "g";
    a.graphs.push_back(g);
    nn::ModelIR b = a;
    auto d = nn::diff_models(a, b);
    CHECK(d);
    CHECK(d.value().identical);
}

TEST(diff_nodes) {
    nn::ModelIR a;
    nn::Graph ga;
    nn::Node n;
    n.id = 0;
    n.name = "x";
    n.op_type = "Add";
    ga.nodes.push_back(n);
    a.graphs.push_back(ga);
    nn::ModelIR b;
    b.graphs.emplace_back();
    auto d = nn::diff_models(a, b);
    CHECK(d);
    CHECK(!d.value().identical);
}
