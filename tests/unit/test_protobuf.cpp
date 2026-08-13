#include "test.h"
#include "util/protobuf.h"

TEST(protobuf_roundtrip) {
    nn::ProtoWriter w;
    w.varint_field(1, 8);
    w.string_field(2, "nn-test");
    auto bytes = w.take();
    nn::ProtoReader r(bytes);
    auto f1 = r.next();
    CHECK(f1);
    CHECK(f1.value().number == 1);
    CHECK(f1.value().varint == 8);
    auto f2 = r.next();
    CHECK(f2);
    CHECK(f2.value().number == 2);
    CHECK(f2.value().bytes.size() == 7);
}
