#include "nn/analysis.h"

#include <algorithm>
#include <vector>

namespace nn {
namespace {

std::vector<MemoryPlanEntry> plan_activations(const std::vector<TensorLifetime>& lts,
                                              uint64_t alignment) {
    struct Interval {
        const TensorLifetime* lt;
    };
    std::vector<Interval> live;
    for (const auto& lt : lts) {
        if (lt.persistent || lt.bytes == 0) {
            continue;
        }
        live.push_back({&lt});
    }
    std::sort(live.begin(), live.end(), [](const Interval& a, const Interval& b) {
        if (a.lt->bytes != b.lt->bytes) {
            return a.lt->bytes > b.lt->bytes;
        }
        return a.lt->birth < b.lt->birth;
    });

    struct Slot {
        uint64_t offset;
        uint64_t size;
        int death;
    };
    std::vector<Slot> occupied;
    std::vector<MemoryPlanEntry> plan;
    uint64_t arena = 0;

    auto align = [&](uint64_t v) {
        if (alignment <= 1) {
            return v;
        }
        return (v + alignment - 1) / alignment * alignment;
    };

    for (const auto& iv : live) {
        const auto& lt = *iv.lt;
        uint64_t chosen = static_cast<uint64_t>(-1);
        uint64_t chosen_size = align(lt.bytes);
        // First-fit among gaps of currently overlapping allocations.
        std::vector<Slot> overlap;
        for (const auto& s : occupied) {
            const bool overlap_time = !(lt.birth > s.death || lt.death < 0 /*dummy*/);
            // lifetime overlap: [birth, death] intersects
            const bool tover = !(lt.death < s.death && lt.death < 0);
            (void)overlap_time;
            (void)tover;
        }
        // Rebuild overlapping slots
        overlap.clear();
        for (const auto& s : occupied) {
            if (lt.birth <= s.death && lt.death >= 0 /* compare */) {
                // slot death is the death of that tensor
            }
        }
        overlap.clear();
        for (std::size_t i = 0; i < plan.size(); ++i) {
            const auto& p = plan[i];
            if (lt.birth <= p.death && lt.death >= p.birth) {
                overlap.push_back({p.offset, p.size, p.death});
            }
        }
        std::sort(overlap.begin(), overlap.end(),
                  [](const Slot& a, const Slot& b) { return a.offset < b.offset; });
        uint64_t cursor = 0;
        for (const auto& s : overlap) {
            if (s.offset >= cursor + chosen_size) {
                chosen = cursor;
                break;
            }
            cursor = std::max(cursor, s.offset + s.size);
            cursor = align(cursor);
        }
        if (chosen == static_cast<uint64_t>(-1)) {
            chosen = cursor;
        }
        MemoryPlanEntry e;
        e.tensor_id = lt.tensor_id;
        e.name = lt.name;
        e.offset = chosen;
        e.size = chosen_size;
        e.birth = lt.birth;
        e.death = lt.death;
        plan.push_back(e);
        arena = std::max(arena, e.offset + e.size);
        occupied.push_back({e.offset, e.size, e.death});
    }
    (void)arena;
    return plan;
}

}  // namespace

MemoryReport analyze_memory(const ModelIR& model, const MemoryOptions& options) {
    MemoryReport report;
    report.file_size = model.file_size;
    const Graph* g = primary_graph(model);
    if (!g) {
        return report;
    }
    report.lifetimes = analyze_lifetimes(*g);
    for (const auto& t : g->tensors) {
        uint64_t bytes = t.storage_bytes;
        if (bytes == 0) {
            if (auto b = tensor_storage_bytes(t)) {
                bytes = b.value();
            }
        }
        if (t.constant) {
            report.constant_bytes += bytes;
            report.weight_bytes += bytes;
        }
    }
    for (const auto& lt : report.lifetimes) {
        if (lt.persistent) {
            report.persistent_bytes += lt.bytes;
        }
    }

    // Sweep-line peak of non-persistent tensors.
    struct Event {
        int time;
        int64_t delta;
    };
    std::vector<Event> ev;
    for (const auto& lt : report.lifetimes) {
        if (lt.persistent || lt.bytes == 0) {
            continue;
        }
        ev.push_back({lt.birth, static_cast<int64_t>(lt.bytes)});
        ev.push_back({lt.death + 1, -static_cast<int64_t>(lt.bytes)});
    }
    std::sort(ev.begin(), ev.end(), [](const Event& a, const Event& b) {
        if (a.time != b.time) {
            return a.time < b.time;
        }
        return a.delta < b.delta;  // frees first
    });
    int64_t cur = 0;
    int64_t peak = 0;
    for (const auto& e : ev) {
        cur += e.delta;
        if (cur > peak) {
            peak = cur;
        }
    }
    report.peak_activation_bytes = static_cast<uint64_t>(peak < 0 ? 0 : peak);
    report.peak_known = true;
    report.scratch_bytes = 0;
    report.estimated_ram_bytes = report.peak_activation_bytes + report.scratch_bytes;
    if (options.plan) {
        report.plan = plan_activations(report.lifetimes, options.alignment);
        uint64_t arena = 0;
        for (const auto& p : report.plan) {
            arena = std::max(arena, p.offset + p.size);
        }
        report.scratch_bytes = arena;
        report.estimated_ram_bytes = arena;
    }
    return report;
}

}  // namespace nn
