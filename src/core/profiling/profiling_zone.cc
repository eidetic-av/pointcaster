#include "profiling_zone.h"

#ifdef TRACY_ENABLE
// - we need to wrap tracy so that theres a single instance in a shared
// library that all modules of code have access to
// - and Tracy's C API is used to avoid template issues in .cu files... nvcc
// throws a spat otherwise
#include <cstring>
#include <tracy/TracyC.h>

#ifndef TRACY_CALLSTACK
#define TRACY_CALLSTACK 0
#endif

namespace pc::profiling {

struct InternalZone {
  TracyCZoneCtx ctx;
};

ProfilingZone::ProfilingZone(std::string_view name) {
  // Allocate our internal zone object.
  auto *iz = new InternalZone;
  // Allocate a source location using the name.
  // ___tracy_alloc_srcloc_name takes: line, file, file length, function,
  // function length, name, name length, color.
  uint64_t srcloc = ___tracy_alloc_srcloc_name(
      __LINE__, __FILE__, std::strlen(__FILE__), __func__,
      std::strlen(__func__), name.data(), name.size(), 0);
  // Begin the zone with callstack depth TRACY_CALLSTACK.
  iz->ctx =
      ___tracy_emit_zone_begin_alloc_callstack(srcloc, TRACY_CALLSTACK, 1);
  zone_data = iz;
}

ProfilingZone::~ProfilingZone() {
  if (zone_data) {
    auto *iz = static_cast<InternalZone *>(zone_data);
    ___tracy_emit_zone_end(iz->ctx);
    delete iz;
  }
}

void ProfilingZone::text(std::string_view text) {
  if (zone_data) {
    auto *iz = static_cast<InternalZone *>(zone_data);
    ___tracy_emit_zone_text(iz->ctx, text.data(),
                            static_cast<unsigned int>(text.size()));
  }
}

} // namespace pc::profiling

#else // TRACY_ENABLE not defined

namespace pc::profiling {

ProfilingZone::ProfilingZone(std::string_view) : zone_data(nullptr) {}
ProfilingZone::~ProfilingZone() {}
void ProfilingZone::text(std::string_view) {}

} // namespace pc::profiling

#endif