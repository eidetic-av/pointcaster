#include "profiler.h"

#include <atomic>
#include <core/logger/logger.h>
#include <tracy/Tracy.hpp>

namespace pc::profiling {

#ifdef TRACY_ENABLE

namespace {
std::atomic<bool> profiler_started{false};
}

void start_profiler() {
  pc::logger()->trace("Starting Tracy profiling session");
  // TracySetProgramName("pointcaster");
  tracy::StartupProfiler();
  profiler_started.store(true, std::memory_order_release);
  pc::logger()->info("Started Tracy profiling session");
}

bool profiler_running() noexcept {
  return profiler_started.load(std::memory_order_acquire);
}

#else
void start_profiler() {}
bool profiler_running() noexcept {
  return false;
}
#endif

} // namespace pc::profiling