#include "profiler.h"

#include <core/logger/logger.h>
#include <tracy/Tracy.hpp>

namespace pc::profiling {

#ifdef TRACY_ENABLE

void start_profiler() {
  pc::logger()->trace("Starting Tracy profiling session");
  // TracySetProgramName("pointcaster");
  tracy::StartupProfiler();
  pc::logger()->info("Started Tracy profiling session");
}

void stop_profiler() {
  pc::logger()->trace("Stopping Tracy profiling session");
  tracy::ShutdownProfiler();
  pc::logger()->info("Stopped Tracy profiling session");
}

#else
void start_profiler() {}
void stop_profiler() {}
#endif

} // namespace pc::profiling