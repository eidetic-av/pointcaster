#include "profiler.h"

#include <core/logger/logger.h>
#include <tracy/Tracy.hpp>

namespace pc::profiling {

#ifdef TRACY_ENABLE

void start_profiler() {
  pc::logger()->info("Starting Tracy profiling session");
  // TracySetProgramName("pointcaster");
  tracy::StartupProfiler();
  pc::logger()->info("Started");
}

void stop_profiler() {
  pc::logger()->info("Stopping Tracy profiling session");
  tracy::ShutdownProfiler();
  pc::logger()->info("Stopped");
}

#else
void start_profiler() {}
void stop_profiler() {}
#endif

} // namespace pc::profiling