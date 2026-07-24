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

#else
void start_profiler() {}
#endif

} // namespace pc::profiling