#pragma once

#include <pointcaster/core.h>

namespace pc::profiling {

POINTCASTER_CORE_EXPORT void start_profiler();

POINTCASTER_CORE_EXPORT bool profiler_running() noexcept;

} // namespace pc::profiling