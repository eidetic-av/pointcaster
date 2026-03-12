#pragma once
#include <pointcaster/core.h>
#include <string_view>

namespace pc::profiling {

class POINTCASTER_CORE_EXPORT ProfilingZone {
public:
  explicit ProfilingZone(std::string_view name);
  ~ProfilingZone();

  ProfilingZone(const ProfilingZone &) = delete;
  ProfilingZone &operator=(const ProfilingZone &) = delete;
  ProfilingZone(ProfilingZone &&) = delete;
  ProfilingZone &operator=(ProfilingZone &&) = delete;

  void text(std::string_view text);

private:
  void *zone_data;
};

} // namespace pc::profiling