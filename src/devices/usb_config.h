#pragma once

#include "../serialization.h"

#include <atomic>
#include <map>
#include <string>
#include <vector>

namespace pc::devices {

struct UsbConfiguration {
  bool open_on_launch = true;
  bool open_on_hotplug = true;
};

} // namespace pc::devices
