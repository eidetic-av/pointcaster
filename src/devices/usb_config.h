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

  DERIVE_SERDE(UsbConfiguration,
               (&Self::open_on_launch, "open_on_launch")
	       (&Self::open_on_hotplug, "open_on_hotplug"))
};

} // namespace pc::devices
