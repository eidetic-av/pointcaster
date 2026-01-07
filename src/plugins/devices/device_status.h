#pragma once

namespace pc::devices {

enum class DeviceStatus {
  Unloaded, // device plugin has not loaded
  Loaded,   // device plugin has loaded and device is idle
  Active,   // device is streaming
  Missing   // device plugin has loaded but device can't be found
};

} // namespace pc::devices