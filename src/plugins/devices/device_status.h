#pragma once

namespace pc::devices {

enum class DeviceStatus {
  Unloaded, // device plugin has not loaded
  Loading,   // device plugin is currently attempting to load plugin or driver
  Loaded,   // device plugin has loaded and device is idle
  Active,   // device is streaming
  Missing   // device plugin has loaded but device can't be found
};

} // namespace pc::devices