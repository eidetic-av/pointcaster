#pragma once
#include "plugins/devices/device_plugin.h"
#include "plugins/devices/device_variants.h"
#include <Corrade/Containers/Pointer.h>

namespace pc {

// this is the config that gets de/serialized
struct WorkspaceConfiguration {
  std::vector<devices::DeviceConfigurationVariant> devices;
};

// this is the state of the application
struct Workspace {
  std::vector<Corrade::Containers::Pointer<devices::DevicePlugin>> devices;
};

} // namespace pc