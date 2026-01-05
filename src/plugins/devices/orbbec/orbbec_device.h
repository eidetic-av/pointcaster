#include "../device.h"
#include "orbbec_device_config.h"

#include <Corrade/PluginManager/AbstractManager.h>

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>

#include <print>

namespace pc::devices {

class OrbbecDevice final : public DeviceBase<OrbbecDeviceConfiguration> {
public:
  explicit OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                        Corrade::Containers::StringView plugin)
      : DeviceBase{manager, plugin} {}

  ~OrbbecDevice() {
    std::println("Unloading orbbec");
  }

  pc::types::PointCloud point_cloud() override { return {{}, {}}; }
};

} // namespace pc::devices