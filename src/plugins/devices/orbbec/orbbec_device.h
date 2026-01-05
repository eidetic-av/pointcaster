#include "../device.h"
#include "orbbec_device_config.h"

#include <Corrade/PluginManager/AbstractManager.h>

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>

namespace pc::devices {

class OrbbecDevice final : public DeviceBase<OrbbecDeviceConfiguration> {
public:
  explicit OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                        Corrade::Containers::StringView plugin)
      : DeviceBase{manager, plugin} {}

  pc::types::PointCloud point_cloud() override { return {{}, {}}; }
};

} // namespace pc::devices