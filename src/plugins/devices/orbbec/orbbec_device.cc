#include "orbbec_device.h"
#include <Corrade/PluginManager/AbstractManager.h>


namespace pc::devices {

class OrbbecDevice final : public AbstractOrbbecDevice {
public:
  explicit OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                          Corrade::Containers::StringView plugin)
      : AbstractOrbbecDevice{manager, plugin} {}
};

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(OrbbecDevice, pc::devices::OrbbecDevice,
                        "net.pointcaster.OrbbecDevice/1.0")