#pragma once

#include "orbbec/orbbec_device_config.gen.h"
#include <variant>

namespace pc::devices {

using DeviceConfigurationVariant = std::variant<OrbbecDeviceConfiguration>;

}