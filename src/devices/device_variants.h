#pragma once

#include "device_config.gen.h"
#include "sequence/ply_sequence_player_config.gen.h"

namespace pc::devices {

struct PlySequencePlayerConfiguration;

using DeviceConfigurationVariant =
    std::variant<DeviceConfiguration, PlySequencePlayerConfiguration>;

template <typename T>
concept ValidDeviceConfig = std::same_as<T, DeviceConfiguration> ||
                            std::same_as<T, PlySequencePlayerConfiguration>;

} // namespace pc::devices