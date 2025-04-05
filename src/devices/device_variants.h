#pragma once

#include "device_config.gen.h"
#include "k4a/k4a_config.gen.h"
#include "sequence/ply_sequence_player_config.gen.h"

namespace pc::devices {

struct PlySequencePlayerConfiguration;

using DeviceConfigurationVariant =
    std::variant<DeviceConfiguration, AzureKinectConfiguration,
                 PlySequencePlayerConfiguration>;

template <typename T>
concept ValidDeviceConfig = std::same_as<T, DeviceConfiguration> ||
                            std::same_as<T, AzureKinectConfiguration> ||
                            std::same_as<T, PlySequencePlayerConfiguration>;

} // namespace pc::devices