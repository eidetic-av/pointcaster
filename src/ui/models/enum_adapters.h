#pragma once

#include <QObject>
#include <plugins/devices/device_status.h>

namespace pc::ui {

Q_NAMESPACE

// status of a device as shown in the workspace device list
enum class WorkspaceDeviceStatus {
  Unloaded = static_cast<int>(devices::DeviceStatus::Unloaded),
  Loaded = static_cast<int>(devices::DeviceStatus::Loaded),
  Active = static_cast<int>(devices::DeviceStatus::Active),
  Missing = static_cast<int>(devices::DeviceStatus::Missing),
};
Q_ENUM_NS(WorkspaceDeviceStatus)

inline WorkspaceDeviceStatus toQt(devices::DeviceStatus s) {
  return static_cast<WorkspaceDeviceStatus>(static_cast<int>(s));
}

inline pc::devices::DeviceStatus fromQt(WorkspaceDeviceStatus s) {
  return static_cast<devices::DeviceStatus>(static_cast<int>(s));
}

// status of a single broadcast channel, shown as a coloured dot in
// StreamChannelList. Disabled: channel is turned off. Live: channel is
// enabled and broadcasting. Connected: enabled and the publisher knows it
// has at least one listener (not yet implemented upstream).
enum class StreamChannelStatus {
  Disabled = 0,
  Live = 1,
  Connected = 2,
};
Q_ENUM_NS(StreamChannelStatus)

} // namespace pc::ui

Q_DECLARE_METATYPE(pc::ui::WorkspaceDeviceStatus)
Q_DECLARE_METATYPE(pc::ui::StreamChannelStatus)
