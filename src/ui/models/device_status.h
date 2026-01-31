#pragma once

#include <QObject>
#include <plugins/devices/device_status.h>

namespace pc::devices::ui {

Q_NAMESPACE

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

} // namespace pc::devices::ui

Q_DECLARE_METATYPE(pc::devices::ui::WorkspaceDeviceStatus)