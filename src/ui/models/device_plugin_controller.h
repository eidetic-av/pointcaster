#pragma once

#include "device_status.h"
#include <QObject>
#include <QString>
#include <plugins/devices/device_plugin.h>

namespace pc::ui {

class DevicePluginController : public QObject {
  Q_OBJECT

  Q_PROPERTY(QString name READ name CONSTANT)
  Q_PROPERTY(Status status READ status NOTIFY statusChanged)

public:
  enum class Status { Unloaded, Loaded, Active, Missing };
  Q_ENUM(Status)

  explicit DevicePluginController(pc::devices::DevicePlugin *plugin,
                                  QString deviceName, QObject *parent = nullptr)
      : QObject(parent), _plugin(plugin), _name(std::move(deviceName)) {
    if (_plugin) {
      // initialise from current core status
      setStatusFromCore(_plugin->status());
    } else {
      _status = Status::Missing;
    }
  }

  QString name() const { return _name; }
  Status status() const { return _status; }

  // called from core / WorkspaceModel when DeviceStatus changes
  void setStatusFromCore(pc::devices::DeviceStatus coreStatus) {
    Status newStatus = Status::Unloaded;
    switch (coreStatus) {
    case pc::devices::DeviceStatus::Unloaded:
      newStatus = Status::Unloaded;
      break;
    case pc::devices::DeviceStatus::Loaded:
      newStatus = Status::Loaded;
      break;
    case pc::devices::DeviceStatus::Active:
      newStatus = Status::Active;
      break;
    case pc::devices::DeviceStatus::Missing:
      newStatus = Status::Missing;
      break;
    }
    updateStatus(newStatus);
  }

  Q_INVOKABLE void start() {
    if (!_plugin) return;
    _plugin->start();
  }

  Q_INVOKABLE void stop() {
    if (!_plugin) return;
    _plugin->stop();
  }

  Q_INVOKABLE void restart() {
    if (!_plugin) return;
    _plugin->restart();
  }

signals:
  void statusChanged();

private:
  void updateStatus(Status s) {
    if (_status != s) {
      _status = s;
      emit statusChanged();
    }
  }

  pc::devices::DevicePlugin *_plugin = nullptr; // non-owning
  QString _name;
  Status _status = Status::Unloaded;
};

} // namespace pc::ui
