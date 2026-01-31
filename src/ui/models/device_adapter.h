#pragma once

#include "config_adapter.h"
#include "device_status.h"

#include <QObject>
#include <QStringList>
#include <QVariant>

#include <plugins/devices/device_plugin.h>
#include <plugins/devices/device_variants.h>

class DeviceAdapter : public ConfigAdapter {
  Q_OBJECT

  Q_PROPERTY(pc::devices::ui::WorkspaceDeviceStatus status READ status NOTIFY
                 statusChanged)

public:
  explicit DeviceAdapter(pc::devices::DevicePlugin *plugin,
                         QObject *parent = nullptr)
      : ConfigAdapter(parent), _plugin(plugin) {}

  using ConfigAdapter::setConfig;

  // Device-only config variant, we use this for any device plugins
  virtual bool setConfig(const pc::devices::DeviceConfigurationVariant &) = 0;

  bool setConfig(const pc::CoreConfigurationVariant &) override {
    // base configs not applicable
    return false;
  }

  // ----------------- identity -----------------
  int deviceIndex() const { return _deviceIndex; }
  void setDeviceIndex(int index) { _deviceIndex = index; }

  pc::devices::DevicePlugin *plugin() const { return _plugin; }

  // ----------------- status -----------------
  pc::devices::ui::WorkspaceDeviceStatus status() const { return _status; }

  void setStatusFromCore(pc::devices::DeviceStatus s) {
    const auto q = pc::devices::ui::toQt(s);
    if (q == _status) return;
    _status = q;
    emit statusChanged();
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

protected:
  pc::devices::DevicePlugin *_plugin = nullptr; // non-owning
  pc::devices::ui::WorkspaceDeviceStatus _status =
      pc::devices::ui::WorkspaceDeviceStatus::Unloaded;

  int _deviceIndex = -1;
};
