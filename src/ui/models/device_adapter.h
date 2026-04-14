#pragma once

#include "config_adapter.h"
#include "device_status.h"
#include "point_cloud_adapter.h"

#include <QObject>
#include <QStringList>
#include <QVariant>
#include <memory>
#include <plugins/devices/device_plugin.h>
#include <plugins/devices/device_variants.h>
#include <pointcaster/point_cloud.h>
#include <qtmetamacros.h>

class DeviceAdapter : public ConfigAdapter, public PointCloudAdapter {
  Q_OBJECT

  Q_PROPERTY(pc::devices::ui::WorkspaceDeviceStatus status READ status NOTIFY
                 statusChanged)

  Q_PROPERTY(
      bool pluginNullState READ pluginNullState NOTIFY pluginNullStateChanged)

  Q_PROPERTY(bool render READ render WRITE setRender NOTIFY renderChanged)

public:
  explicit DeviceAdapter(pc::devices::DevicePlugin *plugin,
                         QObject *parent = nullptr)
      : ConfigAdapter(parent), _plugin(plugin) {}

  using ConfigAdapter::setConfig;

  // Device-only config variant, we use this for any device plugins
  virtual bool setConfig(const pc::devices::DeviceConfigurationVariant &) = 0;

  bool setConfig(const pc::ConfigurationVariant &) override {
    // base configs not applicable
    return false;
  }

  // ----------------- identity -----------------
  int deviceIndex() const { return _deviceIndex; }
  void setDeviceIndex(int index) { _deviceIndex = index; }

  bool pluginNullState() const {
    if (!_plugin) return false;
    return _plugin->plugin_null_state();
  }

  pc::devices::DevicePlugin *plugin() const { return _plugin; }

  // ----------------- status -----------------
  pc::devices::ui::WorkspaceDeviceStatus status() const { return _status; }

  void setStatusFromCore(pc::devices::DeviceStatus s) {
    const auto q = pc::devices::ui::toQt(s);
    if (q == _status) return;
    _status = q;
    emit statusChanged();
  }

  bool render() const { return _render; }
  void setRender(bool render) {
    if (_render == render) return;
    _render = render;
    emit renderChanged();
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
    pc::logger()->debug("Running restart from the device adapter i.e. qml");
    _plugin->restart();
  }

  Q_INVOKABLE std::shared_ptr<pc::PointCloud> point_cloud() override {
    return _plugin->point_cloud();
  };

  Q_INVOKABLE PointCloudAdapter *pointCloudAdapter() {
    return static_cast<PointCloudAdapter *>(this);
  }

  void notifyFieldChanged(const QString &path) override {
    emit fieldChanged(path);
    if (_plugin) _plugin->on_config_field_changed(path.toStdString());
  }

  void notifyPointCloudUpdated() { emit pointCloudUpdated(); }

signals:
  void statusChanged();
  void renderChanged();
  void pluginNullStateChanged();

  void pointCloudUpdated();

protected:
  pc::devices::DevicePlugin *_plugin = nullptr; // non-owning
  pc::devices::ui::WorkspaceDeviceStatus _status =
      pc::devices::ui::WorkspaceDeviceStatus::Unloaded;
  bool _render = true;

  int _deviceIndex = -1;
};
