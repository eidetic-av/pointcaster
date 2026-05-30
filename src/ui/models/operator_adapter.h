#pragma once

#include "config_adapter.h"

#include <QObject>
#include <QString>
#include <plugins/operators/operator_host.h>
#include <plugins/operators/operator_plugin.h>
#include <plugins/operators/operator_variants.h>

// TODO this shouldn't be here
#include <plugins/devices/device_plugin.h>

class OperatorAdapter : public QObject {
  Q_OBJECT
  Q_PROPERTY(ConfigAdapter *configAdapter READ configAdapter NOTIFY
                 configAdapterChanged)

public:
  explicit OperatorAdapter(pc::operators::OperatorPlugin *plugin,
                           pc::operators::OperatorHost *host,
                           QObject *parent = nullptr)
      : QObject(parent), _plugin(plugin), _host(host) {}

  ConfigAdapter *configAdapter() const { return _configAdapter; }

  void setConfigAdapter(ConfigAdapter *adapter) {
    if (_configAdapter == adapter) return;
    if (_configAdapter) {
      disconnect(_configAdapter, nullptr, this, nullptr);
      _configAdapter->deleteLater();
    }
    _configAdapter = adapter;
    if (_configAdapter) {
      _configAdapter->setParent(this);
      connect(_configAdapter, &ConfigAdapter::fieldChanged, this,
              [this](const QString &path) {
                if (_plugin) {
                  _plugin->on_config_field_changed(path.toStdString());
                  // Forward to pipeline worker instances
                  if (auto *device =
                          dynamic_cast<pc::devices::DevicePlugin *>(_host)) {
                    auto updated_config = _plugin->config_variant();
                    device->update_operator_in_pipeline(updated_config,
                                                        path.toStdString());
                  }
                }
                if (_host) _host->reprocess();
              });
    }
    emit configAdapterChanged();
  }

  pc::operators::OperatorPlugin *plugin() const { return _plugin; }

signals:
  void configAdapterChanged();

private:
  pc::operators::OperatorPlugin *_plugin = nullptr;
  ConfigAdapter *_configAdapter = nullptr;
  pc::operators::OperatorHost *_host = nullptr;
};