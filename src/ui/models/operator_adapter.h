#pragma once
#include "config_adapter.h"
#include <QObject>
#include <QString>
#include <plugins/operators/operator_host.h>
#include <plugins/operators/operator_plugin.h>
#include <plugins/operators/operator_variants.h>

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
    pc::logger()->trace("[setCfgAdapter] enter new={} old={}",
                        fmt::ptr(static_cast<void *>(adapter)),
                        fmt::ptr(static_cast<void *>(_configAdapter)));
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
                if (_configAdapter && _configAdapter->isRefreshingAllFields())
                  return;
                if (_configAdapter && _configAdapter->isOutput(path)) return;
                if (_plugin) {
                  _plugin->on_config_field_changed(path.toStdString());

                  if (path == QStringLiteral("backend")) {
                    const auto backend_in_use =
                        static_cast<int>(_plugin->current_backend_type());
                    if (_configAdapter->value(path).toInt() != backend_in_use) {
                      _configAdapter->set(path, backend_in_use);
                      return;
                    }
                  }

                  if (_host) {
                    auto updated_config = _plugin->config_variant();
                    _host->update_operator_in_pipeline(updated_config,
                                                       path.toStdString());
                  }
                }
                if (_host) _host->reprocess();
              });
    }
    pc::logger()->trace("[setCfgAdapter] done");
    emit configAdapterChanged();
  }

  pc::operators::OperatorPlugin *plugin() const { return _plugin; }

  pc::operators::OperatorHost *host() const { return _host; }

  void invalidatePlugin() {
    _plugin = nullptr;
    _host = nullptr;
  }

signals:
  void configAdapterChanged();

private:
  pc::operators::OperatorPlugin *_plugin = nullptr;
  ConfigAdapter *_configAdapter = nullptr;
  pc::operators::OperatorHost *_host = nullptr;
};