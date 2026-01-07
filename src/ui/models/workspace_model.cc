#include "workspace_model.h"
#include <QObject>
#include <QPointer>
#include <QString>
#include <QVariant>
#include <concepts>
#include <functional>
#include <print>
#include <qobject.h>
#include <qurl.h>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <workspace.h>

#include <plugins/devices/orbbec/orbbec_device_config.gen.h>
#include <plugins/devices/orbbec/orbbec_device_config.h>
#include <plugins/plugin_loader.h>

#include "device_status.h"
#include "device_plugin_controller.h"

namespace pc::ui {

using pc::devices::OrbbecDeviceConfiguration;
using pc::devices::OrbbecDeviceConfigurationAdapter;

WorkspaceModel::WorkspaceModel(pc::Workspace &workspace, QObject *parent,
                               std::function<void()> quit_callback)
    : QObject(parent), _workspace(workspace), _quit_callback(quit_callback) {
  rebuildAdapters();
}

void WorkspaceModel::close() {
  _workspace.devices.clear();
  _quit_callback();
}

void WorkspaceModel::loadFromFile(const QUrl &file) {
  const QString localPath = file.toLocalFile();
  if (localPath.isEmpty()) return;
  // load the file on a different thread, and after it's loaded we schedule the
  // main thread to apply the config file to the workspace
  std::jthread([&, path = localPath.toStdString()]() mutable {
    pc::WorkspaceConfiguration loaded_config;
    pc::load_workspace_from_file(loaded_config, path);
    _saveFileUrl = file;
    QMetaObject::invokeMethod(
        this,
        [this, cfg = std::move(loaded_config)]() mutable {
          _workspace.apply_new_config(std::move(cfg));
          rebuildAdapters();
        },
        Qt::QueuedConnection);
  }).detach();
  // TODO: is it healthy to do this?
  // are there cases where the jthread could spin forever?
}

void WorkspaceModel::save() {
  if (_saveFileUrl.isEmpty()) {
    emit openSaveAsDialog();
    return;
  }
  const QString localPath = _saveFileUrl.toLocalFile();
  std::jthread([path = localPath.toStdString(),
                workspace_config = _workspace.config] {
    save_workspace_to_file(workspace_config, path);
  }).detach();
}

QVariant WorkspaceModel::deviceConfigAdapters() const {
  return QVariant::fromValue(_deviceConfigAdapters);
}

void WorkspaceModel::rebuildAdapters() {

  qDeleteAll(_deviceConfigAdapters);
  _deviceConfigAdapters.clear();

  for (QObject *obj : _deviceControllers) obj->deleteLater();
  _deviceControllers.clear();

  for (auto &device_plugin : _workspace.devices) {
    auto *plugin = device_plugin.get();
    std::visit(
        [this, plugin](auto &device_config) {
          using ConfigType = std::decay_t<decltype(device_config)>;
          if constexpr (std::same_as<ConfigType, OrbbecDeviceConfiguration>) {
            auto *adapter = new OrbbecDeviceConfigurationAdapter(device_config,
                                                                 plugin, this);
            _deviceConfigAdapters.append(adapter);

            QString device_id = QString::fromStdString(device_config.id);

            auto *controller =
                new DevicePluginController(plugin, device_id, this);
            _deviceControllers.append(controller);

            // TODO
            // perhaps the configuration adapter and controller could be merged?

            plugin->set_status_callback(
                [adapter = QPointer<OrbbecDeviceConfigurationAdapter>(adapter),
                 controller = QPointer<DevicePluginController>(controller)](
                    pc::devices::DeviceStatus status) {
                  if (adapter) {
                    QMetaObject::invokeMethod(
                        adapter,
                        [adapter, status]() {
                          if (!adapter) return;
                          adapter->setStatusFromCore(status);
                        },
                        Qt::QueuedConnection);
                  }

                  if (controller) {
                    QMetaObject::invokeMethod(
                        controller,
                        [controller, status]() {
                          if (!controller) return;
                          controller->setStatusFromCore(status);
                        },
                        Qt::QueuedConnection);
                  }
                });
          }
        },
        device_plugin->config());
  }

  emit deviceConfigAdaptersChanged();
  emit deviceControllersChanged();
}

} // namespace pc::ui
