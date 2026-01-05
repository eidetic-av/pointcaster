#include "workspace_model.h"
#include <QObject>
#include <QString>
#include <QVariant>
#include <functional>
#include <workspace.h>
#include <string_view>

#include "../../plugins/devices/orbbec/orbbec_device.h"
#include "../../plugins/devices/orbbec/orbbec_device_config.gen.h"
#include "plugins/devices/orbbec/orbbec_device_config.h"

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

void WorkspaceModel::loadFromFile(const QUrl& file) {
    const QString localPath = file.toLocalFile();
    if (localPath.isEmpty()) return;

    pc::WorkspaceConfiguration loaded_config;

    Workspace::load_config_from_file(
        loaded_config,
        localPath.toStdString()
    );

    _workspace.config = std::move(loaded_config);
    _workspace.revert_config();
    rebuildAdapters();
}

QVariant WorkspaceModel::deviceConfigAdapters() const {
  return QVariant::fromValue(_deviceConfigAdapters);
}

void WorkspaceModel::rebuildAdapters() {
  qDeleteAll(_deviceConfigAdapters);
  _deviceConfigAdapters.clear();

  for (auto &device_plugin : _workspace.devices) {
    if (auto *device =
            dynamic_cast<devices::OrbbecDevice *>(device_plugin.get())) {
      addOrbbecDeviceAdapter(device->config());
    } else {
    }
  }

  emit deviceConfigAdaptersChanged();
}

void WorkspaceModel::addOrbbecDeviceAdapter(OrbbecDeviceConfiguration &config) {
  auto *adapter = new OrbbecDeviceConfigurationAdapter(config, this);
  _deviceConfigAdapters.append(adapter);
}

} // namespace pc::ui
