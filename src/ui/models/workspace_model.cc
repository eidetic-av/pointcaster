#include "workspace_model.h"
#include <QObject>
#include <QString>
#include <QVariant>
#include <functional>
#include <qurl.h>
#include <workspace.h>
#include <thread>
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
