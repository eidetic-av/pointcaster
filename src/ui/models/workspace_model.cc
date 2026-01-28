#include "workspace_model.h"

#include "app_settings/app_settings.h"
#include "commands/device_config_commands.h"
#include "device_plugin_controller.h"
#include <QMetaType>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QUndoCommand>
#include <QVariant>
#include <concepts>
#include <core/logger/logger.h>
#include <core/uuid/uuid.h>
#include <functional>
#include <plugins/devices/device_variants.h>
#include <plugins/devices/orbbec/orbbec_device_config.gen.h>
#include <plugins/devices/orbbec/orbbec_device_config.h>
#include <print>
#include <string>
#include <thread>
#include <variant>
#include <workspace.h>

namespace pc::ui {

using pc::devices::OrbbecDeviceConfiguration;
using pc::devices::OrbbecDeviceConfigurationAdapter;

WorkspaceModel::WorkspaceModel(pc::Workspace *workspace, QObject *parent,
                               std::function<void()> quit_callback)
    : QObject(parent), _workspace(*workspace),
      _quit_callback(std::move(quit_callback)) {
  if (workspace->auto_loaded_config) {
    const auto path = AppSettings::instance()->lastSessionPath();
    setSaveFileUrl(QUrl::fromLocalFile(path));
  }
  rebuildAdapters();
}

void WorkspaceModel::close() {
  _workspace.devices.clear();
  _quit_callback();
}

void WorkspaceModel::loadFromFile(const QUrl &file) {
  const QString local_path = file.toLocalFile();
  if (local_path.isEmpty()) return;

  std::jthread([&, local_path]() mutable {
    pc::WorkspaceConfiguration loaded_config;
    pc::load_workspace_from_file(loaded_config, local_path.toStdString());
    QMetaObject::invokeMethod(
        this,
        [this, file, local_path, cfg = std::move(loaded_config)]() mutable {
          setSaveFileUrl(file);
          _workspace.apply_new_config(std::move(cfg));
          rebuildAdapters();
          AppSettings::instance()->setLastSessionPath(local_path);
        },
        Qt::QueuedConnection);
  }).detach();
}

void WorkspaceModel::save(bool update_last_session_path) {
  if (saveFileUrl().isEmpty()) {
    emit openSaveAsDialog();
    return;
  }

  const QString local_path = saveFileUrl().toLocalFile();
  if (local_path.isEmpty()) {
    emit openSaveAsDialog();
    return;
  }

  std::jthread([workspace_config = _workspace.config, local_path,
                update_last_session_path] {
    save_workspace_to_file(workspace_config, local_path.toStdString());
    if (update_last_session_path) {
      AppSettings::instance()->setLastSessionPath(local_path);
    }
  }).detach();
}

QVariant WorkspaceModel::deviceConfigAdapters() const {
  return QVariant::fromValue(_deviceConfigAdapters);
}

QStringList WorkspaceModel::deviceVariantNames() const {
  QStringList names;
  names.reserve(static_cast<int>(
      std::variant_size_v<pc::devices::DeviceConfigurationVariant>));
  pc::devices::for_each_device_config_type(
      [&names]<typename DeviceConfigType>() {
        names.push_back(QString::fromStdString(DeviceConfigType::PluginName));
      });
  return names;
}

QVariantList WorkspaceModel::addDeviceMenuEntries() const {
  QVariantList entries;

  for (auto plugin_name : _workspace.loaded_device_plugin_names) {
    auto it = _workspace.discovery_plugins.find(plugin_name);
    if (it == _workspace.discovery_plugins.end() || !it->second) continue;
    auto discovered_devices = it->second->discovered_devices();

    for (const auto &device : discovered_devices) {
      QVariantMap item;
      item["kind"] = "discovered";
      item["plugin_name"] = QString::fromStdString(plugin_name);
      item["ip"] = QString::fromStdString(device.ip);
      item["id"] = QString::fromStdString(device.id);
      item["display_name"] = QString::fromStdString(device.display_name);
      entries.push_back(item);
    }
  }

  return entries;
}

void WorkspaceModel::addNewDevice(const QString &plugin_name,
                                  const QString &target_ip) {
  if (plugin_name == OrbbecDeviceConfiguration::PluginName) {
    auto config =
        pc::devices::OrbbecDeviceConfiguration{.id = pc::uuid::word()};
    if (!target_ip.isEmpty()) config.ip = target_ip.toStdString();

    auto new_config = _workspace.config;
    new_config.devices.push_back(std::move(config));
    _workspace.apply_new_config(new_config);
  }
  rebuildAdapters();
}

void WorkspaceModel::deleteSelectedDevice() {
  auto new_config = _workspace.config;
  const auto index = selectedDeviceIndex();
  if (index < 0 || index >= new_config.devices.size()) {
    pc::logger()->warn("deleteSelectedDevice: invalid index={} size={}", index,
                       new_config.devices.size());
    return;
  }
  pc::logger()->trace("Deleting selected device index: {}", index);
  new_config.devices.erase(new_config.devices.begin() + index);
  _workspace.apply_new_config(std::move(new_config));
  rebuildAdapters();
}

void WorkspaceModel::rebuildAdapters() {
  pc::logger()->trace("Rebuilding device configuration adapters");
  qDeleteAll(_deviceConfigAdapters);
  _deviceConfigAdapters.clear();

  for (QObject *obj : _deviceControllers) obj->deleteLater();
  _deviceControllers.clear();

  std::scoped_lock lock(_workspace.config_access);

  const auto find_device_index = [this](const std::string &id) -> int {
    for (int i = 0; i < int(_workspace.config.devices.size()); ++i) {
      const auto &variant = _workspace.config.devices[size_t(i)];
      const bool match = std::visit(
          [&](const auto &cfg) -> bool { return cfg.id == id; }, variant);
      if (match) return i;
    }
    return -1;
  };

  pc::logger()->trace("Device count: {}", _workspace.devices.size());
  for (auto &device_plugin : _workspace.devices) {
    auto *plugin = device_plugin.get();

    std::visit(
        [this, plugin, &find_device_index](auto &device_config) {
          using ConfigType = std::decay_t<decltype(device_config)>;

          if constexpr (std::same_as<ConfigType, OrbbecDeviceConfiguration>) {
            auto *adapter = new OrbbecDeviceConfigurationAdapter(device_config,
                                                                 plugin, this);

            const int deviceIndex = find_device_index(device_config.id);
            adapter->setDeviceIndex(deviceIndex);

            connect(adapter, &ConfigAdapter::fieldEditRequested, this,
                    [this, adapter](int fieldIndex, const QVariant &v) {
                      onAdapterFieldEditRequested(adapter, fieldIndex, v);
                    });

            _deviceConfigAdapters.append(adapter);

            const QString device_id = QString::fromStdString(device_config.id);
            auto *controller =
                new DevicePluginController(plugin, device_id, this);
            _deviceControllers.append(controller);

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
  emit deviceVariantNamesChanged();
  emit addDeviceMenuEntriesChanged();

  pc::logger()->trace("Finished rebuilding device configuration adapters");
}

void WorkspaceModel::refreshDeviceAdapterFromCore(int deviceIndex) {
  if (deviceIndex < 0) return;

  pc::devices::DeviceConfigurationVariant variant;
  {
    std::scoped_lock lock(_workspace.config_access);
    const std::size_t i = static_cast<std::size_t>(deviceIndex);
    if (i >= _workspace.config.devices.size()) return;
    variant = _workspace.config.devices[i];
  }

  for (QObject *obj : _deviceConfigAdapters) {
    auto *adapter = qobject_cast<ConfigAdapter *>(obj);
    if (!adapter) continue;
    if (adapter->deviceIndex() != deviceIndex) continue;

    adapter->setConfigFromCore(variant);
    return;
  }
}

void WorkspaceModel::triggerDeviceDiscovery() {
  for (const auto &plugin_name : _workspace.loaded_device_plugin_names) {
    auto it = _workspace.discovery_plugins.find(plugin_name);
    if (it == _workspace.discovery_plugins.end() || !it->second) continue;

    auto &discovery_device = it->second;
    if (!discovery_device->has_discovery_change_callback()) {
      discovery_device->add_discovery_change_callback([this] {
        QMetaObject::invokeMethod(
            this, [this] { emit addDeviceMenuEntriesChanged(); },
            Qt::QueuedConnection);
      });
    }
    discovery_device->refresh_discovery();
  }
}

void WorkspaceModel::onAdapterFieldEditRequested(ConfigAdapter *adapter,
                                                 int fieldIndex,
                                                 const QVariant &value) {
  if (!adapter) return;

  const int device_index = adapter->deviceIndex();
  if (device_index < 0) return;

  pc::devices::DeviceConfigurationVariant before;
  pc::devices::DeviceConfigurationVariant after;
  QString command_text;

  {
    std::scoped_lock lock(_workspace.config_access);
    if (std::size_t(device_index) >= _workspace.config.devices.size()) return;

    auto *plugin = adapter->plugin();
    if (!plugin) return;

    const QVariant old_value = adapter->fieldValue(fieldIndex);

    before = _workspace.config.devices[std::size_t(device_index)];

    const bool changed = adapter->applyFieldValue(fieldIndex, value);
    if (!changed) return;

    const QVariant new_value = adapter->fieldValue(fieldIndex);

    pc::logger()->trace(
        "[UI changed DeviceConfig] deviceIndex={} field={} old='{}' new='{}'",
        device_index, adapter->fieldName(fieldIndex).toStdString(),
        old_value.toString().toStdString(), new_value.toString().toStdString());

    adapter->notifyFieldChanged(fieldIndex);

    after = plugin->config();

    // revert plugin cache so command redo() is the mutator, and thats what
    // undoStack->push does
    plugin->update_config(before);

    command_text =
        QStringLiteral("Edit %1").arg(adapter->fieldName(fieldIndex));
  }

  _undoStack->push(new SetDeviceConfigCommand(
      _workspace, device_index, fieldIndex, std::move(before), std::move(after),
      std::move(command_text), [this](int idx) {
        QMetaObject::invokeMethod(
            this, [this, idx] { refreshDeviceAdapterFromCore(idx); },
            Qt::QueuedConnection);
      }));
}

} // namespace pc::ui
