#include "workspace_model.h"

#include "app_settings/app_settings.h"
#include "models/device_adapter.h"
#include <QHash>
#include <QMetaObject>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QUndoCommand>
#include <QVariant>
#include <core/logger/logger.h>
#include <core/uuid/uuid.h>
#include <functional>
#include <plugins/devices/device_variants.h>
#include <plugins/devices/orbbec/orbbec_device_adapter.gen.h>
#include <plugins/devices/orbbec/orbbec_device_config.h>
#include <session/session_config.h>
#include <session/session_config_adapter.gen.h>
#include <string>
#include <thread>
#include <variant>
#include <workspace/workspace.h>

namespace pc::ui {

using pc::SessionConfiguration;
using pc::SessionConfigurationAdapter;
using pc::devices::OrbbecDeviceConfiguration;

namespace {

// -------- helpers --------

static int find_session_index_by_id(const pc::WorkspaceConfiguration &cfg,
                                    const std::string &session_id) {
  for (int i = 0; i < int(cfg.sessions.size()); ++i) {
    if (cfg.sessions[size_t(i)].id == session_id) return i;
  }
  return -1;
}

static int find_device_index_by_id(const pc::WorkspaceConfiguration &cfg,
                                   const std::string &device_id) {
  for (int i = 0; i < int(cfg.devices.size()); ++i) {
    const auto &variant = cfg.devices[size_t(i)];
    const bool match =
        std::visit([&](const auto &d) { return d.id == device_id; }, variant);
    if (match) return i;
  }
  return -1;
}

// -------- undo commands --------

class SetSessionConfigCommand final : public QUndoCommand {
public:
  using ApplyFn = std::function<void(pc::WorkspaceConfiguration)>;

  SetSessionConfigCommand(QString session_id, SessionConfiguration before_cfg,
                          SessionConfiguration after_cfg, QString command_text,
                          pc::WorkspaceConfiguration base_config_snapshot,
                          ApplyFn apply_fn)
      : QUndoCommand(std::move(command_text)),
        _session_id(std::move(session_id)), _before(std::move(before_cfg)),
        _after(std::move(after_cfg)),
        _base_snapshot(std::move(base_config_snapshot)),
        _apply_fn(std::move(apply_fn)) {}

  void undo() override { apply(_before); }
  void redo() override { apply(_after); }

private:
  QString _session_id;
  SessionConfiguration _before;
  SessionConfiguration _after;
  pc::WorkspaceConfiguration _base_snapshot;
  ApplyFn _apply_fn;

  void apply(const SessionConfiguration &cfg_value) {
    auto new_cfg = _base_snapshot;

    const int idx =
        find_session_index_by_id(new_cfg, _session_id.toStdString());
    if (idx < 0 || idx >= int(new_cfg.sessions.size())) {
      pc::logger()->warn("SetSessionConfigCommand: session id not found '{}'",
                         _session_id.toStdString());
      return;
    }

    new_cfg.sessions[size_t(idx)] = cfg_value;

    if (_apply_fn) _apply_fn(std::move(new_cfg));
  }
};

class SetDeviceConfigCommand final : public QUndoCommand {
public:
  using ApplyFn = std::function<void(pc::WorkspaceConfiguration)>;

  SetDeviceConfigCommand(QString device_id,
                         pc::devices::DeviceConfigurationVariant before_cfg,
                         pc::devices::DeviceConfigurationVariant after_cfg,
                         QString command_text,
                         pc::WorkspaceConfiguration base_config_snapshot,
                         ApplyFn apply_fn)
      : QUndoCommand(std::move(command_text)), _device_id(std::move(device_id)),
        _before(std::move(before_cfg)), _after(std::move(after_cfg)),
        _base_snapshot(std::move(base_config_snapshot)),
        _apply_fn(std::move(apply_fn)) {}

  void undo() override { apply(_before); }
  void redo() override { apply(_after); }

private:
  QString _device_id;
  pc::devices::DeviceConfigurationVariant _before;
  pc::devices::DeviceConfigurationVariant _after;
  pc::WorkspaceConfiguration _base_snapshot;
  ApplyFn _apply_fn;

  void apply(const pc::devices::DeviceConfigurationVariant &cfg_value) {
    auto new_cfg = _base_snapshot;

    const int idx = find_device_index_by_id(new_cfg, _device_id.toStdString());
    if (idx < 0 || idx >= int(new_cfg.devices.size())) {
      pc::logger()->warn("SetDeviceConfigCommand: device id not found '{}'",
                         _device_id.toStdString());
      return;
    }

    new_cfg.devices[size_t(idx)] = cfg_value;

    if (_apply_fn) _apply_fn(std::move(new_cfg));
  }
};

} // namespace

// ----------------- WorkspaceModel helpers -----------------

QString WorkspaceModel::adapterStableId(ConfigAdapter *adapter) {
  if (!adapter) return {};
  const QVariant v = adapter->value(QStringLiteral("id"));
  if (v.isValid()) return v.toString();
  return {};
}

QString WorkspaceModel::adapterStableId(DeviceAdapter *adapter) {
  if (!adapter) return {};
  const QVariant v = adapter->value(QStringLiteral("id"));
  if (v.isValid()) return v.toString();
  return {};
}

void WorkspaceModel::applyWorkspaceConfigAndRebuild(
    pc::WorkspaceConfiguration new_config) {
  _workspace.apply_new_config(std::move(new_config));
  syncAdapters();
}

// ----------------- WorkspaceModel -----------------

WorkspaceModel::WorkspaceModel(pc::Workspace *workspace, QObject *parent,
                               std::function<void()> quit_callback)
    : QObject(parent), _workspace(*workspace),
      _quit_callback(std::move(quit_callback)) {
  if (workspace->auto_loaded_config) {
    const auto path = AppSettings::instance()->lastSessionPath();
    setSaveFileUrl(QUrl::fromLocalFile(path));
  }
  syncAdapters();
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
          applyWorkspaceConfigAndRebuild(std::move(cfg));
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

  QString local_path = saveFileUrl().toLocalFile();
  if (local_path.isEmpty()) {
    emit openSaveAsDialog();
    return;
  }

  using namespace Qt::StringLiterals;
  if (!local_path.endsWith(u".yaml"_s, Qt::CaseInsensitive)) {
    local_path += u".yaml"_s;
  }

  std::jthread([workspace_config = _workspace.config, local_path,
                update_last_session_path] {
    save_workspace_to_file(workspace_config, local_path.toStdString());
    if (update_last_session_path) {
      AppSettings::instance()->setLastSessionPath(local_path);
    }
  }).detach();
}

QList<QObject *> WorkspaceModel::sessionAdapters() const {
  return _sessionAdapters;
}

QVariant WorkspaceModel::deviceAdapters() const {
  return QVariant::fromValue(_deviceAdapters);
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

    const auto discovered_devices = it->second->discovered_devices();
    for (const auto &device : discovered_devices) {
      QVariantMap item;
      item["kind"] = "discovered";
      item["plugin_name"] = QString::fromStdString(plugin_name);
      item["ip"] = QString::fromStdString(device.ip);
      item["id"] = QString::fromStdString(device.id);
      item["label"] = QString::fromStdString(device.label);
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
    applyWorkspaceConfigAndRebuild(std::move(new_config));
  } else {
    syncAdapters();
  }
}

void WorkspaceModel::deleteSelectedDevice() {
  if (_deviceAdapters.isEmpty()) return;

  const int idx = selectedDeviceIndex();
  if (idx < 0 || idx >= _deviceAdapters.size()) {
    pc::logger()->warn(
        "deleteSelectedDevice: invalid selectedDeviceIndex={} size={}", idx,
        _deviceAdapters.size());
    return;
  }

  auto *adapter = qobject_cast<DeviceAdapter *>(_deviceAdapters[idx]);
  const QString device_id_q = adapterStableId(adapter);
  if (device_id_q.isEmpty()) {
    pc::logger()->warn("deleteSelectedDevice: selected adapter has no id");
    return;
  }

  auto new_cfg = _workspace.config;
  const int device_index =
      find_device_index_by_id(new_cfg, device_id_q.toStdString());
  if (device_index < 0 || device_index >= int(new_cfg.devices.size())) {
    pc::logger()->warn("deleteSelectedDevice: device id not found '{}'",
                       device_id_q.toStdString());
    return;
  }

  pc::logger()->trace("Deleting device id='{}' (index {})",
                      device_id_q.toStdString(), device_index);

  new_cfg.devices.erase(new_cfg.devices.begin() + device_index);
  applyWorkspaceConfigAndRebuild(std::move(new_cfg));
}

void WorkspaceModel::initSessionAdapter(SessionConfigurationAdapter *adapter) {
  if (!adapter) return;

  QObject::connect(adapter, &ConfigAdapter::editRequested, this,
                   [this, adapter](const QString &path, const QVariant &value) {
                     auto *session_adapter =
                         qobject_cast<SessionConfigurationAdapter *>(adapter);
                     if (!session_adapter) return;

                     const QString session_id_q = session_adapter->id();
                     if (session_id_q.isEmpty()) return;

                     SessionConfiguration before;
                     SessionConfiguration after;
                     pc::WorkspaceConfiguration base_snapshot;
                     QString command_text;

                     {
                       std::scoped_lock lock2(_workspace.config_access);
                       base_snapshot = _workspace.config;

                       const int idx = find_session_index_by_id(
                           _workspace.config, session_id_q.toStdString());
                       if (idx < 0) return;

                       before = _workspace.config.sessions[size_t(idx)];

                       const bool changed = session_adapter->apply(path, value);
                       if (!changed) return;

                       after = _workspace.config.sessions[size_t(idx)];
                     }

                     command_text = QStringLiteral("Edit %1").arg(path);

                     _undoStack->push(new SetSessionConfigCommand(
                         session_id_q, std::move(before), std::move(after),
                         std::move(command_text), std::move(base_snapshot),
                         [this](pc::WorkspaceConfiguration cfg) {
                           QMetaObject::invokeMethod(
                               this,
                               [this, cfg = std::move(cfg)]() mutable {
                                 applyWorkspaceConfigAndRebuild(std::move(cfg));
                               },
                               Qt::QueuedConnection);
                         }));
                   });
}

template <typename AdapterT>
void WorkspaceModel::initDeviceAdapter(AdapterT *adapter,
                                       pc::devices::DevicePlugin *plugin) {
  QObject::connect(
      adapter, &ConfigAdapter::editRequested, this,
      [this, adapter](const QString &path, const QVariant &value) {
        auto *device_adapter = qobject_cast<DeviceAdapter *>(adapter);
        if (!device_adapter) return;

        const QString device_id_q = adapterStableId(device_adapter);
        if (device_id_q.isEmpty()) return;

        pc::devices::DeviceConfigurationVariant before;
        pc::devices::DeviceConfigurationVariant after;
        pc::WorkspaceConfiguration base_snapshot;
        QString command_text;

        {
          std::scoped_lock lock2(_workspace.config_access);
          base_snapshot = _workspace.config;

          const int idx = find_device_index_by_id(_workspace.config,
                                                  device_id_q.toStdString());
          if (idx < 0) return;

          before = _workspace.config.devices[size_t(idx)];
        }

        const bool changed = device_adapter->apply(path, value);
        if (!changed) return;

        auto *p = device_adapter->plugin();
        if (!p) return;
        after = p->config();

        {
          std::scoped_lock lock3(_workspace.config_access);
          const int idx = find_device_index_by_id(_workspace.config,
                                                  device_id_q.toStdString());
          if (idx < 0) return;
          _workspace.config.devices[size_t(idx)] = after;
        }

        command_text = QStringLiteral("Edit %1").arg(path);

        _undoStack->push(new SetDeviceConfigCommand(
            device_id_q, std::move(before), std::move(after),
            std::move(command_text), std::move(base_snapshot),
            [this](pc::WorkspaceConfiguration cfg) {
              QMetaObject::invokeMethod(
                  this,
                  [this, cfg = std::move(cfg)]() mutable {
                    applyWorkspaceConfigAndRebuild(std::move(cfg));
                  },
                  Qt::QueuedConnection);
            }));
      });

  plugin->set_status_callback([adapterPtr = QPointer<AdapterT>(adapter)](
                                  pc::devices::DeviceStatus status) {
    if (!adapterPtr) return;
    QMetaObject::invokeMethod(
        adapterPtr.data(),
        [adapterPtr, status]() {
          if (!adapterPtr) return;
          adapterPtr->setStatusFromCore(status);
        },
        Qt::QueuedConnection);
  });
}

DeviceAdapter *WorkspaceModel::makeDeviceAdapterForPlugin(
    pc::devices::DevicePlugin *plugin,
    pc::devices::DeviceConfigurationVariant &config_variant) {
  DeviceAdapter *result = nullptr;

  std::visit(
      [this, plugin, &result](auto &device_config) {
        using ConfigType = std::decay_t<decltype(device_config)>;
        using AdapterType =
            pc::devices::device_adapter_for_config_t<ConfigType>;

        auto *adapter = new AdapterType(device_config, plugin, this);
        initDeviceAdapter(adapter, plugin);
        result = adapter;
      },
      config_variant);

  return result;
}

void WorkspaceModel::syncAdapters() {
  pc::logger()->trace("Syncing adapters (sessions + devices)");

  // ---------------- sessions ----------------
  {
    QHash<QString, SessionConfigurationAdapter *> existing_by_id;
    existing_by_id.reserve(_sessionAdapters.size());

    for (QObject *obj : _sessionAdapters) {
      auto *a = qobject_cast<SessionConfigurationAdapter *>(obj);
      if (!a) continue;
      const QString id = a->id();
      if (!id.isEmpty()) existing_by_id.insert(id, a);
    }

    QList<QObject *> new_ordered_sessions;
    QHash<QString, const pc::SessionConfiguration *> new_ptrs_by_id;

    {
      std::scoped_lock lock(_workspace.config_access);

      new_ordered_sessions.reserve(int(_workspace.config.sessions.size()));
      new_ptrs_by_id.reserve(int(_workspace.config.sessions.size()));

      for (auto &session_cfg : _workspace.config.sessions) {
        const QString id = QString::fromStdString(session_cfg.id);
        if (id.isEmpty()) continue;

        const pc::SessionConfiguration *current_ptr = &session_cfg;

        SessionConfigurationAdapter *adapter = existing_by_id.take(id);

        const bool ptr_matches =
            _sessionConfigPtrById.contains(id) &&
            (_sessionConfigPtrById.value(id) == current_ptr);

        if (adapter && !ptr_matches) {
          // Underlying SessionConfiguration moved (vector realloc / reorder).
          // Old adapter holds a dangling reference: must recreate.
          adapter->deleteLater();
          adapter = nullptr;
        }

        if (adapter) {
          (void)adapter->setConfig(session_cfg);
        } else {
          auto *new_adapter =
              new SessionConfigurationAdapter(session_cfg, this);
          initSessionAdapter(new_adapter);
          adapter = new_adapter;
        }

        if (adapter) {
          new_ordered_sessions.append(adapter);
          new_ptrs_by_id.insert(id, current_ptr);
        }
      }
    }

    // Delete adapters for removed sessions (anything left in existing_by_id).
    for (auto it = existing_by_id.begin(); it != existing_by_id.end(); ++it) {
      if (it.value()) it.value()->deleteLater();
    }

    _sessionAdapters = new_ordered_sessions;
    _sessionConfigPtrById = std::move(new_ptrs_by_id);

    emit sessionAdaptersChanged();
  }

  // ---------------- devices ----------------
  {
    QHash<QString, DeviceAdapter *> existing_by_id;
    existing_by_id.reserve(_deviceAdapters.size());

    for (QObject *obj : _deviceAdapters) {
      auto *a = qobject_cast<DeviceAdapter *>(obj);
      if (!a) continue;
      const QString id = adapterStableId(a);
      if (!id.isEmpty()) existing_by_id.insert(id, a);
    }

    QList<QObject *> new_ordered_devices;
    new_ordered_devices.reserve(_deviceAdapters.size());

    std::scoped_lock lock(_workspace.config_access);

    auto device_id_from_variant =
        [](pc::devices::DeviceConfigurationVariant &v) -> QString {
      return std::visit(
          [](auto &cfg) -> QString { return QString::fromStdString(cfg.id); },
          v);
    };

    pc::logger()->trace("Device count: {}", _workspace.devices.size());
    for (auto &device_plugin : _workspace.devices) {
      auto *plugin = device_plugin.get();
      if (!plugin) continue;

      auto &cfg_variant = plugin->config();
      const QString id = device_id_from_variant(cfg_variant);
      if (id.isEmpty()) continue;

      DeviceAdapter *adapter = existing_by_id.take(id);

      if (adapter) {
        if (!adapter->setConfig(cfg_variant)) {
          adapter->deleteLater();
          adapter = nullptr;
        }
      }

      if (!adapter) {
        adapter = makeDeviceAdapterForPlugin(plugin, cfg_variant);
      } else {
        plugin->set_status_callback(
            [adapterPtr = QPointer<DeviceAdapter>(adapter)](
                pc::devices::DeviceStatus status) {
              if (!adapterPtr) return;
              QMetaObject::invokeMethod(
                  adapterPtr.data(),
                  [adapterPtr, status]() {
                    if (!adapterPtr) return;
                    adapterPtr->setStatusFromCore(status);
                  },
                  Qt::QueuedConnection);
            });
      }

      if (adapter) new_ordered_devices.append(adapter);
    }

    for (auto it = existing_by_id.begin(); it != existing_by_id.end(); ++it) {
      if (it.value()) it.value()->deleteLater();
    }

    _deviceAdapters = new_ordered_devices;

    emit deviceAdaptersChanged();
    emit deviceVariantNamesChanged();
    emit addDeviceMenuEntriesChanged();
  }

  pc::logger()->trace("Finished syncing adapters");
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

} // namespace pc::ui
