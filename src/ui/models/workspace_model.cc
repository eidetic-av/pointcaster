#include "workspace_model.h"

#include "app_settings/app_settings.h"
#include "camera_image_provider.h"
#include "layout_saver.h"
#include "models/device_adapter.h"
#include "models/session_recorder_model.h"
#include "plugins/devices/ply/ply_device_config.h"
#include <QHash>
#include <QMetaObject>
#include <QObject>
#include <QPointer>
#include <QSet>
#include <QString>
#include <QUndoCommand>
#include <QVariant>
#include <algorithm>
#include <chrono>
#include <core/logger/logger.h>
#include <core/uuid/uuid.h>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <plugins/devices/device_variants.h>
#include <ranges>
#include <session/session.h>
#include <session/session_config.h>
#include <session/session_config_adapter.gen.h>
#include <spdlog/common.h>
#include <string>
#include <thread>
#include <ui/layout_saver.h>
#include <variant>
#include <workspace/workspace.h>

// TODO these absolutely need to be polymorphic,
// no way they can be defined at this compile stage
#include <plugins/devices/orbbec/orbbec_device_adapter.gen.h>
#include <plugins/devices/orbbec/orbbec_device_config.h>
#include <plugins/devices/ply/ply_device_adapter.gen.h>
#include <plugins/devices/ply/ply_device_config.h>
#include <plugins/operators/fringe_removal/fringe_removal_config_adapter.gen.h>

namespace pc::ui {

using pc::SessionConfiguration;
using pc::SessionConfigurationAdapter;

using pc::devices::OrbbecDeviceConfiguration;
using pc::devices::PlyDeviceConfiguration;

namespace {

// -------- helpers --------

static int find_session_index_by_id(const pc::WorkspaceConfiguration &config,
                                    const std::string &session_id) {
  for (int i = 0; i < int(config.sessions.size()); ++i) {
    if (config.sessions[size_t(i)].id == session_id) return i;
  }
  return -1;
}

static int find_device_index_by_id(const pc::WorkspaceConfiguration &config,
                                   const std::string &device_id) {
  for (int i = 0; i < int(config.devices.size()); ++i) {
    const auto &variant = config.devices[size_t(i)];
    const bool match =
        std::visit([&](const auto &d) { return d.id == device_id; }, variant);
    if (match) return i;
  }
  return -1;
}
static int find_operator_index_by_id(
    const pc::devices::DeviceConfigurationVariant &device_config,
    const std::string &operator_id) {
  return std::visit(
      [&](const auto &dev) -> int {
        for (int i = 0; i < int(dev.operators.size()); ++i) {
          auto [id, pn] =
              pc::operators::operator_info_from_variant(dev.operators[i]);
          if (id == operator_id) return i;
        }
        return -1;
      },
      device_config);
}
static int find_session_operator_index_by_id(const pc::SessionConfiguration &s,
                                             const std::string &operator_id) {
  for (int i = 0; i < int(s.operators.size()); ++i) {
    auto [id, pn] =
        pc::operators::operator_info_from_variant(s.operators[size_t(i)]);
    if (id == operator_id) return i;
  }
  return -1;
}

// compares current adapter list to plugin operators by id & order
static bool operator_structure_changed(DeviceAdapter *adapter,
                                       pc::devices::DevicePlugin *plugin) {
  const auto &current = adapter->operatorAdapters();
  const auto &target = plugin->operators;
  if (int(target.size()) != current.size()) return true;
  for (int i = 0; i < current.size(); ++i) {
    auto *op = current[i];
    if (!op || !op->plugin() || !target[i]) return true;
    auto [a_id, _] = pc::operators::operator_info_from_variant(
        op->plugin()->config_variant());
    auto [b_id, __] =
        pc::operators::operator_info_from_variant(target[i]->config_variant());
    if (a_id != b_id) return true;
  }
  return false;
}
// session analogue of operator_structure_changed
static bool
session_operator_structure_changed(const QList<OperatorAdapter *> &current,
                                   pc::Session *session) {
  if (!session) return !current.isEmpty();
  const auto &target = session->operators;
  if (int(target.size()) != current.size()) return true;
  for (int i = 0; i < current.size(); ++i) {
    auto *op = current[i];
    if (!op || !op->plugin() || !target[size_t(i)]) return true;
    auto [a_id, _] = pc::operators::operator_info_from_variant(
        op->plugin()->config_variant());
    auto [b_id, __] = pc::operators::operator_info_from_variant(
        target[size_t(i)]->config_variant());
    if (a_id != b_id) return true;
  }
  return false;
}

// -------- undo commands --------

class SetSessionConfigCommand final : public QUndoCommand {
public:
  using ApplyFn = std::function<void(pc::WorkspaceConfiguration)>;

  SetSessionConfigCommand(QString session_id,
                          SessionConfiguration before_config,
                          SessionConfiguration after_config,
                          QString command_text,
                          pc::WorkspaceConfiguration base_config_snapshot,
                          ApplyFn apply_fn)
      : QUndoCommand(std::move(command_text)),
        _session_id(std::move(session_id)), _before(std::move(before_config)),
        _after(std::move(after_config)),
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

  void apply(const SessionConfiguration &config_value) {
    auto new_config = _base_snapshot;

    const int idx =
        find_session_index_by_id(new_config, _session_id.toStdString());
    if (idx < 0 || idx >= int(new_config.sessions.size())) {
      pc::logger()->warn("SetSessionConfigCommand: session id not found '{}'",
                         _session_id.toStdString());
      return;
    }

    new_config.sessions[size_t(idx)] = config_value;

    if (_apply_fn) _apply_fn(std::move(new_config));
  }
};

class SetDeviceConfigCommand final : public QUndoCommand {
public:
  using ApplyFn = std::function<void(pc::WorkspaceConfiguration)>;

  SetDeviceConfigCommand(QString device_id,
                         pc::devices::DeviceConfigurationVariant before_config,
                         pc::devices::DeviceConfigurationVariant after_config,
                         QString command_text,
                         pc::WorkspaceConfiguration base_config_snapshot,
                         ApplyFn apply_fn)
      : QUndoCommand(std::move(command_text)), _device_id(std::move(device_id)),
        _before(std::move(before_config)), _after(std::move(after_config)),
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

  void apply(const pc::devices::DeviceConfigurationVariant &config_value) {
    auto new_config = _base_snapshot;
    const int idx =
        find_device_index_by_id(new_config, _device_id.toStdString());
    if (idx < 0 || idx >= int(new_config.devices.size())) {
      pc::logger()->warn("SetDeviceConfigCommand: device id not found '{}'",
                         _device_id.toStdString());
      return;
    }
    new_config.devices[size_t(idx)] = config_value;
    if (_apply_fn) _apply_fn(std::move(new_config));
  }
};

class SetOperatorConfigCommand final : public QUndoCommand {
public:
  using ApplyFn = std::function<void(pc::WorkspaceConfiguration)>;
  SetOperatorConfigCommand(
      QString device_id, QString operator_id,
      pc::operators::OperatorConfigurationVariant before_config,
      pc::operators::OperatorConfigurationVariant after_config,
      QString command_text, pc::WorkspaceConfiguration base_config_snapshot,
      ApplyFn apply_fn)
      : QUndoCommand(std::move(command_text)), _device_id(std::move(device_id)),
        _operator_id(std::move(operator_id)), _before(std::move(before_config)),
        _after(std::move(after_config)),
        _base_snapshot(std::move(base_config_snapshot)),
        _apply_fn(std::move(apply_fn)) {}
  void undo() override { apply(_before); }
  void redo() override { apply(_after); }

private:
  QString _device_id;
  QString _operator_id;
  pc::operators::OperatorConfigurationVariant _before;
  pc::operators::OperatorConfigurationVariant _after;
  pc::WorkspaceConfiguration _base_snapshot;
  ApplyFn _apply_fn;
  void apply(const pc::operators::OperatorConfigurationVariant &config_value) {
    auto new_config = _base_snapshot;
    const int dev_idx =
        find_device_index_by_id(new_config, _device_id.toStdString());
    if (dev_idx < 0 || dev_idx >= int(new_config.devices.size())) return;
    const int op_idx = find_operator_index_by_id(
        new_config.devices[size_t(dev_idx)], _operator_id.toStdString());
    if (op_idx < 0) return;
    std::visit([&](auto &dev) { dev.operators[size_t(op_idx)] = config_value; },
               new_config.devices[size_t(dev_idx)]);
    if (_apply_fn) _apply_fn(std::move(new_config));
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
    pc::WorkspaceConfiguration new_config, RebuildScope scope) {
  const bool sync_devices =
      (scope == RebuildScope::Devices || scope == RebuildScope::All);
  _workspace.apply_new_config(std::move(new_config), sync_devices);
  switch (scope) {
  case RebuildScope::None: {
    break;
  }
  case RebuildScope::Sessions: {
    syncSessionAdapters();
    break;
  }
  case RebuildScope::Devices: {
    syncDeviceAdapters();
    break;
  }
  case RebuildScope::All: {
    syncAdapters();
    break;
  }
  }
}
// ----------------- WorkspaceModel -----------------
WorkspaceModel::WorkspaceModel(pc::Workspace *workspace, QObject *parent)

    : QObject(parent), _workspace(*workspace) {
  if (workspace->auto_loaded_config) {
    const auto path = AppSettings::instance()->lastWorkspacePath();
    setSaveFileUrl(QUrl::fromLocalFile(path));
  }

  _recorderModel = new RecorderModel(workspace->session_recorder.get(), this);
  emit recorderChanged();
}

void WorkspaceModel::close() {
  QCoreApplication::quit();
}

void WorkspaceModel::loadFromFile(const QUrl &file) {
  const QString local_path = file.toLocalFile();
  if (local_path.isEmpty()) return;

  std::jthread([&, local_path]() mutable {
    pc::WorkspaceConfiguration loaded_config;
    pc::load_workspace_from_file(loaded_config, local_path.toStdString());
    QMetaObject::invokeMethod(
        this,
        [this, file, local_path, config = std::move(loaded_config)]() mutable {
          setSaveFileUrl(file);
          applyWorkspaceConfigAndRebuild(std::move(config));
          AppSettings::instance()->setlastWorkspacePath(local_path);
        },
        Qt::QueuedConnection);

    std::filesystem::path session_file_path{local_path.toStdString()};
    auto layout_file_path = session_file_path;
    layout_file_path.replace_filename(session_file_path.stem().string() +
                                      "_layout.json");

    if (std::filesystem::exists(layout_file_path)) {
      QMetaObject::invokeMethod(
          this,
          [fp = layout_file_path.string(), this] {
            pc::ui::LayoutSaver saver(this);
            saver.load_file(fp.c_str());
          },
          Qt::QueuedConnection);
    }
  }).detach();
}

void WorkspaceModel::newWorkspace() {
  QMetaObject::invokeMethod(
      this,
      [this]() {
        setSaveFileUrl({});
        applyWorkspaceConfigAndRebuild(
            {.sessions = {
                 {.id = pc::uuid::word(),
                  .label = "session_1",
                  .camera = CameraConfiguration{.id = pc::uuid::word()}}}});
        setSelectedDeviceIndex(-1);
        emit newWorkspaceLoaded();
      },
      Qt::QueuedConnection);
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
                update_last_session_path, this] {
    save_workspace_to_file(workspace_config, local_path.toStdString());
    if (update_last_session_path) {
      AppSettings::instance()->setlastWorkspacePath(local_path);
    }

    constexpr bool save_adjacent_layout_file = true;
    if (save_adjacent_layout_file) {
      std::filesystem::path session_file_path{local_path.toStdString()};
      auto layout_file_path = session_file_path;
      layout_file_path.replace_filename(session_file_path.stem().string() +
                                        "_layout.json");
      QMetaObject::invokeMethod(
          this,
          [this, save_path = layout_file_path.string()]() {
            pc::ui::LayoutSaver saver(this);
            const bool result = saver.save_file(save_path.c_str());
          },
          Qt::QueuedConnection);
    }
  }).detach();
}

void WorkspaceModel::setImageProvider(CameraImageProvider *provider) {
  if (_imageProvider == provider) return;
  _imageProvider = provider;
  if (_imageProvider) {
    syncAdapters();
  }
}
QList<QObject *> WorkspaceModel::sessionAdapters() const {
  return _sessionAdapters;
}

void WorkspaceModel::setSelectedSessionId(const QString &id) {
  if (_selectedSessionId == id) return;
  _selectedSessionId = id;
  emit selectedSessionChanged();
}

QObject *WorkspaceModel::selectedSessionAdapter() const {
  for (QObject *obj : _sessionAdapters) {
    auto *a = qobject_cast<SessionConfigurationAdapter *>(obj);
    if (a && a->id() == _selectedSessionId) return a;
  }
  // fall back to the first session when no (valid) selection is set
  return _sessionAdapters.isEmpty() ? nullptr : _sessionAdapters.first();
}

QList<OperatorAdapter *>
WorkspaceModel::selectedSessionOperatorAdapters() const {
  QString id = _selectedSessionId;
  if (id.isEmpty() || !_sessionOperatorAdapters.contains(id)) {
    if (!_sessionAdapters.isEmpty()) {
      auto *a =
          qobject_cast<SessionConfigurationAdapter *>(_sessionAdapters.first());
      if (a) id = a->id();
    }
  }
  return _sessionOperatorAdapters.value(id);
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
  QVariantList menu_entries;

  auto &existing_devices = _workspace.devices;

  const auto existing_device_ids =
      std::ranges::transform_view(existing_devices, [](auto &device) {
        return std::visit(
            [&](auto &device_config) -> std::string_view {
              return device_config.id;
            },
            device->config_variant());
      });

  for (auto id : existing_device_ids) {
    pc::logger()->debug("Existing id: {}", id);
  }

  for (auto plugin_name : _workspace.loaded_device_plugin_names) {

    if (plugin_name == "NullDevice") continue;

    QVariantMap entry;
    entry["plugin_name"] = QString::fromStdString(plugin_name);
    entry["kind"] = "plugin";
    menu_entries.push_back(std::move(entry));

    // any discovered network devices of this plugin type
    auto it = _workspace.discovery_plugins.find(plugin_name);
    if (it == _workspace.discovery_plugins.end() || !it->second) continue;
    const auto discovered_devices = it->second->discovered_devices();
    for (const auto &device : discovered_devices) {

      // skip devices we're already connected to
      if (std::ranges::any_of(existing_device_ids,
                              [&](auto id) { return id == device.id; })) {
        continue;
      }

      QVariantMap discovered_device_entry;

      discovered_device_entry["kind"] = "discovered";
      discovered_device_entry["plugin_name"] =
          QString::fromStdString(plugin_name);
      discovered_device_entry["ip"] = QString::fromStdString(device.ip);
      discovered_device_entry["id"] = QString::fromStdString(device.id);
      discovered_device_entry["label"] = QString::fromStdString(device.label);

      menu_entries.push_back(std::move(discovered_device_entry));
    }
  }

  return menu_entries;
}

void WorkspaceModel::setSelectedDeviceIndex(int index) {
  if (_selectedDeviceIndex == index) return;
  _selectedDeviceIndex = index;
  _workspace.config.selectedDeviceIndex = index;
  setSelectedOperatorAdapter(nullptr);
  emit selectedDeviceIndexChanged();
}

void WorkspaceModel::addNewDevice(const QString &plugin_name,
                                  const QString &target_ip,
                                  const QString &target_id) {
  // take a copy of current configuration to manipulate
  auto result_config = _workspace.config;
  // TODO this needs to be polymorphic runtime access
  if (plugin_name == OrbbecDeviceConfiguration::PluginName) {
    result_config.devices.push_back(OrbbecDeviceConfiguration{
        .id = target_id.isEmpty() ? pc::uuid::word() : target_id.toStdString(),
        .network = {.ip_address =
                        target_ip.isEmpty() ? "" : target_ip.toStdString()}});
  } else if (plugin_name == PlyDeviceConfiguration::PluginName) {
    result_config.devices.push_back(
        PlyDeviceConfiguration{.id = pc::uuid::word()});
  }
  applyWorkspaceConfigAndRebuild(std::move(result_config),
                                 RebuildScope::Devices);
  emit deviceAdded();
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

  auto new_config = _workspace.config;
  const int device_index =
      find_device_index_by_id(new_config, device_id_q.toStdString());
  if (device_index < 0 || device_index >= int(new_config.devices.size())) {
    pc::logger()->warn("deleteSelectedDevice: device id not found '{}'",
                       device_id_q.toStdString());
    return;
  }

  pc::logger()->trace("Deleting device id='{}' (index {})",
                      device_id_q.toStdString(), device_index);

  new_config.devices.erase(new_config.devices.begin() + device_index);
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

DeviceAdapter *WorkspaceModel::deviceAdapterAt(int index) const {
  if (index < 0 || index >= _deviceAdapters.size()) return nullptr;
  return qobject_cast<DeviceAdapter *>(_deviceAdapters[index]);
}

void WorkspaceModel::addOperatorToDevice(int deviceIndex,
                                         const QString &operatorPluginName) {
  auto *adapter = deviceAdapterAt(deviceIndex);
  if (!adapter) return;

  const QString device_id = adapterStableId(adapter);
  if (device_id.isEmpty()) return;

  auto new_config = _workspace.config;
  const int idx = find_device_index_by_id(new_config, device_id.toStdString());
  if (idx < 0 || idx >= int(new_config.devices.size())) return;

  const auto op_name = operatorPluginName.toStdString();

  // add the operator config variant to the device's operator list
  bool found_type = false;
  operators::for_each_operator_config_type([&]<typename OperatorConfigType>() {
    if (found_type) return;
    if (op_name == OperatorConfigType::PluginName) {
      std::visit(
          [&](auto &device_config) {
            device_config.operators.push_back(
                OperatorConfigType{.id = pc::uuid::word()});
          },
          new_config.devices[size_t(idx)]);
      found_type = true;
    }
  });

  if (!found_type) {
    pc::logger()->error("Unknown operator plugin name '{}'", op_name);
    return;
  }

  // sync workspace config
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::None);

  // push the updated device variant to the live plugin; sync_operators inside
  // update_config will instantiate the new operator
  if (deviceIndex < int(_workspace.devices.size()) &&
      _workspace.devices[deviceIndex]) {
    _workspace.devices[deviceIndex]->update_config(
        _workspace.config.devices[size_t(idx)]);
  }
  adapter->rebuildOperatorAdapters();
  attachOperatorConfigAdapters(adapter);
}

void WorkspaceModel::removeOperatorFromDevice(int deviceIndex,
                                              int operatorIndex) {
  auto *adapter = deviceAdapterAt(deviceIndex);
  if (!adapter) return;
  const QString device_id = adapterStableId(adapter);
  if (device_id.isEmpty()) return;
  auto new_config = _workspace.config;
  const int idx = find_device_index_by_id(new_config, device_id.toStdString());
  if (idx < 0 || idx >= int(new_config.devices.size())) return;
  std::visit(
      [operatorIndex](auto &device_config) {
        auto &ops = device_config.operators;
        if (operatorIndex >= 0 && operatorIndex < int(ops.size()))
          ops.erase(ops.begin() + operatorIndex);
      },
      new_config.devices[size_t(idx)]);
  // sync workspace config
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::None);
  if (deviceIndex < int(_workspace.devices.size()) &&
      _workspace.devices[deviceIndex]) {
    _workspace.devices[deviceIndex]->update_config(
        _workspace.config.devices[size_t(idx)]);
  }
  adapter->rebuildOperatorAdapters();
  attachOperatorConfigAdapters(adapter);
}

void WorkspaceModel::reorderOperatorOnDevice(int deviceIndex, int fromIndex,
                                             int toIndex) {
  if (fromIndex == toIndex) return;
  auto *adapter = deviceAdapterAt(deviceIndex);
  if (!adapter) return;
  const QString device_id = adapterStableId(adapter);
  if (device_id.isEmpty()) return;
  auto new_config = _workspace.config;
  const int idx = find_device_index_by_id(new_config, device_id.toStdString());
  if (idx < 0 || idx >= int(new_config.devices.size())) return;
  std::visit(
      [fromIndex, toIndex](auto &device_config) {
        auto &ops = device_config.operators;
        if (fromIndex < 0 || fromIndex >= int(ops.size())) return;
        if (toIndex < 0 || toIndex >= int(ops.size())) return;
        auto item = std::move(ops[fromIndex]);
        ops.erase(ops.begin() + fromIndex);
        ops.insert(ops.begin() + toIndex, std::move(item));
      },
      new_config.devices[size_t(idx)]);
  // sync workspace config
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::None);
  if (deviceIndex < int(_workspace.devices.size()) &&
      _workspace.devices[deviceIndex]) {
    _workspace.devices[deviceIndex]->update_config(
        _workspace.config.devices[size_t(idx)]);
  }
  adapter->rebuildOperatorAdapters();
  attachOperatorConfigAdapters(adapter);
}

void WorkspaceModel::addOperatorToSession(const QString &sessionId,
                                          const QString &operatorPluginName) {
  auto new_config = _workspace.config;
  const int idx = find_session_index_by_id(new_config, sessionId.toStdString());
  if (idx < 0 || idx >= int(new_config.sessions.size())) return;

  const auto op_name = operatorPluginName.toStdString();
  bool found_type = false;
  operators::for_each_operator_config_type([&]<typename OperatorConfigType>() {
    if (found_type) return;
    if (op_name == OperatorConfigType::PluginName) {
      new_config.sessions[size_t(idx)].operators.push_back(
          OperatorConfigType{.id = pc::uuid::word()});
      found_type = true;
    }
  });

  if (!found_type) {
    pc::logger()->error("Unknown operator plugin name '{}'", op_name);
    return;
  }

  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Sessions);
}

void WorkspaceModel::removeOperatorFromSession(const QString &sessionId,
                                               int operatorIndex) {
  auto new_config = _workspace.config;
  const int idx = find_session_index_by_id(new_config, sessionId.toStdString());
  if (idx < 0 || idx >= int(new_config.sessions.size())) return;
  auto &ops = new_config.sessions[size_t(idx)].operators;
  if (operatorIndex < 0 || operatorIndex >= int(ops.size())) return;
  ops.erase(ops.begin() + operatorIndex);
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Sessions);
}

void WorkspaceModel::reorderOperatorOnSession(const QString &sessionId,
                                              int fromIndex, int toIndex) {
  if (fromIndex == toIndex) return;
  auto new_config = _workspace.config;
  const int idx = find_session_index_by_id(new_config, sessionId.toStdString());
  if (idx < 0 || idx >= int(new_config.sessions.size())) return;
  auto &ops = new_config.sessions[size_t(idx)].operators;
  if (fromIndex < 0 || fromIndex >= int(ops.size())) return;
  if (toIndex < 0 || toIndex >= int(ops.size())) return;
  auto item = std::move(ops[size_t(fromIndex)]);
  ops.erase(ops.begin() + fromIndex);
  ops.insert(ops.begin() + toIndex, std::move(item));
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Sessions);
}

void WorkspaceModel::attachOperatorConfigAdapters(
    DeviceAdapter *deviceAdapter) {
  for (auto *opAdapter : deviceAdapter->operatorAdapters()) {
    auto *plugin = opAdapter->plugin();
    if (!plugin) continue;
    auto &config_variant = plugin->config_variant();
    ConfigAdapter *adapter = nullptr;
    std::visit(
        [opAdapter, &adapter](auto &config) {
          using ConfigType = std::decay_t<decltype(config)>;
          if constexpr (std::is_same_v<
                            ConfigType,
                            pc::operators::FringeRemovalConfiguration>) {
            // TODO ok this really isnt right...
            adapter = new pc::operators::FringeRemovalConfigurationAdapter(
                const_cast<pc::operators::FringeRemovalConfiguration &>(config),
                opAdapter);
            // adapter = new pc::operators::FringeRemovalConfigurationAdapter(
            //     config, opAdapter);
          }
          // TODO
          // Add branches here for future operator types, or generate a
          // operator_config_adapter_for_config_t trait like devices have.
        },
        config_variant);
    if (adapter) {
      opAdapter->setConfigAdapter(adapter);
      initOperatorAdapter(opAdapter, deviceAdapter);
    }
  }
}

void WorkspaceModel::initOperatorAdapter(OperatorAdapter *opAdapter,
                                         DeviceAdapter *deviceAdapter) {
  auto *innerAdapter = opAdapter->configAdapter();
  if (!innerAdapter) return;
  QObject::connect(
      innerAdapter, &ConfigAdapter::editRequested, this,
      [this, opAdapter, deviceAdapter](const QString &path,
                                       const QVariant &value) {
        // pc::logger()->debug("[op-edit] path='{}' value-typeName='{}' "
        //                     "opAdapter={} deviceAdapter={}",
        //                     path.toStdString(),
        //                     value.typeName() ? value.typeName() : "<null>",
        //                     fmt::ptr(static_cast<void *>(opAdapter)),
        //                     fmt::ptr(static_cast<void *>(deviceAdapter)));
        auto *configAdapter = opAdapter->configAdapter();
        if (!configAdapter) return;
        const QString device_id = adapterStableId(deviceAdapter);
        if (device_id.isEmpty()) return;
        auto *plugin = opAdapter->plugin();
        if (!plugin) return;
        auto [op_id_sv, op_pn] =
            pc::operators::operator_info_from_variant(plugin->config_variant());
        const QString operator_id =
            QString::fromStdString(std::string(op_id_sv));
        pc::operators::OperatorConfigurationVariant before;
        pc::WorkspaceConfiguration base_snapshot;
        {
          std::scoped_lock lock(_workspace.config_access);
          base_snapshot = _workspace.config;
          const int dev_idx = find_device_index_by_id(_workspace.config,
                                                      device_id.toStdString());
          if (dev_idx < 0) return;
          const int op_idx = find_operator_index_by_id(
              _workspace.config.devices[size_t(dev_idx)],
              operator_id.toStdString());
          if (op_idx < 0) return;
          std::visit(
              [&](const auto &dev) { before = dev.operators[size_t(op_idx)]; },
              _workspace.config.devices[size_t(dev_idx)]);
        }
        const bool changed = configAdapter->apply(path, value);
        if (!changed) return;
        pc::operators::OperatorConfigurationVariant after =
            plugin->config_variant();
        // Write the new operator config back into workspace config
        {
          std::scoped_lock lock(_workspace.config_access);
          const int dev_idx = find_device_index_by_id(_workspace.config,
                                                      device_id.toStdString());
          if (dev_idx < 0) {
            return;
          }
          const int op_idx = find_operator_index_by_id(
              _workspace.config.devices[size_t(dev_idx)],
              operator_id.toStdString());
          if (op_idx < 0) {
            return;
          }
          std::visit([&](auto &dev) { dev.operators[size_t(op_idx)] = after; },
                     _workspace.config.devices[size_t(dev_idx)]);
        }

        // TODO
        // sync the new operator config with plugin's versions... but this seems
        // like it should have happened automatically because of above ^ i think
        // we want workspace config to be canonical source of truth throughout
        // application somehow
        const int dev_idx =
            find_device_index_by_id(_workspace.config, device_id.toStdString());
        if (dev_idx >= 0 && dev_idx < int(_workspace.devices.size()) &&
            _workspace.devices[dev_idx]) {
          const int op_idx = find_operator_index_by_id(
              _workspace.devices[dev_idx]->config_variant(),
              operator_id.toStdString());
          if (op_idx >= 0) {
            std::visit(
                [&](auto &dev) { dev.operators[size_t(op_idx)] = after; },
                _workspace.devices[dev_idx]->config_variant());
          }
        }

        // configAdapter->notifyFieldChanged(path);

        QString command_text = QStringLiteral("Edit %1").arg(path);
        _undoStack->push(new SetOperatorConfigCommand(
            device_id, operator_id, std::move(before), std::move(after),
            std::move(command_text), std::move(base_snapshot),
            [this](pc::WorkspaceConfiguration config) {
              QMetaObject::invokeMethod(
                  this,
                  [this, config = std::move(config)]() mutable {
                    _workspace.apply_new_config(std::move(config), false);
                  },
                  Qt::QueuedConnection);
            }));
      });
}

void WorkspaceModel::attachSessionOperatorConfigAdapters(
    const QString &sessionId, const QList<OperatorAdapter *> &ops) {
  for (auto *opAdapter : ops) {
    auto *plugin = opAdapter->plugin();
    if (!plugin) continue;
    auto &config_variant = plugin->config_variant();
    ConfigAdapter *adapter = nullptr;
    std::visit(
        [opAdapter, &adapter](auto &config) {
          using ConfigType = std::decay_t<decltype(config)>;
          if constexpr (std::is_same_v<
                            ConfigType,
                            pc::operators::FringeRemovalConfiguration>) {
            adapter = new pc::operators::FringeRemovalConfigurationAdapter(
                const_cast<pc::operators::FringeRemovalConfiguration &>(config),
                opAdapter);
          }
          // TODO
          // Add branches here for future operator types, mirroring the device
          // attachOperatorConfigAdapters().
        },
        config_variant);
    if (adapter) {
      opAdapter->setConfigAdapter(adapter);
      initSessionOperatorAdapter(opAdapter, sessionId);
    }
  }
}

void WorkspaceModel::initSessionOperatorAdapter(OperatorAdapter *opAdapter,
                                                const QString &sessionId) {
  auto *innerAdapter = opAdapter->configAdapter();
  if (!innerAdapter) return;
  QObject::connect(
      innerAdapter, &ConfigAdapter::editRequested, this,
      [this, opAdapter, sessionId](const QString &path, const QVariant &value) {
        auto *configAdapter = opAdapter->configAdapter();
        if (!configAdapter) return;
        auto *plugin = opAdapter->plugin();
        if (!plugin) return;
        auto [op_id_sv, op_pn] =
            pc::operators::operator_info_from_variant(plugin->config_variant());
        const QString operator_id =
            QString::fromStdString(std::string(op_id_sv));

        SessionConfiguration before;
        SessionConfiguration after;
        pc::WorkspaceConfiguration base_snapshot;
        {
          std::scoped_lock lock(_workspace.config_access);
          base_snapshot = _workspace.config;
          const int s = find_session_index_by_id(_workspace.config,
                                                 sessionId.toStdString());
          if (s < 0) return;
          before = _workspace.config.sessions[size_t(s)];
        }

        const bool changed = configAdapter->apply(path, value);
        if (!changed) return;
        pc::operators::OperatorConfigurationVariant updated =
            plugin->config_variant();

        {
          std::scoped_lock lock(_workspace.config_access);
          const int s = find_session_index_by_id(_workspace.config,
                                                 sessionId.toStdString());
          if (s < 0) return;
          auto &session = _workspace.config.sessions[size_t(s)];
          const int o = find_session_operator_index_by_id(
              session, operator_id.toStdString());
          if (o < 0) return;
          session.operators[size_t(o)] = std::move(updated);
          after = session;
        }

        QString command_text = QStringLiteral("Edit %1").arg(path);
        _undoStack->push(new SetSessionConfigCommand(
            sessionId, std::move(before), std::move(after),
            std::move(command_text), std::move(base_snapshot),
            [this](pc::WorkspaceConfiguration config) {
              QMetaObject::invokeMethod(
                  this,
                  [this, config = std::move(config)]() mutable {
                    // update config + sessions, don't rebuild devices
                    _workspace.apply_new_config(std::move(config), false);
                    syncSessionAdapters();
                  },
                  Qt::QueuedConnection);
            }));
      });
}

void WorkspaceModel::rebuildSessionOperatorAdapters(
    const QString &sessionId, SessionConfigurationAdapter *adapter,
    pc::Session *session) {
  auto &list = _sessionOperatorAdapters[sessionId];
  qDeleteAll(list);
  list.clear();
  if (session) {
    for (auto &op : session->operators) {
      if (!op) continue;
      auto *opAdapter = new OperatorAdapter(op.get(), session, adapter);
      list.append(opAdapter);
    }
  }
  attachSessionOperatorConfigAdapters(sessionId, list);
}

void WorkspaceModel::syncSessionOperatorAdapters() {
  QSet<QString> live_ids;
  for (QObject *obj : _sessionAdapters) {
    auto *adapter = qobject_cast<SessionConfigurationAdapter *>(obj);
    if (!adapter) continue;
    const QString id = adapter->id();
    if (id.isEmpty()) continue;
    live_ids.insert(id);

    pc::Session *session = nullptr;
    {
      auto it = _workspace.sessions.find(id.toStdString());
      if (it != _workspace.sessions.end()) session = it->second.get();
    }

    const bool has_entry = _sessionOperatorAdapters.contains(id);
    const auto &current = _sessionOperatorAdapters[id];
    if (!has_entry || session_operator_structure_changed(current, session)) {
      rebuildSessionOperatorAdapters(id, adapter, session);
    }
  }

  // drop operator adapters for sessions that no longer exist; they were
  // parented to their (now-deleted) session adapter, so just drop references
  for (auto it = _sessionOperatorAdapters.begin();
       it != _sessionOperatorAdapters.end();) {
    if (!live_ids.contains(it.key())) {
      it = _sessionOperatorAdapters.erase(it);
    } else {
      ++it;
    }
  }

  emit selectedSessionChanged();
}

void WorkspaceModel::setSelectedOperatorAdapter(OperatorAdapter *adapter) {
  if (_selectedOperatorAdapter == adapter) return;
  if (_selectedOperatorAdapter) {
    disconnect(_selectedOperatorAdapter, &QObject::destroyed, this, nullptr);
  }
  _selectedOperatorAdapter = adapter;
  if (adapter) {
    connect(adapter, &QObject::destroyed, this, [this] {
      // tell QML when adapters are destroyed
      emit selectedOperatorAdapterChanged();
    });
  }
  emit selectedOperatorAdapterChanged();
}

QVariantMap generateConsoleEntryVariant(const LogEntry &entry) {
  QVariantMap item;
  switch (entry.level) {
  case spdlog::level::trace:
    item["logLevel"] = "trace";
    item["logLevelColor"] = "midlight";
    break;
  case spdlog::level::debug:
    item["logLevel"] = "debug";
    item["logLevelColor"] = "blue";
    break;
  case spdlog::level::info:
    item["logLevel"] = "info";
    item["logLevelColor"] = "text";
    break;
  case spdlog::level::warn:
    item["logLevel"] = "warning";
    item["logLevelColor"] = "yellow";
    break;
  case spdlog::level::err:
    item["logLevel"] = "error";
    item["logLevelColor"] = "red";
    break;
  case spdlog::level::critical:
    item["logLevel"] = "critical";
    item["logLevelColor"] = "red";
    break;
  default:
    item["logLevel"] = "";
    item["logLevelColor"] = "text";
    break;
  }
  item["message"] = QString::fromStdString(entry.message);
  return item;
}

QVariantList WorkspaceModel::consoleOverlayEntries() const {
  QVariantList entries;
  using namespace std::chrono_literals;
  for (auto log_entry : pc::logger_lines(6, 10s)) {
    entries.push_back(generateConsoleEntryVariant(log_entry));
  }
  return entries;
}

QVariantList WorkspaceModel::consoleHistoryEntries() const {
  QVariantList entries;
  using namespace std::chrono_literals;
  constexpr auto consoleHistoryMax = 200;
  for (auto log_entry : pc::logger_lines(consoleHistoryMax)) {
    entries.push_back(generateConsoleEntryVariant(log_entry));
  }
  return entries;
}
// TODO whats the change here?
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
                       std::scoped_lock lock(_workspace.config_access);
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
                         [this](pc::WorkspaceConfiguration config) {
                           QMetaObject::invokeMethod(
                               this,
                               [this, config = std::move(config)]() mutable {
                                 // only update config, don't rebuild devices
                                 _workspace.apply_new_config(std::move(config),
                                                             false);
                                 syncSessionAdapters();
                               },
                               Qt::QueuedConnection);
                         }));
                   });
}
template <typename AdapterT>
void WorkspaceModel::initDeviceAdapter(AdapterT *adapter,
                                       pc::devices::DevicePlugin *plugin) {
  const auto &config =
      std::get<typename AdapterT::config_type>(plugin->config_variant());
  pc::logger()->trace("Initialising device adapter '{}' for plugin '{}'",
                      config.id, adapter->displayName().toStdString());
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
          std::scoped_lock lock(_workspace.config_access);
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
        after = p->config_variant();
        {
          std::scoped_lock lock(_workspace.config_access);
          const int idx = find_device_index_by_id(_workspace.config,
                                                  device_id_q.toStdString());
          if (idx < 0) return;
          _workspace.config.devices[size_t(idx)] = after;
        }
        command_text = QStringLiteral("Edit %1").arg(path);
        _undoStack->push(new SetDeviceConfigCommand(
            device_id_q, std::move(before), std::move(after),
            std::move(command_text), std::move(base_snapshot),
            [this](pc::WorkspaceConfiguration config) {
              QMetaObject::invokeMethod(
                  this,
                  [this, config = std::move(config)]() mutable {
                    applyWorkspaceConfigAndRebuild(std::move(config),
                                                   RebuildScope::Devices);
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
        auto *adapter =
            new AdapterType(device_config, plugin, _imageProvider, this);
        initDeviceAdapter(adapter, plugin);
        result = adapter;
      },
      config_variant);
  return result;
}

void WorkspaceModel::syncAdapters() {
  syncSessionAdapters();
  syncDeviceAdapters();
}

void WorkspaceModel::syncSessionAdapters() {
  pc::logger()->trace("Syncing session adapters");
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

    for (auto &session_config : _workspace.config.sessions) {
      const QString id = QString::fromStdString(session_config.id);
      if (id.isEmpty()) continue;
      const pc::SessionConfiguration *current_ptr = &session_config;
      SessionConfigurationAdapter *adapter = existing_by_id.take(id);
      const bool ptr_matches = _sessionConfigPtrById.contains(id) &&
                               (_sessionConfigPtrById.value(id) == current_ptr);
      if (adapter && !ptr_matches) {
        // Underlying SessionConfiguration moved (vector realloc / reorder).
        // Old adapter holds a dangling reference: must recreate. Its operator
        // adapters are children and will be deleted with it; drop our refs.
        adapter->deleteLater();
        adapter = nullptr;
        _sessionOperatorAdapters.remove(id);
      }
      if (adapter) {
        (void)adapter->setConfig(session_config);
      } else {
        auto *new_adapter =
            new SessionConfigurationAdapter(session_config, this);
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

  // Rebuild per-session operator adapters from each session's live
  // OperatorPlugin instances.
  syncSessionOperatorAdapters();

  pc::logger()->trace("Finished syncing sessionAdapters");
}

void WorkspaceModel::syncDeviceAdapters() {
  pc::logger()->trace("Syncing device adapters");
  pc::logger()->trace("Workspace device count: {}", _workspace.devices.size());
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
  for (auto &device_plugin : _workspace.devices) {
    auto *plugin = device_plugin.get();
    DeviceAdapter *adapter = nullptr;
    using DeviceConfigVariantRef =
        std::reference_wrapper<devices::DeviceConfigurationVariant>;
    std::optional<DeviceConfigVariantRef> device_config;
    if (plugin) device_config = plugin->config_variant();
    if (device_config.has_value()) {
      auto config = device_config.value();
      auto [device_id, plugin_name] = devices::device_info_from_variant(config);
      const QString id(device_id.data());
      if (!id.isEmpty()) {
        // try an existing adapter
        adapter = existing_by_id.take(id);
        if (adapter) {
          bool sync_success = false;
          try {
            sync_success = adapter->setConfig(config);
          } catch (...) {
            pc::logger()->error("Exception thrown setting config for '{}' '{}'",
                                plugin_name, device_id);
          }
          if (!sync_success) {
            pc::logger()->trace(
                "Existing device adapter with id '{}' failed to set new "
                "configuration. Recreating...",
                device_id);
            adapter->deleteLater();
            adapter = nullptr;
          }
        }
      }
      if (!adapter) {
        adapter = makeDeviceAdapterForPlugin(plugin, config);
      }
      if (adapter) {
        auto adapterPtr = QPointer<DeviceAdapter>(adapter);
        // hook up the qt signal so that when the plugin updates its
        // pointcloud, quick3d can react to this event and update geometry
        plugin->set_point_cloud_updated_callback([adapterPtr]() {
          QMetaObject::invokeMethod(adapterPtr.data(), [adapterPtr]() {
            if (!adapterPtr) {
              pc::logger()->error("Invalid point cloud plugin ptr");
              return;
            }
            adapterPtr->notifyPointCloudUpdated();
            // also update operator projections attached to this device we
            // might want to visualise in the UI
            adapterPtr->syncCameraFrames();
          });
        });

        // TODO what is this for
        plugin->set_status_callback(
            [adapterPtr](pc::devices::DeviceStatus status) {
              if (!adapterPtr) return;
              QMetaObject::invokeMethod(
                  adapterPtr.data(),
                  [adapterPtr, status]() {
                    if (!adapterPtr) return;
                    adapterPtr->setStatusFromCore(status);
                  },
                  Qt::QueuedConnection);
            });
        if (adapter && operator_structure_changed(adapter, plugin)) {
          adapter->rebuildOperatorAdapters();
          attachOperatorConfigAdapters(adapter);
        }
      }
    }
    // if we managed to construct or get an existing valid apater,
    // add this to our new devices list
    if (adapter)
      new_ordered_devices.append(adapter);
    else {
      pc::logger()->error("no adapter");
      // we we didn't finish this loop with a valid adapter, we need one in
      // a null state adapter = makeDeviceAdapterForPlugin(nullptr);
    }
  }

  for (auto it = existing_by_id.begin(); it != existing_by_id.end(); ++it) {
    if (it.value()) it.value()->deleteLater();
  }
  _deviceAdapters = new_ordered_devices;
  int selectedDeviceIndex = _workspace.config.selectedDeviceIndex.value();
  if (selectedDeviceIndex >= _deviceAdapters.size() || selectedDeviceIndex < 0)
    selectedDeviceIndex = 0;
  setSelectedDeviceIndex(selectedDeviceIndex);
  emit deviceAdaptersChanged();
  emit deviceVariantNamesChanged();
  emit addDeviceMenuEntriesChanged();
  pc::logger()->trace("Workspace device count: {}", _workspace.devices.size());
}

void WorkspaceModel::syncConsole() {
  static QVariantList last_entries(6);
  auto current_entries = consoleOverlayEntries();
  if (last_entries != current_entries) {
    emit consoleOverlayEntriesChanged();
    emit consoleHistoryEntriesChanged();
    last_entries = current_entries;
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

} // namespace pc::ui