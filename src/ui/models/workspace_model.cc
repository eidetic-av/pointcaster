#include "workspace_model.h"

#include "app_settings/app_settings.h"
#include "camera_image_provider.h"
#include "layout_saver.h"
#include "models/device_adapter.h"
#include "models/session_recorder_model.h"
#include "models/settings_page_registry.h"
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
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <plugins/devices/device_group_config.h>
#include <plugins/devices/device_tree.h>
#include <plugins/devices/device_variants.h>
#include <ranges>
#include <session/session.h>
#include <session/session_config.h>
#include <session/session_config_adapter.gen.h>
#include <set>
#include <spdlog/common.h>
#include <string>
#include <string_view>
#include <thread>
#include <ui/layout_saver.h>
#include <unordered_map>
#include <variant>
#include <workspace/workspace.h>

#include <plugins/devices/device_group_config_adapter.gen.h>
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

static int
find_device_group_index_by_id(const pc::WorkspaceConfiguration &config,
                              const std::string &group_id) {
  for (int i = 0; i < int(config.device_groups.size()); ++i) {
    if (config.device_groups[size_t(i)].id == group_id) return i;
  }
  return -1;
}

static int find_operator_index_by_id(
    const pc::devices::DeviceConfigurationVariant &device_config,
    const std::string &operator_id) {
  return std::visit(
      [&](const auto &config) -> int {
        if constexpr (requires { config.operators; }) {
          for (int i = 0; i < int(config.operators.size()); ++i) {
            auto [id, pn] =
                pc::operators::operator_info_from_variant(config.operators[i]);
            if (id == operator_id) return i;
          }
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

// the config registry prefix a device or group's fields live under
std::string node_path_prefix(const pc::WorkspaceConfiguration &config,
                             const std::string &node_id) {
  return "device/" + pc::devices::device_address(config, node_id);
}

class SetDeviceGroupConfigCommand final : public QUndoCommand {
public:
  using ApplyFn = std::function<void(pc::WorkspaceConfiguration)>;

  SetDeviceGroupConfigCommand(QString group_id,
                              pc::devices::DeviceGroupConfiguration before,
                              pc::devices::DeviceGroupConfiguration after,
                              QString command_text,
                              pc::WorkspaceConfiguration base_snapshot,
                              ApplyFn apply_fn)
      : QUndoCommand(std::move(command_text)), _group_id(std::move(group_id)),
        _before(std::move(before)), _after(std::move(after)),
        _base_snapshot(std::move(base_snapshot)),
        _apply_fn(std::move(apply_fn)) {}

  void undo() override { apply(_before); }
  void redo() override { apply(_after); }

private:
  QString _group_id;
  pc::devices::DeviceGroupConfiguration _before;
  pc::devices::DeviceGroupConfiguration _after;
  pc::WorkspaceConfiguration _base_snapshot;
  ApplyFn _apply_fn;

  void apply(const pc::devices::DeviceGroupConfiguration &value) {
    auto new_config = _base_snapshot;
    const int gi =
        find_device_group_index_by_id(new_config, _group_id.toStdString());
    if (gi < 0 || gi >= int(new_config.device_groups.size())) return;
    new_config.device_groups[size_t(gi)] = value;
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
    const int device_index =
        find_device_index_by_id(new_config, _device_id.toStdString());
    if (device_index < 0 || device_index >= int(new_config.devices.size()))
      return;
    const int op_idx = find_operator_index_by_id(
        new_config.devices[size_t(device_index)], _operator_id.toStdString());
    if (op_idx < 0) return;
    std::visit(
        [&](auto &device_config) {
          if constexpr (requires { device_config.operators; }) {
            device_config.operators[size_t(op_idx)] = config_value;
          }
        },
        new_config.devices[size_t(device_index)]);
    if (_apply_fn) _apply_fn(std::move(new_config));
  }
};

class SetPointStreamerConfigCommand final : public QUndoCommand {
public:
  using ApplyFn = std::function<void(pc::WorkspaceConfiguration)>;
  SetPointStreamerConfigCommand(
      pc::networking::PointStreamerConfiguration before,
      pc::networking::PointStreamerConfiguration after, QString command_text,
      pc::WorkspaceConfiguration base_snapshot, ApplyFn apply_fn)
      : QUndoCommand(std::move(command_text)), _before(std::move(before)),
        _after(std::move(after)), _base_snapshot(std::move(base_snapshot)),
        _apply_fn(std::move(apply_fn)) {}
  void undo() override { apply(_before); }
  void redo() override { apply(_after); }

private:
  pc::networking::PointStreamerConfiguration _before;
  pc::networking::PointStreamerConfiguration _after;
  pc::WorkspaceConfiguration _base_snapshot;
  ApplyFn _apply_fn;
  void apply(const pc::networking::PointStreamerConfiguration &value) {
    auto new_config = _base_snapshot;
    new_config.point_streamer.set(value);
    if (_apply_fn) _apply_fn(std::move(new_config));
  }
};

struct SessionUpdateGate {
  std::atomic<bool> pending{false};
  QPointer<SessionAdapter> adapter;
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
  if (_pointStreamerAdapter) {
    _pointStreamerAdapter->setConfig(
        pc::ConfigurationVariant{_workspace.config.point_streamer.value()});
  }
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
  emit publishPathsChanged();
  emit pushPathsChanged();
  _streamChannelModel->refresh();
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

  _pointStreamerAdapter = new pc::networking::PointStreamerConfigurationAdapter(
      _workspace.config.point_streamer.value(), this);
  initPointStreamerAdapter();

  _streamChannelModel = new StreamChannelListModel(workspace, this);
  _streamChannelModel->refresh();
}

void WorkspaceModel::close() {
  QCoreApplication::quit();
}

void WorkspaceModel::registerPluginSettingsPages() {
  for (const auto &[plugin_name, plugin] : _workspace.discovery_plugins) {
    if (!plugin) continue;
    for (const auto &page : plugin->settings_pages()) {
      pc::logger()->trace("Registering settings page '{}' from plugin '{}'",
                          page.key, plugin_name);
      SettingsPageRegistry::instance()->addSection("Devices");
      SettingsPageRegistry::instance()->addPage(
          QString::fromStdString(page.key), QString::fromStdString(page.title),
          QUrl::fromLocalFile(QString::fromStdString(page.qml_file_path)));
    }
  }
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

namespace {
QStringList to_string_list(const std::set<std::string> &paths) {
  QStringList list;
  list.reserve(int(paths.size()));
  for (const auto &path : paths) list.append(QString::fromStdString(path));
  return list;
}
} // namespace

QStringList WorkspaceModel::publishPaths() const {
  std::scoped_lock lock(_workspace.config_access);
  return to_string_list(_workspace.config.publish_paths.value());
}

QStringList WorkspaceModel::pushPaths() const {
  std::scoped_lock lock(_workspace.config_access);
  return to_string_list(_workspace.config.push_paths.value());
}

// a field's entry is its adapter's configPath joined with the field path.
// RebuildScope::None because no adapter or device is affected by the sets
// changing, only the rows drawing their published state
void WorkspaceModel::addPublishPath(const QString &path) {
  auto new_config = _workspace.config;
  new_config.publish_paths.value().insert(path.toStdString());
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::None);
}

void WorkspaceModel::removePublishPath(const QString &path) {
  auto new_config = _workspace.config;
  new_config.publish_paths.value().erase(path.toStdString());
  // a path cannot be pushed if it is not also published
  new_config.push_paths.value().erase(path.toStdString());
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::None);
}

void WorkspaceModel::addPushPath(const QString &path) {
  auto new_config = _workspace.config;
  new_config.push_paths.value().insert(path.toStdString());
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::None);
}

void WorkspaceModel::removePushPath(const QString &path) {
  auto new_config = _workspace.config;
  new_config.push_paths.value().erase(path.toStdString());
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::None);
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
        if constexpr (requires { DeviceConfigType::PluginName; }) {
          names.push_back(QString::fromStdString(DeviceConfigType::PluginName));
        }
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
      discovered_device_entry["type_label"] =
          QString::fromStdString(device.type_label);

      menu_entries.push_back(std::move(discovered_device_entry));
    }
  }

  return menu_entries;
}

void WorkspaceModel::setSelectedStreamChannelIndex(int index) {
  const int clamped =
      std::clamp(index, -1, _streamChannelModel->rowCount() - 1);
  if (_selectedStreamChannelIndex == clamped) return;
  _selectedStreamChannelIndex = clamped;
  emit selectedStreamChannelIndexChanged();
}

void WorkspaceModel::setSelectedDeviceIndex(int index) {
  const QString node_id = adapterStableId(deviceAdapterAt(index));
  const QString node_kind =
      node_id.isEmpty() ? QString() : QStringLiteral("device");

  if (_selectedDeviceIndex == index && _selectedNodeId == node_id &&
      _selectedNodeKind == node_kind)
    return;

  _selectedDeviceIndex = index;
  _workspace.config.selectedDeviceIndex = index;
  setSelectedOperatorAdapter(nullptr);

  _selectedNodeId = node_id;
  _selectedNodeKind = node_kind;

  emit selectedDeviceIndexChanged();
  emit selectedNodeChanged();

  rebuildSelectedGroupAdapter();
}

void WorkspaceModel::selectNode(const QString &node_id) {
  const std::string id = node_id.toStdString();
  if (id.empty()) return;

  // group node: leave the device index alone, switch the editor by kind
  if (find_device_group_index_by_id(_workspace.config, id) >= 0) {
    if (_selectedNodeKind == QStringLiteral("group") &&
        _selectedNodeId == node_id)
      return;
    setSelectedOperatorAdapter(nullptr);
    _selectedNodeId = node_id;
    _selectedNodeKind = QStringLiteral("group");
    rebuildSelectedGroupAdapter();
    emit selectedNodeChanged();
    return;
  }

  // device node: route through the existing device-selection path
  const int device_index = find_device_index_by_id(_workspace.config, id);
  if (device_index >= 0) setSelectedDeviceIndex(device_index);
}

QMatrix4x4
WorkspaceModel::nodeAncestorWorldMatrix(const QString &node_id) const {
  pc::float4x4 world;
  {
    std::scoped_lock lock(_workspace.config_access);
    world = pc::devices::effective_world_transform(_workspace.config,
                                                   node_id.toStdString());
  }
  std::array<float, 16> v = world.values;
  v[3] *= 0.001f;
  v[7] *= 0.001f;
  v[11] *= 0.001f;
  return QMatrix4x4(v.data());
}

void WorkspaceModel::addNewDevice(const QString &plugin_name,
                                  const QString &target_ip,
                                  const QString &target_id,
                                  const QString &target_type_label) {
  // take a copy of current configuration to manipulate
  auto result_config = _workspace.config;
  // TODO this needs to be polymorphic runtime access
  if (plugin_name == OrbbecDeviceConfiguration::PluginName) {

    OrbbecDeviceConfiguration::SensorConfigurationVariant sensor_config;
    if (QString::compare(target_type_label, "rgbd") == 0) {
      sensor_config = OrbbecDeviceConfiguration::RgbdSensorConfiguration{};
    } else if (QString::compare(target_type_label, "lidar") == 0) {
      sensor_config = OrbbecDeviceConfiguration::LidarSensorConfiguration{};
    } else {
      pc::logger()->error("Invalid Orbbec device type.");
      return;
    }

    OrbbecDeviceConfiguration orbbec_config{
        .id = target_id.isEmpty() ? pc::uuid::word() : target_id.toStdString(),
        .sensor = std::move(sensor_config)};

    if (!target_ip.isEmpty()) {
      orbbec_config.network.set({.ip_address = target_ip.toStdString()});
    }
    result_config.devices.push_back(std::move(orbbec_config));
  } else if (plugin_name == PlyDeviceConfiguration::PluginName) {
    result_config.devices.push_back(
        PlyDeviceConfiguration{.id = pc::uuid::word()});
  }
  applyWorkspaceConfigAndRebuild(std::move(result_config),
                                 RebuildScope::Devices);
  emit deviceAdded();
}

void WorkspaceModel::deleteDevice(const QString &device_id) {
  if (_deviceAdapters.isEmpty()) return;

  auto new_config = _workspace.config;

  const auto device_id_str = device_id.toStdString();
  const int device_index = find_device_index_by_id(new_config, device_id_str);

  if (device_index < 0 || device_index >= int(new_config.devices.size())) {
    pc::logger()->warn("deleteSelectedDevice: device id not found '{}'",
                       device_id_str);
    return;
  }

  pc::logger()->trace("Deleting device id='{}' (index {})", device_id_str,
                      device_index);

  new_config.devices.erase(new_config.devices.begin() + device_index);
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
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

void WorkspaceModel::duplicateDeviceNode(const QString &node_id) {
  const std::string id = node_id.toStdString();
  if (id.empty()) return;

  // TODO the implementation here should really be inside workspace's own
  // source, not the Qt model for workspace!!

  auto new_config = _workspace.config;
  const int group_index = find_device_group_index_by_id(new_config, id);

  if (group_index >= 0) {
    // build old_id -> new_id map for the entire subtree (group + descendants)
    std::unordered_map<std::string, std::string> id_map;

    std::function<void(const std::string &)> collect_subtree_ids;
    collect_subtree_ids = [&](const std::string &group_id) {
      id_map[group_id] = pc::uuid::word();

      for (auto &device_group : _workspace.config.device_groups) {
        if (device_group.parent_id.value() == group_id)
          collect_subtree_ids(device_group.id);
      }
      for (auto &device_config : _workspace.config.devices) {
        std::visit(
            [&](const auto &device_config) {
              if constexpr (requires {
                              device_config.id;
                              device_config.parent_id;
                            }) {
                if (device_config.parent_id.value() == group_id)
                  id_map[device_config.id] = pc::uuid::word();
              }
            },
            device_config);
      }
    };

    collect_subtree_ids(id);

    // copy groups in the subtree with remapped ids
    for (const auto &group_config : _workspace.config.device_groups) {
      if (!id_map.count(group_config.id)) continue;
      auto group_copy = group_config;
      group_copy.id = id_map.at(group_config.id);
      const std::string old_parent = group_config.parent_id.value();
      if (id_map.count(old_parent))
        group_copy.parent_id.set(id_map.at(old_parent));
      new_config.device_groups.push_back(std::move(group_copy));
    }

    // copy devices in the subtree with remapped ids and fresh operator ids
    for (auto device_config_variant : _workspace.config.devices) {
      bool in_subtree = false;
      std::visit(
          [&](const auto &device_config) {
            in_subtree = id_map.count(device_config.id) > 0;
          },
          device_config_variant);
      if (!in_subtree) continue;
      std::visit(
          [&](auto &device_config) {
            device_config.id = id_map.at(device_config.id);
            const std::string old_parent = device_config.parent_id.value();
            if (id_map.count(old_parent))
              device_config.parent_id.set(id_map.at(old_parent));
            if constexpr (requires { device_config.operators; }) {
              for (auto &operator_config_variant : device_config.operators) {
                std::visit(
                    [](auto &operator_config) {
                      operator_config.id = pc::uuid::word();
                    },
                    operator_config_variant);
              }
            }
          },
          device_config_variant);
      new_config.devices.push_back(std::move(device_config_variant));
    }
  } else {
    // single device duplicate
    const int device_index = find_device_index_by_id(new_config, id);
    if (device_index < 0) return;
    auto device_copy = new_config.devices[size_t(device_index)];
    std::visit(
        [](auto &device_config) {
          device_config.id = pc::uuid::word();
          if constexpr (requires { device_config.operators; }) {
            for (auto &operator_config_variant : device_config.operators) {
              std::visit(
                  [](auto &operator_config) {
                    operator_config.id = pc::uuid::word();
                  },
                  operator_config_variant);
            }
          }
        },
        device_copy);
    new_config.devices.push_back(std::move(device_copy));
  }

  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

DeviceAdapter *WorkspaceModel::deviceAdapterAt(int index) const {
  if (index < 0 || index >= _deviceAdapters.size()) return nullptr;
  return qobject_cast<DeviceAdapter *>(_deviceAdapters[index]);
}

QObject *WorkspaceModel::selectedDeviceGroupAdapter() const {
  return _selectedDeviceGroupAdapter.data();
}

void WorkspaceModel::rebuildSelectedGroupAdapter() {
  if (_selectedDeviceGroupAdapter) {
    _selectedDeviceGroupAdapter->deleteLater();
    _selectedDeviceGroupAdapter = nullptr;
  }

  if (_selectedNodeKind == QStringLiteral("group")) {
    const int gi = find_device_group_index_by_id(_workspace.config,
                                                 _selectedNodeId.toStdString());
    if (gi >= 0) {
      // groups have no plugin and no image provider, so pass nullptr for both.
      auto *adapter = new pc::devices::DeviceGroupConfigurationAdapter(
          _workspace.config.device_groups[size_t(gi)], nullptr, nullptr, this);
      adapter->setConfigPath(QString::fromStdString(
          node_path_prefix(_workspace.config, _selectedNodeId.toStdString())));
      initGroupAdapter(adapter);
      _selectedDeviceGroupAdapter = adapter;
    }
  }

  emit selectedDeviceGroupAdapterChanged();
}

void WorkspaceModel::initGroupAdapter(
    pc::devices::DeviceGroupConfigurationAdapter *adapter) {
  if (!adapter) return;
  QObject::connect(
      adapter, &ConfigAdapter::editRequested, this,
      [this, adapter](const QString &path, const QVariant &value) {
        const QString group_id_q = adapter->id();
        if (group_id_q.isEmpty()) return;

        pc::devices::DeviceGroupConfiguration before;
        pc::devices::DeviceGroupConfiguration after;
        pc::WorkspaceConfiguration base_snapshot;
        {
          std::scoped_lock lock(_workspace.config_access);
          base_snapshot = _workspace.config;
          const int group_index = find_device_group_index_by_id(
              _workspace.config, group_id_q.toStdString());
          if (group_index < 0) return;
          before = _workspace.config.device_groups[size_t(group_index)];
        }

        // apply mutates the live group config through the adapter's reference.
        const bool changed = adapter->apply(path, value);
        if (!changed) return;

        {
          std::scoped_lock lock(_workspace.config_access);
          const int group_index = find_device_group_index_by_id(
              _workspace.config, group_id_q.toStdString());
          if (group_index < 0) return;
          after = _workspace.config.device_groups[size_t(group_index)];
        }

        const std::string group_id = group_id_q.toStdString();
        QString command_text = QStringLiteral("Edit %1").arg(path);
        _undoStack->push(new SetDeviceGroupConfigCommand(
            group_id_q, std::move(before), std::move(after),
            std::move(command_text), std::move(base_snapshot),
            [this, group_id](pc::WorkspaceConfiguration config) {
              QMetaObject::invokeMethod(
                  this,
                  [this, group_id, config = std::move(config)]() mutable {
                    applyWorkspaceConfigAndRebuild(std::move(config),
                                                   RebuildScope::Devices);
                    retransformGroupDescendants(group_id);
                  },
                  Qt::QueuedConnection);
            }));
      });
}

void WorkspaceModel::retransformGroupDescendants(const std::string &group_id) {
  // orbbec re-applies the world transform every frame, so this only matters for
  // static / sequence PLY devices that won't otherwise re-run apply_transform.
  for (auto &device_plugin : _workspace.devices) {
    if (!device_plugin) continue;
    auto [device_id, _] =
        pc::devices::device_info_from_variant(device_plugin->config_variant());
    if (isDescendantOf(std::string(device_id), group_id))
      device_plugin->on_config_field_changed("transform");
  }
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
            if constexpr (requires { device_config.operators; }) {
              device_config.operators.push_back(
                  OperatorConfigType{.id = pc::uuid::word()});
            }
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
    auto *p = _workspace.devices[deviceIndex].get();
    pc::logger()->debug(
        "addOperatorToDevice: deviceIndex={} idx={} plugin={} has_workspace={}",
        deviceIndex, idx, fmt::ptr(p), p->has_workspace());
    p->update_config(_workspace.config.devices[size_t(idx)]);
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
        if constexpr (requires { device_config.operators; }) {
          auto &ops = device_config.operators;
          if (operatorIndex >= 0 && operatorIndex < int(ops.size()))
            ops.erase(ops.begin() + operatorIndex);
        }
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
        if constexpr (requires { device_config.operators; }) {
          auto &ops = device_config.operators;
          if (fromIndex < 0 || fromIndex >= int(ops.size())) return;
          if (toIndex < 0 || toIndex >= int(ops.size())) return;
          auto item = std::move(ops[fromIndex]);
          ops.erase(ops.begin() + fromIndex);
          ops.insert(ops.begin() + toIndex, std::move(item));
        }
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
          const int device_index = find_device_index_by_id(
              _workspace.config, device_id.toStdString());
          if (device_index < 0) return;
          const int op_idx = find_operator_index_by_id(
              _workspace.config.devices[size_t(device_index)],
              operator_id.toStdString());
          if (op_idx < 0) return;
          std::visit(
              [&](const auto &device_config) {
                if constexpr (requires { device_config.operators; }) {
                  before = device_config.operators[size_t(op_idx)];
                }
              },
              _workspace.config.devices[size_t(device_index)]);
        }
        const bool changed = configAdapter->apply(path, value);
        if (!changed) return;
        pc::operators::OperatorConfigurationVariant after =
            plugin->config_variant();
        // Write the new operator config back into workspace config
        {
          std::scoped_lock lock(_workspace.config_access);
          const int device_index = find_device_index_by_id(
              _workspace.config, device_id.toStdString());
          if (device_index < 0) {
            return;
          }
          const int op_idx = find_operator_index_by_id(
              _workspace.config.devices[size_t(device_index)],
              operator_id.toStdString());
          if (op_idx < 0) {
            return;
          }
          std::visit(
              [&](auto &device_config) {
                if constexpr (requires { device_config.operators; }) {
                  device_config.operators[size_t(op_idx)] = after;
                }
              },
              _workspace.config.devices[size_t(device_index)]);
        }

        // TODO
        // sync the new operator config with plugin's versions... but this seems
        // like it should have happened automatically because of above ^ i think
        // we want workspace config to be canonical source of truth throughout
        // application somehow
        const int device_index =
            find_device_index_by_id(_workspace.config, device_id.toStdString());
        if (device_index >= 0 &&
            device_index < int(_workspace.devices.size()) &&
            _workspace.devices[device_index]) {
          const int op_idx = find_operator_index_by_id(
              _workspace.devices[device_index]->config_variant(),
              operator_id.toStdString());
          if (op_idx >= 0) {
            std::visit(
                [&](auto &device_config) {
                  if constexpr (requires { device_config.operators; }) {
                    device_config.operators[size_t(op_idx)] = after;
                  }
                },
                _workspace.devices[device_index]->config_variant());
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

void WorkspaceModel::initPointStreamerAdapter() {
  if (!_pointStreamerAdapter) return;

  // Subscribe to external (OSC/etc.) changes on the streamer config
  _workspace.config_registry.remove_subscriptions("streaming/");
  auto adapterPtr = QPointer<ConfigAdapter>(_pointStreamerAdapter.data());
  _workspace.config_registry.on_change(
      "streaming/", [adapterPtr](std::string_view path) {
        if (!adapterPtr) return;
        const std::string local(
            path.substr(std::string_view("streaming/").size()));
        const QString qpath = QString::fromStdString(local);
        QMetaObject::invokeMethod(
            adapterPtr.data(),
            [adapterPtr, qpath]() {
              if (adapterPtr) adapterPtr->notifyFieldChanged(qpath);
            },
            Qt::QueuedConnection);
      });

  QObject::connect(
      _pointStreamerAdapter, &ConfigAdapter::editRequested, this,
      [this](const QString &path, const QVariant &value) {
        auto *adapter = _pointStreamerAdapter.data();
        if (!adapter) return;

        pc::networking::PointStreamerConfiguration before;
        pc::networking::PointStreamerConfiguration after;
        pc::WorkspaceConfiguration base_snapshot;
        {
          std::scoped_lock lock(_workspace.config_access);
          base_snapshot = _workspace.config;
          before = _workspace.config.point_streamer.value();
          // apply() mutates the live config member directly (m_config aliases
          // it)
          const bool changed = adapter->apply(path, value);
          if (!changed) return;
          after = _workspace.config.point_streamer.value();
        }

        QString command_text = QStringLiteral("Edit %1").arg(path);
        _undoStack->push(new SetPointStreamerConfigCommand(
            std::move(before), std::move(after), std::move(command_text),
            std::move(base_snapshot),
            [this](pc::WorkspaceConfiguration config) {
              QMetaObject::invokeMethod(
                  this,
                  [this, config = std::move(config)]() mutable {
                    _workspace.apply_new_config(std::move(config), false);
                    if (_pointStreamerAdapter)
                      _pointStreamerAdapter->setConfig(pc::ConfigurationVariant{
                          _workspace.config.point_streamer.value()});
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

void WorkspaceModel::syncSessionPointCloudAdapters() {
  // Sync session point cloud adapters (one per live session).
  // Same callback pattern as device adapters.
  QSet<QString> live_session_ids;
  for (auto &[session_id_str, session_ptr] : _workspace.sessions) {
    const QString id = QString::fromStdString(session_id_str);
    live_session_ids.insert(id);

    if (!_sessionPointCloudAdapters.contains(id) ||
        !_sessionPointCloudAdapters[id]) {
      auto *adapter =
          new SessionAdapter(session_ptr.get(), _imageProvider, this);
      _sessionPointCloudAdapters[id] = adapter;
    }

    // Re-wire the callback every sync so a rebuilt session gets a fresh hook.
    auto gate = std::make_shared<SessionUpdateGate>();
    gate->adapter = QPointer<SessionAdapter>(_sessionPointCloudAdapters[id]);
    auto *model = this; // outlives all sessions; safe to capture raw

    session_ptr->set_point_cloud_updated_callback([gate, model]() {
      // Called on a pipeline worker thread... but if an update is already
      // queued for the UI thread, just drop this one
      bool expected = false;
      if (!gate->pending.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel))
        return;
      QMetaObject::invokeMethod(
          model,
          [gate]() {
            gate->pending.store(false, std::memory_order_release);
            if (gate->adapter) gate->adapter->notifyPointCloudUpdated();
          },
          Qt::QueuedConnection);
    });

    auto pbGate = std::make_shared<SessionUpdateGate>();
    pbGate->adapter = QPointer<SessionAdapter>(_sessionPointCloudAdapters[id]);

    session_ptr->set_playback_changed_callback([pbGate, model]() {
      bool expected = false;
      if (!pbGate->pending.compare_exchange_strong(expected, true,
                                                   std::memory_order_acq_rel))
        return;
      QMetaObject::invokeMethod(
          model,
          [pbGate]() {
            pbGate->pending.store(false, std::memory_order_release);
            if (pbGate->adapter) pbGate->adapter->notifyPlaybackChanged();
          },
          Qt::QueuedConnection);
    });
  }
  // Remove adapters for sessions that no longer exist
  for (auto it = _sessionPointCloudAdapters.begin();
       it != _sessionPointCloudAdapters.end();) {
    if (!live_session_ids.contains(it.key())) {
      if (it.value()) it.value()->deleteLater();
      it = _sessionPointCloudAdapters.erase(it);
    } else {
      ++it;
    }
  }
}

SessionAdapter *
WorkspaceModel::sessionPointCloudAdapterFor(const QString &sessionId) const {
  auto it = _sessionPointCloudAdapters.find(sessionId);
  return (it != _sessionPointCloudAdapters.end() && it.value())
             ? it.value().data()
             : nullptr;
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
  using ConfigT = typename AdapterT::config_type;
  if constexpr (std::same_as<ConfigT, devices::DeviceGroupConfiguration>) {
    return;
  }
  const auto &[device_id, plugin_name] =
      devices::device_info_from_variant(plugin->config_variant());
  pc::logger()->trace("Initialising device adapter '{}' for plugin '{}'",
                      device_id, plugin_name);
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
        if constexpr (!std::same_as<ConfigType,
                                    pc::devices::DeviceGroupConfiguration>) {
          using AdapterType =
              pc::devices::device_adapter_for_config_t<ConfigType>;
          auto *adapter =
              new AdapterType(device_config, plugin, _imageProvider, this);
          initDeviceAdapter(adapter, plugin);
          result = adapter;
        }
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

  // Refresh registry subscriptions so OSC/external changes notify QML.
  _workspace.config_registry.remove_subscriptions("session/");
  for (QObject *obj : _sessionAdapters) {
    auto *adapter = qobject_cast<ConfigAdapter *>(obj);
    if (!adapter) continue;
    const std::string id = adapterStableId(adapter).toStdString();
    if (id.empty()) continue;
    const std::string prefix = "session/" + id + "/";
    auto adapterPtr = QPointer<ConfigAdapter>(adapter);
    _workspace.config_registry.on_change(
        prefix, [adapterPtr, prefix](std::string_view path) {
          if (!adapterPtr) return;
          std::string local(path.substr(prefix.size()));
          // collapse float3 component sub-paths to the parent path
          // e.g. "transform/position/x" becomes "transform/position"
          for (std::string_view sfx : {"/x", "/y", "/z"}) {
            if (local.ends_with(sfx)) {
              local = local.substr(0, local.size() - sfx.size());
              break;
            }
          }
          const QString qpath = QString::fromStdString(local);
          QMetaObject::invokeMethod(
              adapterPtr.data(),
              [adapterPtr, qpath]() {
                if (adapterPtr) adapterPtr->notifyFieldChanged(qpath);
              },
              Qt::QueuedConnection);
        });
  }

  emit sessionAdaptersChanged();

  syncSessionPointCloudAdapters();

  // Rebuild per-session operator adapters from each session's live
  // OperatorPlugin instances.
  syncSessionOperatorAdapters();

  pc::logger()->trace("Finished syncing sessionAdapters");
}

void WorkspaceModel::syncDeviceAdapters() {
  pc::logger()->trace("syncDeviceAdapters: begin, workspace device count={}",
                      _workspace.devices.size());

  QHash<QString, DeviceAdapter *> existing_by_id;
  existing_by_id.reserve(_deviceAdapters.size());
  for (QObject *obj : _deviceAdapters) {
    auto *a = qobject_cast<DeviceAdapter *>(obj);
    if (!a) continue;
    const QString id = adapterStableId(a);
    if (!id.isEmpty()) existing_by_id.insert(id, a);
  }
  pc::logger()->trace("syncDeviceAdapters: existing adapter count={}",
                      existing_by_id.size());

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
      pc::logger()->trace(
          "syncDeviceAdapters: processing device id='{}' plugin='{}'",
          device_id, plugin_name);

      if (!id.isEmpty()) {
        adapter = existing_by_id.take(id);
        if (adapter) {
          pc::logger()->trace(
              "syncDeviceAdapters: found existing adapter for id='{}'",
              device_id);
          bool sync_success = false;
          try {
            sync_success = adapter->setConfig(config);
          } catch (...) {
            pc::logger()->error(
                "syncDeviceAdapters: exception setting config for '{}' '{}'",
                plugin_name, device_id);
          }
          if (!sync_success) {
            pc::logger()->trace("syncDeviceAdapters: setConfig failed for "
                                "id='{}', recreating adapter",
                                device_id);
            adapter->deleteLater();
            adapter = nullptr;
          } else {
            pc::logger()->trace("syncDeviceAdapters: setConfig ok for id='{}'",
                                device_id);
          }
        } else {
          pc::logger()->trace("syncDeviceAdapters: no existing adapter for "
                              "id='{}', will construct",
                              device_id);
        }
      }

      if (!adapter) {
        pc::logger()->trace("syncDeviceAdapters: constructing new adapter for "
                            "id='{}' plugin='{}'",
                            device_id, plugin_name);
        adapter = makeDeviceAdapterForPlugin(plugin, config);
        pc::logger()->trace("syncDeviceAdapters: construction {}",
                            adapter ? "ok" : "FAILED");
      }

      if (adapter) {
        pc::logger()->trace("syncDeviceAdapters: hooking callbacks for id='{}'",
                            device_id);
        auto adapterPtr = QPointer<DeviceAdapter>(adapter);
        plugin->set_point_cloud_updated_callback([adapterPtr]() {
          QMetaObject::invokeMethod(adapterPtr.data(), [adapterPtr]() {
            if (!adapterPtr) {
              pc::logger()->error("Invalid point cloud plugin ptr");
              return;
            }
            adapterPtr->notifyPointCloudUpdated();
            adapterPtr->syncCameraFrames();
          });
        });

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

        if (operator_structure_changed(adapter, plugin)) {
          pc::logger()->trace("syncDeviceAdapters: operator structure changed "
                              "for id='{}', rebuilding",
                              device_id);
          adapter->rebuildOperatorAdapters();
          attachOperatorConfigAdapters(adapter);
        }
      }
    } else {
      pc::logger()->trace(
          "syncDeviceAdapters: plugin has no config variant, skipping");
    }

    if (adapter) {
      new_ordered_devices.append(adapter);
      pc::logger()->trace("syncDeviceAdapters: appended adapter, list size={}",
                          new_ordered_devices.size());
    } else {
      pc::logger()->error(
          "syncDeviceAdapters: no adapter produced for a device plugin");
    }
  }

  pc::logger()->trace("syncDeviceAdapters: deleting {} stale adapters",
                      existing_by_id.size());
  for (auto it = existing_by_id.begin(); it != existing_by_id.end(); ++it) {
    if (it.value()) {
      pc::logger()->trace("syncDeviceAdapters: deleting stale adapter id='{}'",
                          it.key().toStdString());
      it.value()->deleteLater();
    }
  }

  _deviceAdapters = new_ordered_devices;
  pc::logger()->trace("syncDeviceAdapters: device adapters rebuilt, count={}",
                      _deviceAdapters.size());

  _workspace.config_registry.remove_subscriptions("device/");
  pc::logger()->trace(
      "syncDeviceAdapters: removed existing device/ registry subscriptions");

  for (QObject *obj : _deviceAdapters) {
    auto *adapter = qobject_cast<ConfigAdapter *>(obj);
    if (!adapter) continue;
    const std::string id = adapterStableId(adapter).toStdString();
    if (id.empty()) continue;
    const std::string node_prefix = node_path_prefix(_workspace.config, id);
    adapter->setConfigPath(QString::fromStdString(node_prefix));
    const std::string prefix = node_prefix + "/";
    pc::logger()->trace(
        "syncDeviceAdapters: registering on_change for prefix='{}'", prefix);
    auto adapterPtr = QPointer<ConfigAdapter>(adapter);
    _workspace.config_registry.on_change(
        prefix, [adapterPtr, prefix, id,
                 &workspace = _workspace](std::string_view path) {
          {
            std::scoped_lock lock(workspace.config_access);
            for (auto &device_plugin : workspace.devices) {
              if (!device_plugin) continue;
              auto [did, _] = pc::devices::device_info_from_variant(
                  device_plugin->config_variant());
              if (std::string(did) != id) continue;
              for (auto &wc : workspace.config.devices) {
                const bool match =
                    std::visit([&id](const auto &d) { return d.id == id; }, wc);
                if (match) {
                  wc = device_plugin->config_variant();
                  break;
                }
              }
              break;
            }
          }
          if (!adapterPtr) return;
          std::string local(path.substr(prefix.size()));
          for (std::string_view sfx : {"/x", "/y", "/z"}) {
            if (local.ends_with(sfx)) {
              local = local.substr(0, local.size() - sfx.size());
              break;
            }
          }
          const QString qpath = QString::fromStdString(local);
          QMetaObject::invokeMethod(
              adapterPtr.data(),
              [adapterPtr, qpath]() {
                if (adapterPtr) adapterPtr->notifyFieldChanged(qpath);
              },
              Qt::QueuedConnection);
        });
  }

  int selectedDeviceIndex = _workspace.config.selectedDeviceIndex.value();
  if (selectedDeviceIndex >= _deviceAdapters.size() || selectedDeviceIndex < 0)
    selectedDeviceIndex = 0;

  const bool group_still_selected =
      _selectedNodeKind == QStringLiteral("group") &&
      find_device_group_index_by_id(_workspace.config,
                                    _selectedNodeId.toStdString()) >= 0;

  pc::logger()->trace(
      "syncDeviceAdapters: selection — index={}, group_still_selected={}, "
      "selectedNodeKind='{}', selectedNodeId='{}'",
      selectedDeviceIndex, group_still_selected,
      _selectedNodeKind.toStdString(), _selectedNodeId.toStdString());

  if (group_still_selected) {
    if (_selectedDeviceIndex != selectedDeviceIndex) {
      _selectedDeviceIndex = selectedDeviceIndex;
      _workspace.config.selectedDeviceIndex = selectedDeviceIndex;
      emit selectedDeviceIndexChanged();
    }
  } else {
    setSelectedDeviceIndex(selectedDeviceIndex);
  }

  emit deviceAdaptersChanged();
  emit deviceVariantNamesChanged();
  emit addDeviceMenuEntriesChanged();
  emit deviceTreeRowsChanged();

  pc::logger()->trace("syncDeviceAdapters: done, workspace device count={}",
                      _workspace.devices.size());
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

bool WorkspaceModel::isDescendantOf(
    const std::string &node_id, const std::string &maybe_ancestor_id) const {
  std::string current_id = node_id;
  while (!current_id.empty()) {
    if (current_id == maybe_ancestor_id) return true;
    const int group_index =
        find_device_group_index_by_id(_workspace.config, current_id);
    if (group_index < 0) break;
    current_id =
        _workspace.config.device_groups[size_t(group_index)].parent_id.value();
  }
  return false;
}

void WorkspaceModel::setDeviceGroupRender(const QString &group_id,
                                          bool render) {
  auto new_config = _workspace.config;
  const int group_index =
      find_device_group_index_by_id(new_config, group_id.toStdString());
  if (group_index < 0) return;
  new_config.device_groups[size_t(group_index)].render.set(render);
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

void WorkspaceModel::setDeviceGroupActive(const QString &group_id,
                                          bool active) {
  auto new_config = _workspace.config;
  const int group_index =
      find_device_group_index_by_id(new_config, group_id.toStdString());
  if (group_index < 0) return;
  new_config.device_groups[size_t(group_index)].active.set(active);
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

void WorkspaceModel::createDeviceGroup(const QString &label,
                                       const QString &parent_id) {
  auto new_config = _workspace.config;

  // reject a parent that doesn't exist (empty string is valid, root level)
  const std::string parent = parent_id.toStdString();
  if (!parent.empty() &&
      find_device_group_index_by_id(new_config, parent) < 0) {
    pc::logger()->warn("createDeviceGroup: parent group not found '{}'",
                       parent);
    return;
  }

  pc::devices::DeviceGroupConfiguration group_config{.id = pc::uuid::word()};
  group_config.label.set(label.toStdString());
  group_config.parent_id.set(parent);
  new_config.device_groups.push_back(std::move(group_config));

  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

void WorkspaceModel::deleteDeviceGroup(const QString &group_id) {
  auto new_config = _workspace.config;
  const std::string target_group_id = group_id.toStdString();

  const int group_index =
      find_device_group_index_by_id(new_config, target_group_id);
  if (group_index < 0) {
    pc::logger()->warn("deleteDeviceGroup: group id not found '{}'",
                       target_group_id);
    return;
  }

  // children (groups and devices) are reparented to the deleted group's parent
  const std::string surviving_parent_id =
      new_config.device_groups[size_t(group_index)].parent_id.value();

  for (auto &group_config : new_config.device_groups) {
    if (group_config.parent_id.value() == target_group_id)
      group_config.parent_id.set(surviving_parent_id);
  }
  for (auto &device_variant : new_config.devices) {
    std::visit(
        [&](auto &device_config) {
          if (device_config.parent_id.value() == target_group_id)
            device_config.parent_id.set(surviving_parent_id);
        },
        device_variant);
  }

  new_config.device_groups.erase(new_config.device_groups.begin() +
                                 group_index);

  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

void WorkspaceModel::setDeviceGroupCollapsed(const QString &group_id,
                                             bool collapsed) {
  auto new_config = _workspace.config;
  const int group_index =
      find_device_group_index_by_id(new_config, group_id.toStdString());
  if (group_index < 0) return;
  new_config.device_groups[size_t(group_index)].collapsed.set(collapsed);
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

void WorkspaceModel::setDeviceGroupLabel(const QString &group_id,
                                         const QString &label) {
  auto new_config = _workspace.config;
  const int group_index =
      find_device_group_index_by_id(new_config, group_id.toStdString());
  if (group_index < 0) return;
  new_config.device_groups[size_t(group_index)].label.set(label.toStdString());
  applyWorkspaceConfigAndRebuild(std::move(new_config), RebuildScope::Devices);
}

QVariantList WorkspaceModel::deviceTreeRows() const {
  QVariantList rows;

  struct Entry {
    bool is_group;
    int order;
    int index;
  };

  std::function<void(const std::string &, int)> walk =
      [&](const std::string &parent_id, int depth) {
        std::vector<Entry> entries;

        for (int group_index = 0;
             group_index < int(_workspace.config.device_groups.size());
             group_index++) {
          const auto &group_config =
              _workspace.config.device_groups[size_t(group_index)];
          if (group_config.parent_id.value() == parent_id)
            entries.push_back({true, group_config.order.value(), group_index});
        }
        for (int i = 0; i < int(_workspace.config.devices.size()); ++i) {
          const auto &device_variant = _workspace.config.devices[size_t(i)];
          const std::string device_parent = std::visit(
              [](const auto &device_config) {
                return device_config.parent_id.value();
              },
              device_variant);
          if (device_parent != parent_id) continue;
          const int device_order = std::visit(
              [](const auto &device_config) {
                return device_config.order.value();
              },
              device_variant);
          entries.push_back({false, device_order, i});
        }

        std::stable_sort(
            entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.order < b.order; });

        for (const auto &entry : entries) {
          if (entry.is_group) {
            const auto &group_config =
                _workspace.config.device_groups[size_t(entry.index)];
            QVariantMap group_row;
            group_row["kind"] = "group";
            group_row["id"] = QString::fromStdString(group_config.id);
            group_row["parentId"] = QString::fromStdString(parent_id);
            group_row["label"] =
                QString::fromStdString(group_config.label.value());
            group_row["depth"] = depth;
            group_row["collapsed"] = group_config.collapsed.value();
            group_row["render"] = group_config.render.value();
            group_row["active"] = group_config.active.value();
            group_row["effectiveRender"] = pc::devices::effective_render(
                _workspace.config, group_config.id);
            group_row["effectiveActive"] = pc::devices::effective_active(
                _workspace.config, group_config.id);
            rows.push_back(std::move(group_row));
            if (!group_config.collapsed.value())
              walk(group_config.id, depth + 1);
          } else {
            const auto &device_variant =
                _workspace.config.devices[size_t(entry.index)];
            const std::string device_id = std::visit(
                [](const auto &device_config) { return device_config.id; },
                device_variant);
            QVariantMap device_row;
            device_row["kind"] = "device";
            device_row["id"] = QString::fromStdString(device_id);
            device_row["parentId"] = QString::fromStdString(parent_id);
            device_row["deviceIndex"] = entry.index;
            device_row["depth"] = depth;
            device_row["effectiveRender"] =
                pc::devices::effective_render(_workspace.config, device_id);
            device_row["effectiveActive"] =
                pc::devices::effective_active(_workspace.config, device_id);
            rows.push_back(std::move(device_row));
          }
        }
      };

  walk("", 0);
  return rows;
}

void WorkspaceModel::moveDeviceNode(const QString &node_id,
                                    const QString &new_parent_id,
                                    const QString &before_node_id) {
  const std::string id = node_id.toStdString();
  const std::string new_parent = new_parent_id.toStdString();
  const std::string before = before_node_id.toStdString();

  if (id.empty() || id == new_parent) return;

  // accessors that work uniformly across groups and the device variant
  const auto node_parent =
      [&](const std::string &nid) -> std::optional<std::string> {
    if (const int group_index =
            find_device_group_index_by_id(_workspace.config, nid);
        group_index >= 0)
      return _workspace.config.device_groups[size_t(group_index)]
          .parent_id.value();
    for (const auto &device_variant : _workspace.config.devices) {
      std::optional<std::string> found;
      std::visit(
          [&](const auto &device_config) {
            if (device_config.id == nid)
              found = device_config.parent_id.value();
          },
          device_variant);
      if (found.has_value()) return found;
    }
    return std::nullopt;
  };
  const auto set_parent = [&](const std::string &nid,
                              const std::string &parent) {
    if (const int group_index =
            find_device_group_index_by_id(_workspace.config, nid);
        group_index >= 0) {
      _workspace.config.device_groups[size_t(group_index)].parent_id = parent;
      return;
    }
    for (auto &device_variant : _workspace.config.devices)
      std::visit(
          [&](auto &device_config) {
            if (device_config.id == nid) device_config.parent_id = parent;
          },
          device_variant);
  };
  const auto set_order = [&](const std::string &nid, int order) {
    if (const int group_index =
            find_device_group_index_by_id(_workspace.config, nid);
        group_index >= 0) {
      _workspace.config.device_groups[size_t(group_index)].order = order;
      return;
    }
    for (auto &device_variant : _workspace.config.devices)
      std::visit(
          [&](auto &device_config) {
            if (device_config.id == nid) device_config.order = order;
          },
          device_variant);
  };

  // node must exist
  if (!node_parent(id).has_value()) return;

  // prevent moving a group into its own subtree (walk up from the new parent)
  for (std::string p = new_parent; !p.empty();) {
    if (p == id) return;
    const auto pp = node_parent(p);
    if (!pp.has_value()) break;
    p = *pp;
  }

  // reparent first so sibling gathering reflects the destination
  set_parent(id, new_parent);

  // gather destination siblings (both kinds), excluding the moved node,
  // in their current visual order
  struct Sibling {
    std::string id;
    int order;
  };
  std::vector<Sibling> siblings;
  for (const auto &group_config : _workspace.config.device_groups) {
    if (group_config.id == id) continue;
    if (group_config.parent_id.value() == new_parent)
      siblings.push_back({group_config.id, group_config.order.value()});
  }
  for (const auto &device_variant : _workspace.config.devices) {
    std::string did;
    std::string dparent;
    int dorder = 0;
    std::visit(
        [&](const auto &device_config) {
          did = device_config.id;
          dparent = device_config.parent_id.value();
          dorder = device_config.order.value();
        },
        device_variant);
    if (did == id) continue;
    if (dparent == new_parent) siblings.push_back({did, dorder});
  }
  std::stable_sort(
      siblings.begin(), siblings.end(),
      [](const Sibling &a, const Sibling &b) { return a.order < b.order; });

  // resolve insertion position (append when before is empty or not found)
  size_t insert_at = siblings.size();
  if (!before.empty()) {
    for (size_t i = 0; i < siblings.size(); ++i) {
      if (siblings[i].id == before) {
        insert_at = i;
        break;
      }
    }
  }

  // build the resulting order and reassign contiguous indices to all siblings
  std::vector<std::string> ordered;
  ordered.reserve(siblings.size() + 1);
  for (size_t i = 0; i < siblings.size(); ++i) {
    if (i == insert_at) ordered.push_back(id);
    ordered.push_back(siblings[i].id);
  }
  if (insert_at >= siblings.size()) ordered.push_back(id);

  for (int i = 0; i < int(ordered.size()); ++i)
    set_order(ordered[size_t(i)], i);

  _workspace.rebuild_config_registry();
  syncDeviceAdapters();
  emit publishPathsChanged();
  emit pushPathsChanged();
  _streamChannelModel->refresh();
}

} // namespace pc::ui