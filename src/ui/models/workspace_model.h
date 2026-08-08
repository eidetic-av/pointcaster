#pragma once

#include "config_adapter.h"
#include "device_adapter.h"
#include "enum_adapters.h"
#include "operator_adapter.h"
#include "session_adapter.h"
#include "session_recorder_model.h"
#include "stream_channel_model.h"
#include <QMatrix4x4>
#include <QObject>
#include <QPointer>
#include <QUndoStack>
#include <QUrl>
#include <QVariant>
#include <config/config_variant.h>
#include <functional>
#include <networking/point_streamer_config_adapter.gen.h>
#include <qtmetamacros.h>
#include <session/session_config_adapter.gen.h>
#include <workspace/workspace_config.h>

namespace pc {
class Workspace;
class Session;
struct SessionConfiguration;

namespace devices {
class DevicePlugin;
class DeviceGroupConfigurationAdapter;
} // namespace devices

} // namespace pc

class CameraImageProvider;

namespace pc::ui {

class WorkspaceModel : public QObject {
  Q_OBJECT

  Q_PROPERTY(QUndoStack *undoStack READ undoStack CONSTANT)

  // Sessions
  Q_PROPERTY(QList<QObject *> sessionAdapters READ sessionAdapters NOTIFY
                 sessionAdaptersChanged)
  Q_PROPERTY(QString selectedSessionId READ selectedSessionId WRITE
                 setSelectedSessionId NOTIFY selectedSessionChanged)
  Q_PROPERTY(QObject *selectedSessionAdapter READ selectedSessionAdapter NOTIFY
                 selectedSessionChanged)
  Q_PROPERTY(QList<OperatorAdapter *> selectedSessionOperatorAdapters READ
                 selectedSessionOperatorAdapters NOTIFY selectedSessionChanged)

  // Devices
  Q_PROPERTY(
      QVariant deviceAdapters READ deviceAdapters NOTIFY deviceAdaptersChanged)
  Q_PROPERTY(QStringList deviceVariantNames READ deviceVariantNames NOTIFY
                 deviceVariantNamesChanged)
  Q_PROPERTY(QVariantList addDeviceMenuEntries READ addDeviceMenuEntries NOTIFY
                 addDeviceMenuEntriesChanged)
  Q_PROPERTY(int selectedDeviceIndex READ selectedDeviceIndex WRITE
                 setSelectedDeviceIndex NOTIFY selectedDeviceIndexChanged)
  Q_PROPERTY(QVariantList deviceTreeRows READ deviceTreeRows NOTIFY
                 deviceTreeRowsChanged)
  Q_PROPERTY(
      QString selectedNodeId READ selectedNodeId NOTIFY selectedNodeChanged)
  Q_PROPERTY(
      QString selectedNodeKind READ selectedNodeKind NOTIFY selectedNodeChanged)

  Q_PROPERTY(QObject *selectedDeviceGroupAdapter READ selectedDeviceGroupAdapter
                 NOTIFY selectedDeviceGroupAdapterChanged)

  // publish field paths
  Q_PROPERTY(
      QStringList publishPaths READ publishPaths NOTIFY publishPathsChanged)
  Q_PROPERTY(QStringList pushPaths READ pushPaths NOTIFY pushPathsChanged)

  // Selected operator (owned by a device)
  Q_PROPERTY(OperatorAdapter *selectedOperatorAdapter READ
                 selectedOperatorAdapter WRITE setSelectedOperatorAdapter NOTIFY
                     selectedOperatorAdapterChanged)

  Q_PROPERTY(QUrl saveFileUrl READ saveFileUrl WRITE setSaveFileUrl NOTIFY
                 saveFileUrlChanged)

  // Console logger window
  Q_PROPERTY(QVariantList consoleOverlayEntries READ consoleOverlayEntries
                 NOTIFY consoleOverlayEntriesChanged)
  Q_PROPERTY(QVariantList consoleHistoryEntries READ consoleHistoryEntries
                 NOTIFY consoleHistoryEntriesChanged)

  Q_PROPERTY(QVariantMap foldedPropertyPaths READ foldedPropertyPaths NOTIFY
                 foldedPropertyPathsChanged)

  Q_PROPERTY(QVariantMap uiState READ uiState NOTIFY uiStateChanged)

  // Streaming
  Q_PROPERTY(QObject *pointStreamerAdapter READ pointStreamerAdapter CONSTANT)
  Q_PROPERTY(QAbstractListModel *streamChannels READ streamChannels CONSTANT)
  Q_PROPERTY(int selectedStreamChannelIndex READ selectedStreamChannelIndex
                 WRITE setSelectedStreamChannelIndex NOTIFY
                     selectedStreamChannelIndexChanged)

  // Recording
  Q_PROPERTY(RecorderModel *recorder READ recorder NOTIFY recorderChanged)

public:
  explicit WorkspaceModel(pc::Workspace *workspace, QObject *parent);

  Q_INVOKABLE void close();
  Q_INVOKABLE void loadFromFile(const QUrl &file);
  Q_INVOKABLE void save(bool update_last_session_path = true);

  Q_INVOKABLE void newWorkspace();

  Q_INVOKABLE void registerPluginSettingsPages();

  QUndoStack *undoStack() const { return _undoStack; }

  Q_INVOKABLE void triggerDeviceDiscovery();
  Q_INVOKABLE void addNewDevice(const QString &plugin_name,
                                const QString &target_ip = "",
                                const QString &target_id = "",
                                const QString &target_type_label = "");
  Q_INVOKABLE void deleteDevice(const QString &target_id);
  Q_INVOKABLE void deleteSelectedDevice();
  Q_INVOKABLE void duplicateDeviceNode(const QString &node_id);

  QList<QObject *> sessionAdapters() const;
  Q_INVOKABLE QList<QObject *> sessionAdaptersCallable() const {
    return sessionAdapters();
  }

  QString selectedSessionId() const { return _selectedSessionId; }
  void setSelectedSessionId(const QString &id);
  QObject *selectedSessionAdapter() const;
  QList<OperatorAdapter *> selectedSessionOperatorAdapters() const;

  Q_INVOKABLE void addOperatorToSession(const QString &sessionId,
                                        const QString &operatorPluginName);
  Q_INVOKABLE void removeOperatorFromSession(const QString &sessionId,
                                             int operatorIndex);
  Q_INVOKABLE void reorderOperatorOnSession(const QString &sessionId,
                                            int fromIndex, int toIndex);

  Q_INVOKABLE SessionAdapter *
  sessionPointCloudAdapterFor(const QString &sessionId) const;

  QVariant deviceAdapters() const;

  QStringList deviceVariantNames() const;
  QVariantList addDeviceMenuEntries() const;

  int selectedDeviceIndex() const { return _selectedDeviceIndex; }
  void setSelectedDeviceIndex(int index);

  Q_INVOKABLE DeviceAdapter *deviceAdapterAt(int index) const;
  Q_INVOKABLE DeviceAdapter *selectedDeviceAdapter() const {
    return deviceAdapterAt(_selectedDeviceIndex);
  }

  QVariantList deviceTreeRows() const;

  QString selectedNodeId() const { return _selectedNodeId; }
  QString selectedNodeKind() const { return _selectedNodeKind; }
  Q_INVOKABLE void selectNode(const QString &node_id);

  QObject *selectedDeviceGroupAdapter() const;

  Q_INVOKABLE QMatrix4x4 nodeAncestorWorldMatrix(const QString &node_id) const;

  Q_INVOKABLE void moveDeviceNode(const QString &node_id,
                                  const QString &new_parent_id,
                                  const QString &before_node_id = "");

  Q_INVOKABLE void createDeviceGroup(const QString &label,
                                     const QString &parent_id = "");
  Q_INVOKABLE void deleteDeviceGroup(const QString &group_id);
  Q_INVOKABLE void setDeviceGroupActive(const QString &group_id, bool active);
  Q_INVOKABLE void setDeviceGroupRender(const QString &group_id, bool render);
  Q_INVOKABLE void setDeviceGroupCollapsed(const QString &group_id,
                                           bool collapsed);
  Q_INVOKABLE void setDeviceGroupLabel(const QString &group_id,
                                       const QString &label);

  OperatorAdapter *selectedOperatorAdapter() const {
    return _selectedOperatorAdapter;
  }
  void setSelectedOperatorAdapter(OperatorAdapter *adapter);

  Q_INVOKABLE void addOperatorToDevice(int deviceIndex,
                                       const QString &operatorPluginName);
  Q_INVOKABLE void removeOperatorFromDevice(int deviceIndex, int operatorIndex);
  Q_INVOKABLE void reorderOperatorOnDevice(int deviceIndex, int fromIndex,
                                           int toIndex);

  Q_INVOKABLE void attachOperatorConfigAdapters(DeviceAdapter *deviceAdapter);
  Q_INVOKABLE void initOperatorAdapter(OperatorAdapter *opAdapter,
                                       DeviceAdapter *deviceAdapter);

  QUrl saveFileUrl() const { return _saveFileUrl; }
  void setSaveFileUrl(const QUrl &url) {
    if (_saveFileUrl == url) return;
    _saveFileUrl = url;
    emit saveFileUrlChanged();
  }

  QVariantList consoleOverlayEntries() const;
  QVariantList consoleHistoryEntries() const;

  QObject *pointStreamerAdapter() const { return _pointStreamerAdapter.data(); }

  QAbstractListModel *streamChannels() const { return _streamChannelModel; }

  int selectedStreamChannelIndex() const { return _selectedStreamChannelIndex; }
  void setSelectedStreamChannelIndex(int index);

  RecorderModel *recorder() const { return _recorderModel; }

  Q_INVOKABLE QVariantMap foldedPropertyPaths() const {
    return _foldedPropertyPaths;
  }

  Q_INVOKABLE void setFoldedProperty(const QString &path, bool folded) {
    _foldedPropertyPaths[path] = folded;
    foldedPropertyPathsChanged();
  }

  Q_INVOKABLE QVariantMap uiState() const { return _uiState; }

  Q_INVOKABLE void setUiState(const QVariantMap &state) {
    _uiState = state;
    emit uiStateChanged();
  }

  Q_INVOKABLE void setUiStateValue(const QString &key, const QVariant &value) {
    _uiState[key] = value;
    emit uiStateChanged();
  }

  void setImageProvider(CameraImageProvider *provider);

  QStringList publishPaths() const;
  QStringList pushPaths() const;

  Q_INVOKABLE void addPublishPath(const QString &path);
  Q_INVOKABLE void removePublishPath(const QString &path);

  Q_INVOKABLE void addPushPath(const QString &path);
  Q_INVOKABLE void removePushPath(const QString &path);

public slots:
  void syncAdapters();
  void syncSessionAdapters();
  void syncDeviceAdapters();
  void syncConsole();

signals:
  void openSaveAsDialog();

  void newWorkspaceLoaded();

  void sessionAdaptersChanged();
  void selectedSessionChanged();

  void deviceAdaptersChanged();
  void deviceVariantNamesChanged();
  void addDeviceMenuEntriesChanged();
  void selectedDeviceIndexChanged();

  void selectedNodeChanged();

  void selectedDeviceGroupAdapterChanged();

  void deviceAdded();
  void deviceDeleted();

  void deviceTreeRowsChanged();

  void publishPathsChanged();
  void pushPathsChanged();

  void selectedOperatorAdapterChanged();

  void consoleOverlayEntriesChanged();
  void consoleHistoryEntriesChanged();

  void foldedPropertyPathsChanged();

  void saveFileUrlChanged();

  void uiStateChanged();

  void recorderChanged();

  void selectedStreamChannelIndexChanged();

private:
  pc::Workspace &_workspace;

  CameraImageProvider *_imageProvider = nullptr;

  // TODO raw pointer and new? really?
  QUndoStack *_undoStack = new QUndoStack(this);

  QList<QObject *> _sessionAdapters;
  QList<QObject *> _deviceAdapters;

  QHash<QString, QPointer<SessionAdapter>> _sessionPointCloudAdapters;

  QPointer<ConfigAdapter> _selectedDeviceGroupAdapter;

  QPointer<OperatorAdapter> _selectedOperatorAdapter;

  QPointer<ConfigAdapter> _pointStreamerAdapter;

  StreamChannelListModel *_streamChannelModel;
  int _selectedStreamChannelIndex = -1;

  RecorderModel *_recorderModel;

  // Tracks whether an existing SessionConfigurationAdapter is still bound to
  // a valid underlying SessionConfiguration object address.
  QHash<QString, const pc::SessionConfiguration *> _sessionConfigPtrById;

  // Currently selected (focused) session, and the per-session operator
  // adapters wrapping each session's live OperatorPlugin instances.
  QString _selectedSessionId;
  QHash<QString, QList<OperatorAdapter *>> _sessionOperatorAdapters;

  QVariantMap _foldedPropertyPaths;
  QVariantMap _uiState;

  int _selectedDeviceIndex = 0;
  QString _selectedNodeId;
  QString _selectedNodeKind; // "device" | "group" | ""

  QUrl _saveFileUrl;

  // applies a new config and syncs adapters on the UI thread
  enum class RebuildScope { None, Sessions, Devices, All };
  void applyWorkspaceConfigAndRebuild(pc::WorkspaceConfiguration new_config,
                                      RebuildScope scope = RebuildScope::All);

  // Helpers to get stable ids from adapters without relying on Q_PROPERTY
  // names.
  static QString adapterStableId(ConfigAdapter *adapter);
  static QString adapterStableId(DeviceAdapter *adapter);

  void initDeviceAdapter(auto *adapter, pc::devices::DevicePlugin *plugin);
  void initSessionAdapter(pc::SessionConfigurationAdapter *adapter);

  void syncSessionPointCloudAdapters();

  // Session operator adapters (analogous to attach/init for devices).
  void attachSessionOperatorConfigAdapters(const QString &sessionId,
                                           const QList<OperatorAdapter *> &ops);
  void initSessionOperatorAdapter(OperatorAdapter *opAdapter,
                                  const QString &sessionId);
  void rebuildSessionOperatorAdapters(const QString &sessionId,
                                      pc::SessionConfigurationAdapter *adapter,
                                      pc::Session *session);
  void syncSessionOperatorAdapters();

  void rebuildSelectedGroupAdapter();
  void initGroupAdapter(pc::devices::DeviceGroupConfigurationAdapter *adapter);
  void retransformGroupDescendants(const std::string &group_id);

  DeviceAdapter *makeDeviceAdapterForPlugin(
      pc::devices::DevicePlugin *plugin,
      pc::devices::DeviceConfigurationVariant &config_variant);

  bool isDescendantOf(const std::string &node_id,
                      const std::string &maybe_ancestor_id) const;

  void initPointStreamerAdapter();
};

} // namespace pc::ui