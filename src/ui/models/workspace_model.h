#pragma once

#include "config_adapter.h"
#include "device_adapter.h"
#include "device_status.h"
#include "operator_adapter.h"
#include "session_recorder_model.h"
#include <QObject>
#include <QPointer>
#include <QUndoStack>
#include <QUrl>
#include <QVariant>
#include <functional>
#include <qtmetamacros.h>
#include <session/session_config_adapter.gen.h>
#include <workspace/workspace_config.h>

namespace pc {
class Workspace;
class Session;
struct SessionConfiguration;

namespace devices {
class DevicePlugin;
}

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

  // Selected operator (owned by a device)
  Q_PROPERTY(OperatorAdapter *selectedOperatorAdapter READ
                 selectedOperatorAdapter WRITE setSelectedOperatorAdapter NOTIFY
                     selectedOperatorAdapterChanged)

  Q_PROPERTY(QUrl saveFileUrl READ saveFileUrl WRITE setSaveFileUrl)

  // Console logger window
  Q_PROPERTY(QVariantList consoleOverlayEntries READ consoleOverlayEntries
                 NOTIFY consoleOverlayEntriesChanged)
  Q_PROPERTY(QVariantList consoleHistoryEntries READ consoleHistoryEntries
                 NOTIFY consoleHistoryEntriesChanged)

  Q_PROPERTY(QVariantMap foldedPropertyPaths READ foldedPropertyPaths NOTIFY
                 foldedPropertyPathsChanged)

  // Recording
  Q_PROPERTY(RecorderModel *recorder READ recorder NOTIFY recorderChanged)

public:
  explicit WorkspaceModel(pc::Workspace *workspace, QObject *parent);

  Q_INVOKABLE void close();
  Q_INVOKABLE void loadFromFile(const QUrl &file);
  Q_INVOKABLE void save(bool update_last_session_path = true);

  Q_INVOKABLE void newWorkspace();

  QUndoStack *undoStack() const { return _undoStack; }

  Q_INVOKABLE void triggerDeviceDiscovery();
  Q_INVOKABLE void addNewDevice(const QString &plugin_name,
                                const QString &target_ip = "",
                                const QString &target_id = "");
  Q_INVOKABLE void deleteSelectedDevice();

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

  QVariant deviceAdapters() const;

  QStringList deviceVariantNames() const;
  QVariantList addDeviceMenuEntries() const;

  int selectedDeviceIndex() const { return _selectedDeviceIndex; }
  void setSelectedDeviceIndex(int index);

  Q_INVOKABLE DeviceAdapter *deviceAdapterAt(int index) const;
  Q_INVOKABLE DeviceAdapter *selectedDeviceAdapter() const {
    return deviceAdapterAt(_selectedDeviceIndex);
  }

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
  void setSaveFileUrl(const QUrl &url) { _saveFileUrl = url; }

  QVariantList consoleOverlayEntries() const;
  QVariantList consoleHistoryEntries() const;

  RecorderModel *recorder() const { return _recorderModel; }

  Q_INVOKABLE QVariantMap foldedPropertyPaths() const {
    return _foldedPropertyPaths;
  }

  Q_INVOKABLE void setFoldedProperty(const QString &path, bool folded) {
    _foldedPropertyPaths[path] = folded;
    foldedPropertyPathsChanged();
  }

  void setImageProvider(CameraImageProvider *provider);

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

  void deviceAdded();
  void deviceDeleted();

  void selectedOperatorAdapterChanged();

  void consoleOverlayEntriesChanged();
  void consoleHistoryEntriesChanged();

  void foldedPropertyPathsChanged();

  void recorderChanged();

private:
  pc::Workspace &_workspace;

  CameraImageProvider *_imageProvider = nullptr;

  // TODO raw pointer and new? really?
  QUndoStack *_undoStack = new QUndoStack(this);

  QList<QObject *> _sessionAdapters;
  QList<QObject *> _deviceAdapters;

  QPointer<OperatorAdapter> _selectedOperatorAdapter;

  RecorderModel *_recorderModel;

  // Tracks whether an existing SessionConfigurationAdapter is still bound to
  // a valid underlying SessionConfiguration object address.
  QHash<QString, const pc::SessionConfiguration *> _sessionConfigPtrById;

  // Currently selected (focused) session, and the per-session operator
  // adapters wrapping each session's live OperatorPlugin instances.
  QString _selectedSessionId;
  QHash<QString, QList<OperatorAdapter *>> _sessionOperatorAdapters;

  QVariantMap _foldedPropertyPaths;

  int _selectedDeviceIndex = 0;
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

  // Session operator adapters (analogous to attach/init for devices).
  void attachSessionOperatorConfigAdapters(const QString &sessionId,
                                           const QList<OperatorAdapter *> &ops);
  void initSessionOperatorAdapter(OperatorAdapter *opAdapter,
                                  const QString &sessionId);
  void rebuildSessionOperatorAdapters(const QString &sessionId,
                                      pc::SessionConfigurationAdapter *adapter,
                                      pc::Session *session);
  void syncSessionOperatorAdapters();

  DeviceAdapter *makeDeviceAdapterForPlugin(
      pc::devices::DevicePlugin *plugin,
      pc::devices::DeviceConfigurationVariant &config_variant);
};

} // namespace pc::ui