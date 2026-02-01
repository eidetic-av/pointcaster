#pragma once

#include "config_adapter.h"
#include "device_adapter.h"
#include "device_status.h"
#include <QObject>
#include <QPointer>
#include <QUndoStack>
#include <QUrl>
#include <QVariant>
#include <functional>
#include <session/session_config_adapter.gen.h>
#include <workspace/workspace_config.h>

namespace pc {
class Workspace;
struct SessionConfiguration;
namespace devices {
class DevicePlugin;
}
} // namespace pc

namespace pc::ui {

class WorkspaceModel : public QObject {
  Q_OBJECT

  Q_PROPERTY(QUndoStack *undoStack READ undoStack CONSTANT)

  // Sessions
  Q_PROPERTY(QList<QObject *> sessionAdapters READ sessionAdapters NOTIFY
                 sessionAdaptersChanged)

  // Devices
  Q_PROPERTY(
      QVariant deviceAdapters READ deviceAdapters NOTIFY deviceAdaptersChanged)
  Q_PROPERTY(QStringList deviceVariantNames READ deviceVariantNames NOTIFY
                 deviceVariantNamesChanged)
  Q_PROPERTY(QVariantList addDeviceMenuEntries READ addDeviceMenuEntries NOTIFY
                 addDeviceMenuEntriesChanged)
  Q_PROPERTY(int selectedDeviceIndex READ selectedDeviceIndex NOTIFY
                 selectedDeviceIndexChanged)

  Q_PROPERTY(QUrl saveFileUrl READ saveFileUrl WRITE setSaveFileUrl)

public:
  explicit WorkspaceModel(pc::Workspace *workspace, QObject *parent,
                          std::function<void()> quit_callback);

  Q_INVOKABLE void close();
  Q_INVOKABLE void loadFromFile(const QUrl &file);
  Q_INVOKABLE void save(bool update_last_session_path = true);

  QUndoStack *undoStack() const { return _undoStack; }

  Q_INVOKABLE void triggerDeviceDiscovery();
  Q_INVOKABLE void addNewDevice(const QString &plugin_name,
                                const QString &target_ip = "");
  Q_INVOKABLE void deleteSelectedDevice();

  QList<QObject *> sessionAdapters() const;
  Q_INVOKABLE QList<QObject *> sessionAdaptersCallable() const {
    return sessionAdapters();
  }

  QVariant deviceAdapters() const;

  QStringList deviceVariantNames() const;
  QVariantList addDeviceMenuEntries() const;

  int selectedDeviceIndex() const { return _selectedDeviceIndex; }
  void setSelectedDeviceIndex(int index) {
    if (_selectedDeviceIndex == index) return;
    _selectedDeviceIndex = index;
    emit selectedDeviceIndexChanged();
  }

  QUrl saveFileUrl() const { return _saveFileUrl; }
  void setSaveFileUrl(const QUrl &url) { _saveFileUrl = url; }

public slots:
  void syncAdapters();

signals:
  void openSaveAsDialog();

  void sessionAdaptersChanged();

  void deviceAdaptersChanged();
  void deviceVariantNamesChanged();
  void addDeviceMenuEntriesChanged();
  void selectedDeviceIndexChanged();

private:
  pc::Workspace &_workspace;

  QUndoStack *_undoStack = new QUndoStack(this);

  QList<QObject *> _sessionAdapters;
  QList<QObject *> _deviceAdapters;

  // Tracks whether an existing SessionConfigurationAdapter is still bound to
  // a valid underlying SessionConfiguration object address.
  QHash<QString, const pc::SessionConfiguration *> _sessionConfigPtrById;

  int _selectedDeviceIndex = 0;
  QUrl _saveFileUrl;

  std::function<void()> _quit_callback;

  // applies a new config and syncs adapters on the UI thread
  void applyWorkspaceConfigAndRebuild(pc::WorkspaceConfiguration new_config);

  // Helpers to get stable ids from adapters without relying on Q_PROPERTY
  // names.
  static QString adapterStableId(ConfigAdapter *adapter);
  static QString adapterStableId(DeviceAdapter *adapter);

  template <typename AdapterT>
  void initDeviceAdapter(AdapterT *adapter, pc::devices::DevicePlugin *plugin);

  DeviceAdapter *makeDeviceAdapterForPlugin(
      pc::devices::DevicePlugin *plugin,
      pc::devices::DeviceConfigurationVariant &config_variant);

  void initSessionAdapter(pc::SessionConfigurationAdapter *adapter);
};

} // namespace pc::ui
