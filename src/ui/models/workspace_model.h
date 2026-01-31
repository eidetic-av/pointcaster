#pragma once

#include "config_adapter.h"
#include "device_adapter.h"
#include "device_status.h"
#include <QObject>
#include <QUndoStack>
#include <QUrl>
#include <QVariant>
#include <functional>

namespace pc {
class Workspace;
}

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

  // File IO
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
  void rebuildAdapters();
  void refreshDeviceAdapterFromCore(int deviceIndex);

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

  int _selectedDeviceIndex = 0;
  QUrl _saveFileUrl;

  std::function<void()> _quit_callback;

  void onAdapterFieldEditRequested(DeviceAdapter *adapter, int fieldIndex,
                                   const QVariant &value);
};

} // namespace pc::ui
