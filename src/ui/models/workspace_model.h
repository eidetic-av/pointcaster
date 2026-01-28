#pragma once

#include "config_adapter.h"
#include "device_status.h"
#include <QObject>
#include <QUndoStack>
#include <QUrl>
#include <QVariant>
#include <functional>

namespace pc {
struct Workspace;
}

namespace pc::ui {

class WorkspaceModel : public QObject {
  Q_OBJECT

  Q_PROPERTY(QUndoStack* undoStack READ undoStack CONSTANT)

  Q_PROPERTY(QVariant deviceConfigAdapters READ deviceConfigAdapters NOTIFY
                 deviceConfigAdaptersChanged)
  Q_PROPERTY(QList<QObject *> deviceControllers READ deviceControllers NOTIFY
                 deviceControllersChanged)
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

  QUndoStack* undoStack() const { return _undoStack; }

  Q_INVOKABLE void triggerDeviceDiscovery();
  Q_INVOKABLE void addNewDevice(const QString &plugin_name,
                                const QString &target_ip = "");
  Q_INVOKABLE void deleteSelectedDevice();

  QVariant deviceConfigAdapters() const;
  const QList<QObject *> &deviceControllers() const {
    return _deviceControllers;
  }

  QStringList deviceVariantNames() const;
  QVariantList addDeviceMenuEntries() const;

  int selectedDeviceIndex() const { return _selectedDeviceIndex; }
  void setSelectedDeviceIndex(int index) {
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

  void deviceConfigAdaptersChanged();
  void deviceControllersChanged();
  void deviceVariantNamesChanged();
  void addDeviceMenuEntriesChanged();
  void selectedDeviceIndexChanged();

private:
  pc::Workspace &_workspace;

  QUndoStack* _undoStack = new QUndoStack(this);

  QList<QObject *> _deviceConfigAdapters;
  QList<QObject *> _deviceControllers;

  int _selectedDeviceIndex = 0;
  QUrl _saveFileUrl;

  std::function<void()> _quit_callback;

  void onAdapterFieldEditRequested(ConfigAdapter *adapter, int fieldIndex,
                                   const QVariant &value);
};

} // namespace pc::ui
