#pragma once

#include "device_status.h"
#include <QObject>
#include <QUrl>
#include <QVariant>
#include <functional>

namespace pc {
struct Workspace;
}

namespace pc::ui {

class WorkspaceModel : public QObject {
  Q_OBJECT

  Q_PROPERTY(QVariant deviceConfigAdapters READ deviceConfigAdapters NOTIFY
                 deviceConfigAdaptersChanged)
  Q_PROPERTY(QList<QObject *> deviceControllers READ deviceControllers NOTIFY
                 deviceControllersChanged)
  Q_PROPERTY(QUrl saveFileUrl READ saveFileUrl WRITE setSaveFileUrl)

public:
  explicit WorkspaceModel(pc::Workspace *workspace, QObject *parent,
                          std::function<void()> quit_callback);

  Q_INVOKABLE void close();

  Q_INVOKABLE void loadFromFile(const QUrl &file);
  Q_INVOKABLE void save();

  QVariant deviceConfigAdapters() const;
  const QList<QObject *> &deviceControllers() const {
    return _deviceControllers;
  }

  QUrl saveFileUrl() const { return _saveFileUrl; };
  void setSaveFileUrl(const QUrl &url) { _saveFileUrl = url; }

public slots:
  void rebuildAdapters();

signals:
  void openSaveAsDialog();
  void deviceConfigAdaptersChanged();
  void deviceControllersChanged();

private:
  pc::Workspace &_workspace;
  QList<QObject *> _deviceConfigAdapters;
  QList<QObject *> _deviceControllers;

  QUrl _saveFileUrl;

  std::function<void()> _quit_callback;
};

} // namespace pc::ui
