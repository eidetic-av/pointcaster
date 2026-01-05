#pragma once

#include "../../plugins/devices/orbbec/orbbec_device_config.h"

#include <QObject>
#include <QVariant>
#include <QUrl>
#include <functional>

namespace pc {
struct Workspace;
}

namespace pc::ui {

class WorkspaceModel : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariant deviceConfigAdapters READ deviceConfigAdapters NOTIFY
                 deviceConfigAdaptersChanged)

public:
  explicit WorkspaceModel(pc::Workspace &workspace, QObject *parent,
                          std::function<void()> quit_callback);

  Q_INVOKABLE void close();

  Q_INVOKABLE void loadFromFile(const QUrl& file);

  QVariant deviceConfigAdapters() const;

  void addOrbbecDeviceAdapter(devices::OrbbecDeviceConfiguration &config);

public slots:
  void rebuildAdapters();

signals:
  void deviceConfigAdaptersChanged();

private:
  pc::Workspace &_workspace;
  QList<QObject *> _deviceConfigAdapters;

  std::function<void()> _quit_callback;
};

} // namespace pc::ui
