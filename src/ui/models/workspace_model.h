#pragma once

#include "../../plugins/devices/orbbec/orbbec_device_config.h"

#include <QObject>
#include <QVariant>

namespace pc {
struct Workspace;
}

namespace pc::ui {

class WorkspaceModel : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariant deviceConfigAdapters READ deviceConfigAdapters NOTIFY
                 deviceConfigAdaptersChanged)

public:
  explicit WorkspaceModel(pc::Workspace &workspace, QObject *parent = nullptr);

  QVariant deviceConfigAdapters() const;

  void addOrbbecDeviceAdapter(devices::OrbbecDeviceConfiguration &config);

signals:
  void deviceConfigAdaptersChanged();

private:
  pc::Workspace &_workspace;
  QList<QObject *> _deviceConfigAdapters;

  void rebuildAdapters();
};

} // namespace pc::ui
