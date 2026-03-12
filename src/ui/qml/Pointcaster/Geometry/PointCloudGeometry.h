#pragma once

#include <QByteArray>
#include <QObject>
#include <QQuick3DGeometry>
#include <QtQmlIntegration/qqmlintegration.h>
#include <functional>
#include <optional>
#include <pointcaster/point_cloud.h>

#include <ui/models/device_adapter.h>

namespace pc::ui::qml {

class PointCloudGeometry : public QQuick3DGeometry {
  Q_OBJECT
  QML_NAMED_ELEMENT(PointCloudGeometry)

  // TODO the device stuff being here means it probs should
  // be a specialised class, not inside generic point cloud geo
  Q_PROPERTY(DeviceAdapter *deviceAdapter READ deviceAdapter WRITE
                 setDeviceAdapter NOTIFY deviceAdapterChanged)

public:
  PointCloudGeometry();

  DeviceAdapter *deviceAdapter() const { return _deviceAdapter; };
  void setDeviceAdapter(DeviceAdapter *adapter) { _deviceAdapter = adapter; };

  Q_INVOKABLE void updateGeometry();

signals:
  void deviceAdapterChanged();

private:
  DeviceAdapter *_deviceAdapter = nullptr;
  const PointCloud *_lastPointCloud = nullptr;
};

} // namespace pc::ui::qml