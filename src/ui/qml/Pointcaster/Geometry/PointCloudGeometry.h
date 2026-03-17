#pragma once

#include <QByteArray>
#include <QObject>
#include <QQuick3DGeometry>
#include <QtQmlIntegration/qqmlintegration.h>
#include <functional>
#include <optional>
#include <pointcaster/point_cloud.h>

#include <ui/models/point_cloud_adapter.h>

namespace pc::ui::qml {

class PointCloudGeometry : public QQuick3DGeometry {
  Q_OBJECT
  QML_NAMED_ELEMENT(PointCloudGeometry)

  Q_PROPERTY(PointCloudAdapter *pointCloudAdapter READ pointCloudAdapter WRITE
                 setPointCloudAdapter NOTIFY pointCloudAdapterChanged)

public:
  PointCloudGeometry();

  PointCloudAdapter *pointCloudAdapter() const { return _pointCloudAdapter; };
  void setPointCloudAdapter(PointCloudAdapter *adapter) { _pointCloudAdapter = adapter; };

  Q_INVOKABLE void updateGeometry();

signals:
  void pointCloudAdapterChanged();

private:
  PointCloudAdapter *_pointCloudAdapter = nullptr;
  const PointCloud *_lastPointCloud = nullptr;
};

} // namespace pc::ui::qml