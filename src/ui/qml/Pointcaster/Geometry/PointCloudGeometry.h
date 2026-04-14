#pragma once
#include <QObject>
#include <QQuick3DGeometry>
#include <QtQmlIntegration/qqmlintegration.h>
#include <memory>
#include <pointcaster/point_cloud.h>
#include <ui/models/point_cloud_adapter.h>

namespace pc::ui::qml {

class PointCloudGeometry : public QQuick3DGeometry {
  Q_OBJECT
  QML_NAMED_ELEMENT(PointCloudGeometry)
  Q_PROPERTY(PointCloudAdapter *pointCloudAdapter READ pointCloudAdapter WRITE
                 setPointCloudAdapter NOTIFY pointCloudAdapterChanged)
  Q_PROPERTY(bool enabled READ enabled WRITE setEnabled NOTIFY enabledChanged)

public:
  PointCloudGeometry();

  PointCloudAdapter *pointCloudAdapter() const { return _pointCloudAdapter; }
  void setPointCloudAdapter(PointCloudAdapter *adapter) {
    _pointCloudAdapter = adapter;
  }

  bool enabled() const { return _enabled; }
  void setEnabled(bool enabled) {
    if (_enabled == enabled) return;
    _enabled = enabled;
    emit enabledChanged();
  }

  Q_INVOKABLE void updateGeometry();

signals:
  void pointCloudAdapterChanged();
  void enabledChanged();

private:
  PointCloudAdapter *_pointCloudAdapter = nullptr;
  bool _enabled = true;
};

} // namespace pc::ui::qml