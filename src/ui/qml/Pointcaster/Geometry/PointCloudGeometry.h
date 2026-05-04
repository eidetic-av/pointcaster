#pragma once
#include <QObject>
#include <QQuick3DGeometry>
#include <QVector3D>
#include <QtQmlIntegration/qqmlintegration.h>
#include <limits>
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

  Q_PROPERTY(QVector3D boundsMin READ boundsMin NOTIFY boundsChanged)
  Q_PROPERTY(QVector3D boundsMax READ boundsMax NOTIFY boundsChanged)
  Q_PROPERTY(QVector3D boundsCenter READ boundsCenter NOTIFY boundsChanged)

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

  QVector3D boundsMin() const { return _boundsMin; }
  QVector3D boundsMax() const { return _boundsMax; }
  QVector3D boundsCenter() const { return _boundsCenter; }

  Q_INVOKABLE void updateGeometry();

  Q_INVOKABLE QVector3D pointPosition(int index) const;

signals:
  void pointCloudAdapterChanged();
  void enabledChanged();
  void boundsChanged();

private:
  PointCloudAdapter *_pointCloudAdapter = nullptr;
  bool _enabled = true;

  QVector3D _boundsMin;
  QVector3D _boundsMax;
  QVector3D _boundsCenter;
};

} // namespace pc::ui::qml