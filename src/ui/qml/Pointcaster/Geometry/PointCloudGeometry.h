#pragma once
#include <QByteArray>
#include <QFutureWatcher>
#include <QObject>
#include <QQuick3DGeometry>
#include <QtQmlIntegration/qqmlintegration.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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

  PointCloudAdapter *pointCloudAdapter() const { return _pointCloudAdapter; };
  void setPointCloudAdapter(PointCloudAdapter *adapter) {
    _pointCloudAdapter = adapter;
  };

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
  void applyVertexData(QByteArray vertexBytes,
                       std::shared_ptr<PointCloud> cloud);
  void startConversion(std::shared_ptr<PointCloud> cloud);

  PointCloudAdapter *_pointCloudAdapter = nullptr;
  bool _enabled = true;

  QFutureWatcher<QByteArray> _conversionWatcher;
  std::atomic<bool> _conversionInFlight{false};
  std::shared_ptr<PointCloud> _lastPointCloud = nullptr;

  std::mutex _pendingMutex;
  std::shared_ptr<PointCloud> _pendingCloud = nullptr; // cloud being converted
  std::shared_ptr<PointCloud> _queuedCloud = nullptr;  // newer cloud waiting
};

} // namespace pc::ui::qml