#include "PointCloudGeometry.h"
#include <QVector3D>

namespace pc::ui::qml {

PointCloudGeometry::PointCloudGeometry() : QQuick3DGeometry() {
  update();
}

void PointCloudGeometry::updateGeometry() {
  if (!_enabled || !_pointCloudAdapter) return;

  auto cloud = _pointCloudAdapter->point_cloud();
  if (!cloud || cloud->empty()) return;

  auto render_buffer = _pointCloudAdapter->render_data();
  if (!render_buffer || render_buffer->empty()) return;

  clear();
  setVertexData(
      QByteArray(reinterpret_cast<const char *>(render_buffer->data()),
                 static_cast<qsizetype>(render_buffer->size())));
  setStride(12);
  setPrimitiveType(QQuick3DGeometry::PrimitiveType::Points);
  addAttribute(Attribute::PositionSemantic, 0, Attribute::F32Type);

  const auto &bounds = cloud->bounds;
  _boundsMin = QVector3D(bounds.min.x, bounds.min.y, bounds.min.z);
  _boundsMax = QVector3D(bounds.max.x, bounds.max.y, bounds.max.z);
  _boundsCenter = ((_boundsMin + _boundsMax) * 0.5) * 0.1;
  setBounds(_boundsMin, _boundsMax);
  emit boundsChanged();

  update();
}

QVector3D PointCloudGeometry::pointPosition(int index) const {
  QVector3D result;
  if (_pointCloudAdapter) {
    auto cloud = _pointCloudAdapter->point_cloud();
    if (cloud && index < cloud->size()) {
      auto pos = cloud->positions[index];
      result = QVector3D(pos.x * 0.1, pos.y * 0.1, pos.z * 0.1);
    }
  }
  return result;
}

} // namespace pc::ui::qml