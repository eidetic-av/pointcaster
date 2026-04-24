#include "PointCloudGeometry.h"
#include <QVector3D>

namespace pc::ui::qml {

PointCloudGeometry::PointCloudGeometry() : QQuick3DGeometry() {
  update();
}

void PointCloudGeometry::updateGeometry() {
  if (!_enabled || !_pointCloudAdapter) return;

  auto render_buffer = _pointCloudAdapter->render_data();
  if (!render_buffer || render_buffer->empty()) return;

  clear();
  setVertexData(
      QByteArray(reinterpret_cast<const char *>(render_buffer->data()),
                 static_cast<qsizetype>(render_buffer->size())));
  setStride(12);
  setPrimitiveType(QQuick3DGeometry::PrimitiveType::Points);
  addAttribute(Attribute::PositionSemantic, 0, Attribute::F32Type);

  setBounds(QVector3D(-1e10, -1e10, -1e10), QVector3D(1e10, 1e10, 1e10));
  update();
}

} // namespace pc::ui::qml