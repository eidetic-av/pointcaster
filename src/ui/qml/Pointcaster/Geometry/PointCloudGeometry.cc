#include "PointCloudGeometry.h"
#include <QVector3D>
#include <cmath>
#include <core/logger/logger.h>
#include <random>
#include <ranges>

pc::ui::qml::PointCloudGeometry::PointCloudGeometry() : QQuick3DGeometry() {
  update();
}

void pc::ui::qml::PointCloudGeometry::updateGeometry() {

  constexpr auto float3_stride = 3 * sizeof(float);
  constexpr auto float4_stride = 4 * sizeof(float);
  constexpr auto point_stride =
      float3_stride + float4_stride + sizeof(float); // pos + col + padding

  // if this geometry has a device plugin assigned to render
  if (_pointCloudAdapter) {
    const auto &inputCloud = _pointCloudAdapter->point_cloud();

    bool localGeometryNeedsUpdating =
        _lastPointCloud == nullptr || _lastPointCloud != &inputCloud;

    localGeometryNeedsUpdating = true;

    if (localGeometryNeedsUpdating) {
      const auto &positions = inputCloud.positions;
      const auto &colors = inputCloud.colors;

      QByteArray v;
      v.resize(point_stride * positions.size());

      constexpr auto mm_to_cm = [](int16_t millimetre_value) -> float {
        return static_cast<float>(millimetre_value) / 10.0f;
      };
      constexpr auto char_to_norm = [](unsigned char color) -> float {
        // the incoming point colour elemented are 0-255 in gamma space but we
        // want 0-1 in linear space
        return std::pow(static_cast<float>(color) / 255.0f, 2.2f);
      };

      // TODO profile this, its probs really expensive with large clouds

      float *p = reinterpret_cast<float *>(v.data());
      for (const auto &[pos, col] : std::views::zip(positions, colors)) {
        *p++ = mm_to_cm(pos.x);
        *p++ = mm_to_cm(pos.y);
        *p++ = mm_to_cm(pos.z);
        *p++ = char_to_norm(col.r);
        *p++ = char_to_norm(col.g);
        *p++ = char_to_norm(col.b);
        *p++ = 1.0f; // alpha
        *p++ = 0;    // padding
      }

      setVertexData(v);

      setStride(point_stride);
      setPrimitiveType(QQuick3DGeometry::PrimitiveType::Points);

      addAttribute(QQuick3DGeometry::Attribute::PositionSemantic, 0,
                   QQuick3DGeometry::Attribute::F32Type);
      addAttribute(QQuick3DGeometry::Attribute::ColorSemantic, float3_stride,
                   QQuick3DGeometry::Attribute::F32Type);

      // TODO calc bounds from position min maxes
      setBounds(QVector3D(-500, -500, 0.0f), QVector3D(+500, +500, 0.0f));

      update();
    }

    _lastPointCloud = &inputCloud;
  }
}