#include "PointCloudGeometry.h"
#include <QVector3D>
#include <algorithm>
#include <cmath>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <execution>
#include <ranges>

using pc::profiling::ProfilingZone;

pc::ui::qml::PointCloudGeometry::PointCloudGeometry() : QQuick3DGeometry() {
  update();
}

void pc::ui::qml::PointCloudGeometry::updateGeometry() {

  struct FlatVertexData {
    float x, y, z, _p;
    float r, g, b, a;
  };
  constexpr auto vertex_stride = sizeof(FlatVertexData);
  constexpr auto float4_stride = sizeof(float) * 4;

  // if this geometry has a device plugin assigned to render
  if (_pointCloudAdapter) {
    const auto inputCloud = _pointCloudAdapter->point_cloud();

    bool localGeometryNeedsUpdating =
        _lastPointCloud == nullptr || _lastPointCloud != inputCloud;

    localGeometryNeedsUpdating = true;

    if (localGeometryNeedsUpdating) {

      ProfilingZone update_geometry_zone("PointCloudGeometry::updateGeometry");

      const auto &positions = inputCloud->positions;
      const auto &colors = inputCloud->colors;
      const auto indices =
          std::views::iota(0, static_cast<int>(inputCloud->size()));

      const auto points = std::views::zip(indices, positions, colors);

      QByteArray vertexBytes;
      vertexBytes.resize(vertex_stride * points.size());

      auto *vertexData = reinterpret_cast<FlatVertexData *>(vertexBytes.data());

      constexpr auto mm_to_cm = [](int16_t millimetre_value) -> float {
        // we store point positions as mm shorts but we want floats in centimetres for Qt
        return static_cast<float>(millimetre_value) / 10.0f;
      };
      constexpr auto char_to_norm = [](unsigned char color) -> float {
        // the incoming point colour elements are 0-255 in gamma space
        // but we want 0-1 in linear colour space
        return std::pow(static_cast<float>(color) / 255.0f, 2.2f);
      };

      std::for_each(std::execution::par_unseq, points.begin(), points.end(),
                    [&vertexData](const auto &p) {
                      const auto &[i, pos, col] = p;
                      vertexData[i] = {.x = mm_to_cm(pos.x),
                                       .y = mm_to_cm(pos.y),
                                       .z = mm_to_cm(pos.z),
                                       ._p = 0,
                                       .r = char_to_norm(col.r),
                                       .g = char_to_norm(col.g),
                                       .b = char_to_norm(col.b),
                                       .a = 1};
                    });

      setVertexData(vertexBytes);

      setStride(vertex_stride);
      setPrimitiveType(QQuick3DGeometry::PrimitiveType::Points);

      addAttribute(QQuick3DGeometry::Attribute::PositionSemantic, 0,
                   QQuick3DGeometry::Attribute::F32Type);

      addAttribute(QQuick3DGeometry::Attribute::ColorSemantic, float4_stride,
                   QQuick3DGeometry::Attribute::F32Type);

      // TODO calc bounds from position min maxes
      setBounds(QVector3D(-500, -500, 0.0f), QVector3D(+500, +500, 0.0f));

      update();
    }

    _lastPointCloud = inputCloud;
  }
}