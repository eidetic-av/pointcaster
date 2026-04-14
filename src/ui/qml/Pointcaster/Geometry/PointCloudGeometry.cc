#include "PointCloudGeometry.h"
#include <QVector3D>
#include <QtConcurrent>
#include <algorithm>
#include <cmath>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <execution>
#include <ranges>

using pc::profiling::ProfilingZone;

namespace pc::ui::qml {

PointCloudGeometry::PointCloudGeometry() : QQuick3DGeometry() {
  connect(&_conversionWatcher, &QFutureWatcher<QByteArray>::finished, this,
          [this]() {
            std::shared_ptr<PointCloud> converted;
            std::shared_ptr<PointCloud> queued;
            {
              std::lock_guard lock(_pendingMutex);
              converted = _pendingCloud;
              queued = _queuedCloud;
              _queuedCloud = nullptr;
            }

            applyVertexData(_conversionWatcher.result(), converted);
            _conversionInFlight.store(false);

            // if newer data arrived while we were converting, immediately
            // kick off another conversion with the latest cloud
            if (queued && queued != converted) {
              startConversion(queued);
            }
          });
  update();
}

void PointCloudGeometry::updateGeometry() {
  if (!_enabled) return;
  if (!_pointCloudAdapter) return;

  const auto inputCloud = _pointCloudAdapter->point_cloud();

  if (_conversionInFlight.load()) {
    // conversion running — stash this as the next one to process
    std::lock_guard lock(_pendingMutex);
    _queuedCloud = inputCloud;
    return;
  }

  startConversion(inputCloud);
}

void PointCloudGeometry::startConversion(std::shared_ptr<PointCloud> cloud) {
  _conversionInFlight.store(true);
  {
    std::lock_guard lock(_pendingMutex);
    _pendingCloud = cloud;
    _queuedCloud = nullptr;
  }

  _conversionWatcher.setFuture(QtConcurrent::run([cloud]() -> QByteArray {
    ProfilingZone zone("PointCloudGeometry::conversion");

    struct FlatVertexData {
      float x, y, z, _p, r, g, b, a;
    };
    constexpr auto vertex_stride = sizeof(FlatVertexData);

    const auto &positions = cloud->positions;
    const auto &colors = cloud->colors;
    const auto indices = std::views::iota(0, static_cast<int>(cloud->size()));
    const auto points = std::views::zip(indices, positions, colors);

    QByteArray vertexBytes;
    vertexBytes.resize(vertex_stride * points.size());
    auto *vertexData = reinterpret_cast<FlatVertexData *>(vertexBytes.data());

    constexpr auto mm_to_cm = [](int16_t v) -> float {
      return static_cast<float>(v) / 10.0f;
    };
    constexpr auto char_to_norm = [](unsigned char c) -> float {
      return std::pow(static_cast<float>(c) / 255.0f, 2.2f);
    };

    // TODO this needs a thrust or shader-based path for GPU

    std::for_each(std::execution::par_unseq, points.begin(), points.end(),
                  [&](const auto &p) {
                    const auto &[i, pos, col] = p;
                    vertexData[i] = {mm_to_cm(pos.x),     mm_to_cm(pos.y),
                                     mm_to_cm(pos.z),     0,
                                     char_to_norm(col.r), char_to_norm(col.g),
                                     char_to_norm(col.b), 1};
                  });

    return vertexBytes;
  }));
}

void PointCloudGeometry::applyVertexData(QByteArray vertexBytes,
                                         std::shared_ptr<PointCloud> cloud) {
  constexpr auto vertex_stride = sizeof(float) * 8;
  constexpr auto float4_stride = sizeof(float) * 4;

  setVertexData(vertexBytes);

  setStride(vertex_stride);
  setPrimitiveType(QQuick3DGeometry::PrimitiveType::Points);

  addAttribute(Attribute::PositionSemantic, 0, Attribute::F32Type);
  addAttribute(Attribute::ColorSemantic, float4_stride, Attribute::F32Type);

  // TODO calc bounds from position min maxes
  setBounds(QVector3D(-500, -500, 0), QVector3D(500, 500, 0));

  update();
  _lastPointCloud = cloud;
}

} // namespace pc::ui::qml