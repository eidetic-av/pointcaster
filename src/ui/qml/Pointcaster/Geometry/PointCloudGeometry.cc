#include "PointCloudGeometry.h"
#include <QVector3D>
#include <QtConcurrent>
#include <algorithm>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <cstring>
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

    const auto count = static_cast<int>(cloud->size());
    constexpr int kPosSize = 8; // position: int16 x,y,z + pad = 8 bytes
    constexpr int kColSize = 4; // color: uint8 r,g,b,a = 4 bytes
    constexpr int kStride = kPosSize + kColSize; // 12 bytes per vertex

    QByteArray vertexBytes;
    vertexBytes.resize(kStride * count);

    auto *dst = vertexBytes.data();
    const auto *pos = reinterpret_cast<const char *>(cloud->positions.data());
    const auto *col = reinterpret_cast<const char *>(cloud->colors.data());

    // byte interleaving because the qt quick 3d api only accepts a single vertex buffer...
    // the shader unpacks these into pos and color 
    auto indices = std::views::iota(0, count);
    std::for_each(
        std::execution::par_unseq, indices.begin(), indices.end(), [=](int i) {
          std::memcpy(dst + i * kStride, pos + i * kPosSize, kPosSize);
          std::memcpy(dst + i * kStride + kPosSize, col + i * kColSize,
                      kColSize);
        });

    return vertexBytes;
  }));
}

void PointCloudGeometry::applyVertexData(QByteArray vertexBytes,
                                         std::shared_ptr<PointCloud> cloud) {
  clear();
  setVertexData(vertexBytes);
  setStride(12);
  setPrimitiveType(QQuick3DGeometry::PrimitiveType::Points);
  // 3 × F32Type = 12 bytes per vertex...
  // the shader reinterprets the raw bits via floatBitsToInt/floatBitsToUint.
  addAttribute(Attribute::PositionSemantic, 0, Attribute::F32Type);
  setBounds(QVector3D(-500, -500, -500), QVector3D(500, 500, 500));
  update();
  _lastPointCloud = cloud;
}

} // namespace pc::ui::qml