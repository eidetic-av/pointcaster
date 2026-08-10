#include "StreamInstancing.h"

#include <QVector3D>
#include <ranges>
#include <utility>

namespace pc::ui::qml {

namespace {

// positions are in millimetres and the scene is in centimetres, the same
// conversion the point cloud shader does before it places a vertex
constexpr float scene_scale = 0.1f;

// and the built-in meshes an instance draws are 100 units across
constexpr float mesh_size = 100.0f;

QVector3D to_scene(const pc::position &p) {
  return QVector3D(p.x, p.y, p.z) * scene_scale;
}

QColor to_qcolor(const pc::color &c) {
  return QColor::fromRgb(c.r, c.g, c.b, c.a);
}

} // namespace

void StreamInstancing::setStreamAdapter(StreamAdapter *adapter) {
  if (_streamAdapter == adapter) return;
  _streamAdapter = adapter;
  emit streamAdapterChanged();
  updateInstances();
}

void StreamInstancing::setColor(const QColor &color) {
  if (_color == color) return;
  _color = color;
  emit colorChanged();
  updateInstances();
}

void StreamInstancing::updateInstances() {
  // resize rather than clear so the buffer keeps the capacity it reached
  _instanceData.resize(0);
  const auto previous_count = std::exchange(_instanceCount, 0);

  const auto add_instance = [&](const QVector3D &centre, const QVector3D &size,
                                const QColor &colour) {
    const auto instance =
        calculateTableEntry(centre, size / mesh_size, {}, colour);
    _instanceData.append(reinterpret_cast<const char *>(&instance),
                         sizeof(instance));
    _instanceCount++;
  };

  if (const auto aabbs =
          _streamAdapter ? _streamAdapter->aabb_list() : nullptr) {
    // an instance spanning each aabb's min and max corners, in the colour the
    // operator gave it to tell it apart from the others
    const auto boxes = std::views::zip(
        aabbs->min_positions(), aabbs->max_positions(), aabbs->colors);

    for (const auto &[min, max, colour] : boxes) {
      const auto min_corner = to_scene(min);
      const auto max_corner = to_scene(max);
      add_instance((min_corner + max_corner) / 2.0f, max_corner - min_corner,
                   to_qcolor(colour));
    }
  } else if (const auto voxels = _streamAdapter
                                     ? _streamAdapter->voxelised_cloud()
                                     : nullptr) {
    // or one the size of the grid centred on each occupied voxel. a voxel has
    // no identity to key a colour off, so the whole grid takes the flat one
    const auto voxel_size =
        QVector3D(1, 1, 1) * float(voxels->voxel_size) * scene_scale;

    for (const auto &voxel : voxels->positions) {
      add_instance(to_scene(voxel), voxel_size, _color);
    }
  }

  if (_instanceCount != previous_count) emit instanceCountChanged();
  markDirty();
}

QByteArray StreamInstancing::getInstanceBuffer(int *instanceCount) {
  if (instanceCount) *instanceCount = _instanceCount;
  return _instanceData;
}

} // namespace pc::ui::qml
