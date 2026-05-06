#include "AlignmentController.h"

#include <logger/logger.h>
#include <registration/registration.h>
#include <util/geometry_utils.h>

#include <exception>
#include <vector>

namespace pc::ui::qml {

AlignmentController::AlignmentController(QObject *parent) : QObject(parent) {}

void AlignmentController::computeFromPairs(const QVariantList &pairs) {
  if (pairs.size() < 3) {
    pc::logger()->error("AlignmentController: need >= 3 pairs");
    return;
  }

  std::vector<pc::float3> source;
  std::vector<pc::float3> target;

  source.reserve(static_cast<std::size_t>(pairs.size()));
  target.reserve(static_cast<std::size_t>(pairs.size()));

  for (const auto &pair : pairs) {
    const auto map = pair.toMap();

    const auto secondary = map["secondary"].value<QVector3D>();
    const auto primary = map["primary"].value<QVector3D>();

    source.push_back(pc::float3{
        secondary.x(),
        secondary.y(),
        secondary.z(),
    });

    target.push_back(pc::float3{
        primary.x(),
        primary.y(),
        primary.z(),
    });
  }

  try {
    const auto transform =
        pc::registration::compute_rigid_transform(source, target);

    decomposeTransform(transform);

    _hasResult = true;

    pc::logger()->info("AlignmentController: computed transform from {} pairs",
                       pairs.size());

    emit resultChanged();
  } catch (const std::exception &e) {
    pc::logger()->error("AlignmentController: {}", e.what());
  }
}

void AlignmentController::decomposeTransform(const pc::float4x4 &matrix) {
  const auto [position, rotation] = pc::decompose_transform(matrix);
  _transform = QMatrix4x4(matrix.values);
  _position = QVector3D(position.x, position.y, position.z);
  _rotation = QQuaternion(rotation.scalar, rotation.x, rotation.y, rotation.z);
}

void AlignmentController::reset() {
  _hasResult = false;
  _transform = QMatrix4x4();
  _position = QVector3D();
  _rotation = QQuaternion();

  emit resultChanged();
}

} // namespace pc::ui::qml