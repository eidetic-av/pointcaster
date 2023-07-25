#include "point_cloud_renderer.h"

#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/Assert.h>
#include <Magnum/Magnum.h>
#include <Magnum/GL/Attribute.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Trade/MeshData.h>

#include <spdlog/spdlog.h>

namespace pc {

using namespace Magnum;
using namespace Shaders;
using namespace Math::Literals;

PointCloudRenderer::PointCloudRenderer(float particleRadius)
    : _particleRadius(particleRadius),
      _meshParticles(GL::MeshPrimitive::Points) {
  points = pc::types::PointCloud{};
  _meshParticles.addVertexBuffer(
      _positions_buffer, 0,
      Generic3D::Position{Generic3D::Position::Components::Two,
			  Generic3D::Position::DataType::Int});
  _meshParticles.addVertexBuffer(_color_buffer, 0, GL::Attribute<2, float>());
  _particleShader.reset(new ParticleSphereShader);
  setDirty();
}

PointCloudRenderer &
PointCloudRenderer::draw(Magnum::SceneGraph::Camera3D& camera,
		    const Vector2i &viewportSize) {
  if (points.empty()) return *this;

  if (_dirty) {
    Containers::ArrayView<const int64_t> position_data(
	reinterpret_cast<const int64_t *>(&points.positions[0]), points.size());
    _positions_buffer.setData(position_data);

    Containers::ArrayView<const float> color_data(
	reinterpret_cast<const float *>(&points.colors[0]), points.size());
    _color_buffer.setData(color_data);

    _meshParticles.setCount(static_cast<int>(points.size()));
    _dirty = false;
  }

  (*_particleShader)
      /* particle data */
      .setParticleRadius(_particleRadius)
      /* sphere render data */
      .setPointSizeScale(
	  static_cast<float>(viewportSize.y()) /
	  Math::tan(22.5_degf)) /* tan(half field-of-view angle (45_deg)*/
      /* view/prj matrices and light */
      .setViewMatrix(camera.cameraMatrix())
      .setProjectionMatrix(camera.projectionMatrix())
      .draw(_meshParticles);

  return *this;
}

} // namespace pc
