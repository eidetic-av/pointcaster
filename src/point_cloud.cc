#include "point_cloud.h"

#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/Assert.h>
#include <Magnum/GL/Attribute.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Trade/MeshData.h>

#include <spdlog/spdlog.h>

namespace bob {

using namespace Magnum;
using namespace Math::Literals;

PointCloudRenderer::PointCloudRenderer(float particleRadius)
    : _particleRadius(particleRadius),
      _meshParticles(GL::MeshPrimitive::Points) {
  _meshParticles.addVertexBuffer(_positions_buffer, 0, GL::Attribute<0, Vector4>());
  _meshParticles.addVertexBuffer(_color_buffer, 0, GL::Attribute<2, float>());
  _particleShader.reset(new ParticleSphereShader);
}

PointCloudRenderer &
PointCloudRenderer::draw(Containers::Pointer<SceneGraph::Camera3D> &camera,
		    const Vector2i &viewportSize) {
  if (_points.empty()) return *this;

  if (_dirty) {
    Containers::ArrayView<const float> position_data(
	reinterpret_cast<const float *>(&_points.positions[0]), _points.size() * 3);
    _positions_buffer.setData(position_data);

    Containers::ArrayView<const float> color_data(
	reinterpret_cast<const float *>(&_points.colors[0]), _points.size());
    _color_buffer.setData(color_data);

    _meshParticles.setCount(static_cast<int>(_points.size()));
    _dirty = false;
  }

  // spdlog::info(_points.size());

  (*_particleShader)
      /* particle data */
      .setParticleRadius(_particleRadius)
      /* sphere render data */
      .setPointSizeScale(
	  static_cast<float>(viewportSize.y()) /
	  Math::tan(22.5_degf)) /* tan(half field-of-view angle (45_deg)*/
      /* view/prj matrices and light */
      .setViewMatrix(camera->cameraMatrix())
      .setProjectionMatrix(camera->projectionMatrix())
      .draw(_meshParticles);
    ;

  return *this;
}

} // namespace bob
