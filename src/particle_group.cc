#include "particle_group.h"

#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/Assert.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Trade/MeshData.h>

#include <spdlog/spdlog.h>

namespace bob {

using namespace Magnum;
using namespace Math::Literals;

ParticleGroup::ParticleGroup(const std::vector<Vector3> &points,
                             float particleRadius)
    : _points(points), _particleRadius(particleRadius),
      _meshParticles(GL::MeshPrimitive::Points) {
  _meshParticles.addVertexBuffer(_bufferParticles, 0,
				 Shaders::Generic3D::Position{});
  _particleShader.reset(new ParticleSphereShader);
}

ParticleGroup &
ParticleGroup::draw(Containers::Pointer<SceneGraph::Camera3D> &camera,
		    const Vector2i &viewportSize) {
  if (_points.empty()) return *this;

  if (_dirty) {
    Containers::ArrayView<const float> data(
	reinterpret_cast<const float *>(&_points[0]), _points.size() * 3);
    _bufferParticles.setData(data);
    _meshParticles.setCount(static_cast<int>(_points.size()));
    _dirty = false;
  }

  (*_particleShader)
      /* particle data */
      .setNumParticles(static_cast<int>(_points.size()))
      .setParticleRadius(_particleRadius)
      /* sphere render data */
      .setPointSizeScale(
	  static_cast<float>(viewportSize.y()) /
	  Math::tan(22.5_degf)) /* tan(half field-of-view angle (45_deg)*/
      .setColorMode(_colorMode)
      .setAmbientColor(_ambientColor)
      .setDiffuseColor(_diffuseColor)
      .setSpecularColor(_specularColor)
      .setShininess(_shininess)
      /* view/prj matrices and light */
      .setViewMatrix(camera->cameraMatrix())
      .setProjectionMatrix(camera->projectionMatrix())
      .setLightDirection(_lightDir)
      .draw(_meshParticles);

  return *this;
}

} // namespace bob
