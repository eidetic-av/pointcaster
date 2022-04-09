#include "particle_sphere.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Shaders/Generic.h>

namespace bob::shaders {

using namespace Magnum;

ParticleSphereShader::ParticleSphereShader() {
  Utility::Resource rs("data");

  GL::Shader vertShader{GL::Version::GL330, GL::Shader::Type::Vertex};
  GL::Shader fragShader{GL::Version::GL330, GL::Shader::Type::Fragment};
  vertShader.addSource(rs.get("particle_sphere.vert"));
  fragShader.addSource(rs.get("particle_sphere.frag"));

  CORRADE_INTERNAL_ASSERT(GL::Shader::compile({vertShader, fragShader}));
  attachShaders({vertShader, fragShader});
  CORRADE_INTERNAL_ASSERT(link());

  _uParticleRadius = uniformLocation("particleRadius");
  _uPointSizeScale = uniformLocation("pointSizeScale");
  _uViewMatrix = uniformLocation("viewMatrix");
  _uProjectionMatrix = uniformLocation("projectionMatrix");
}

ParticleSphereShader &ParticleSphereShader::setParticleRadius(Float radius) {
  setUniform(_uParticleRadius, radius);
  return *this;
}

ParticleSphereShader &ParticleSphereShader::setPointSizeScale(Float scale) {
  setUniform(_uPointSizeScale, scale);
  return *this;
}

ParticleSphereShader &
ParticleSphereShader::setViewMatrix(const Matrix4 &matrix) {
  setUniform(_uViewMatrix, matrix);
  return *this;
}

ParticleSphereShader &
ParticleSphereShader::setProjectionMatrix(const Matrix4 &matrix) {
  setUniform(_uProjectionMatrix, matrix);
  return *this;
}

} // namespace bob::shaders
