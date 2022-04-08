#include "particle_sphere.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>

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

  _uNumParticles = uniformLocation("numParticles");
  _uParticleRadius = uniformLocation("particleRadius");

  _uPointSizeScale = uniformLocation("pointSizeScale");
  _uColorMode = uniformLocation("colorMode");
  _uAmbientColor = uniformLocation("ambientColor");
  _uDiffuseColor = uniformLocation("diffuseColor");
  _uSpecularColor = uniformLocation("specularColor");
  _uShininess = uniformLocation("shininess");

  _uViewMatrix = uniformLocation("viewMatrix");
  _uProjectionMatrix = uniformLocation("projectionMatrix");
  _uLightDir = uniformLocation("lightDir");
}

ParticleSphereShader &ParticleSphereShader::setNumParticles(Int numParticles) {
  setUniform(_uNumParticles, numParticles);
  return *this;
}

ParticleSphereShader &ParticleSphereShader::setParticleRadius(Float radius) {
  setUniform(_uParticleRadius, radius);
  return *this;
}

ParticleSphereShader &ParticleSphereShader::setPointSizeScale(Float scale) {
  setUniform(_uPointSizeScale, scale);
  return *this;
}

ParticleSphereShader &ParticleSphereShader::setColorMode(Int colorMode) {
  setUniform(_uColorMode, colorMode);
  return *this;
}

ParticleSphereShader &
ParticleSphereShader::setAmbientColor(const Color3 &color) {
  setUniform(_uAmbientColor, color);
  return *this;
}

ParticleSphereShader &
ParticleSphereShader::setDiffuseColor(const Color3 &color) {
  setUniform(_uDiffuseColor, color);
  return *this;
}

ParticleSphereShader &
ParticleSphereShader::setSpecularColor(const Color3 &color) {
  setUniform(_uSpecularColor, color);
  return *this;
}

ParticleSphereShader &ParticleSphereShader::setShininess(Float shininess) {
  setUniform(_uShininess, shininess);
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

ParticleSphereShader &
ParticleSphereShader::setLightDirection(const Vector3 &lightDir) {
  setUniform(_uLightDir, lightDir);
  return *this;
}

} // namespace bob::shaders
