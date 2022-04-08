#include <vector>
#include <Corrade/Containers/Pointer.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Camera.h>

#include "shaders/particle_sphere.h"

namespace bob {

  using namespace Magnum;
  using namespace bob::shaders;

  class PointCloud {
  public:
    explicit PointCloud(const std::vector<Vector3>& points, float particleRadius);

    PointCloud& draw(Containers::Pointer<SceneGraph::Camera3D>& camera, const Vector2i& viewportSize);

    bool isDirty() const { return _dirty; }

    PointCloud& setDirty() {
      _dirty = true;
      return *this;
    }

    Float particleRadius() const { return _particleRadius; }

    PointCloud& setParticleRadius(Float radius) {
      _particleRadius = radius;
      return *this;
    }

    ParticleSphereShader::ColorMode colorMode() const { return _colorMode; }

    PointCloud& setColorMode(ParticleSphereShader::ColorMode colorMode) {
      _colorMode = colorMode;
      return *this;
    }

    Color3 ambientColor() const { return _ambientColor; }

    PointCloud& setAmbientColor(const Color3& color) {
      _ambientColor = color;
      return *this;
    }

    Color3 diffuseColor() const { return _diffuseColor; }

    PointCloud& setDiffuseColor(const Color3& color) {
      _diffuseColor = color;
      return *this;
    }

    Color3 specularColor() const { return _specularColor; }

    PointCloud& setSpecularColor(const Color3& color) {
      _specularColor = color;
      return *this;
    }

    Float shininess() const { return _shininess; }

    PointCloud& setShininess(Float shininess) {
      _shininess = shininess;
      return *this;
    }

    Vector3 lightDirection() const { return _lightDir; }

    PointCloud& setLightDirection(const Vector3& lightDir) {
      _lightDir = lightDir;
      return *this;
    }

    PointCloud& setPoints(const std::vector<Vector3>& points) {
      _points = points;
      return *this;
    }

  private:
    const std::vector<Vector3>& _points;

    bool _dirty = false;

    Float _particleRadius = 1.0f;
    ParticleSphereShader::ColorMode _colorMode = ParticleSphereShader::ColorMode::RampColorById;
    Color3 _ambientColor{0.1f};
    Color3 _diffuseColor{0.0f, 0.5f, 0.9f};
    Color3 _specularColor{ 1.0f};
    Float _shininess = 150.0f;
    Vector3 _lightDir{1.0f, 1.0f, 2.0f};

    GL::Buffer _bufferParticles;
    GL::Mesh _meshParticles;
    Containers::Pointer<ParticleSphereShader> _particleShader;
  };

}
