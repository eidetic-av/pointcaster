#include <Magnum/GL/AbstractShaderProgram.h>

namespace bob::shaders {

  using namespace Magnum;

  class ParticleSphereShader: public GL::AbstractShaderProgram {
  public:
    enum ColorMode {
      UniformDiffuseColor = 0,
      RampColorById,
      ConsistentRandom
    };

    explicit ParticleSphereShader();

    ParticleSphereShader& setNumParticles(Int numParticles);
    ParticleSphereShader& setParticleRadius(Float radius);

    ParticleSphereShader& setPointSizeScale(Float scale);
    ParticleSphereShader& setColorMode(Int colorMode);
    ParticleSphereShader& setAmbientColor(const Color3& color);
    ParticleSphereShader& setDiffuseColor(const Color3& color);
    ParticleSphereShader& setSpecularColor(const Color3& color);
    ParticleSphereShader& setShininess(Float shininess);

    ParticleSphereShader& setViewport(const Vector2i& viewport);
    ParticleSphereShader& setViewMatrix(const Matrix4& matrix);
    ParticleSphereShader& setProjectionMatrix(const Matrix4& matrix);
    ParticleSphereShader& setLightDirection(const Vector3& lightDir);

  private:
    Int _uNumParticles,
      _uParticleRadius,
      _uPointSizeScale,
      _uColorMode,
      _uAmbientColor,
      _uDiffuseColor,
      _uSpecularColor,
      _uShininess,
      _uViewMatrix,
      _uProjectionMatrix,
      _uLightDir;
  };

} //bob::shaders
