#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/Shaders/Generic.h>

namespace bob::shaders {

  using namespace Magnum;

  class ParticleSphereShader: public GL::AbstractShaderProgram {
  public:
    explicit ParticleSphereShader();

    ParticleSphereShader& setParticleRadius(Float radius);
    ParticleSphereShader& setPointSizeScale(Float scale);
    ParticleSphereShader& setViewport(const Vector2i& viewport);
    ParticleSphereShader& setViewMatrix(const Matrix4& matrix);
    ParticleSphereShader& setProjectionMatrix(const Matrix4& matrix);

  private:
    Int _uParticleRadius,
      _uPointSizeScale,
      _uViewMatrix,
      _uProjectionMatrix;
  };

} //bob::shaders
