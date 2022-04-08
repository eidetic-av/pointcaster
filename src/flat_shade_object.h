#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

namespace bob {

  using namespace Magnum;

  using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;

  class FlatShadeObject: public SceneGraph::Drawable3D {
  public:
    explicit FlatShadeObject(Object3D& object, Shaders::Flat3D& shader, const Color3& color, GL::Mesh& mesh, SceneGraph::DrawableGroup3D* const drawables): SceneGraph::Drawable3D{object, drawables}, _shader(shader), _color(color), _mesh(mesh) {}

    void draw(const Matrix4& transformation, SceneGraph::Camera3D& camera) override {
      _shader
	.setColor(_color)
	.setTransformationProjectionMatrix(camera.projectionMatrix() * transformation)
	.draw(_mesh);
    }

    FlatShadeObject& setColor(const Color3& color) { _color = color; return *this; }

  private:
    Shaders::Flat3D& _shader;
    Color3 _color;
    GL::Mesh& _mesh;
  };

}
