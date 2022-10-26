#include <Corrade/Containers/Pointer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Grid.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/Trade/MeshData.h>

#include "flat_shade_object.h"

namespace bob {

  using namespace Magnum;

  using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
  using Scene3D  = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

  class WireframeObject {
  public:
    explicit WireframeObject(Scene3D* const scene, SceneGraph::DrawableGroup3D* const drawableGroup) {
      _obj3D.reset(new Object3D{scene});
      _flatShader = Shaders::Flat3D{};
      _drawableObj.reset(new FlatShadeObject{*_obj3D, _flatShader, Color3{0.75f}, _mesh, drawableGroup});
    }

    WireframeObject& setColor(const Color3& color) {
      _drawableObj->setColor(color);
      return *this;
    }
    WireframeObject& transform(const Matrix4& matrix) {
      _obj3D->transform(matrix);
      return *this;
    }
    WireframeObject& setTransformation(const Matrix4& matrix) {
      _obj3D->setTransformation(matrix);
      return *this;
    }

  protected:
    GL::Mesh _mesh{NoCreate};
    Shaders::Flat3D _flatShader{NoCreate};
    Containers::Pointer<Object3D> _obj3D;
    Containers::Pointer<FlatShadeObject> _drawableObj;
  };

  class WireframeBox: public WireframeObject {
  public:
    explicit WireframeBox(Scene3D* const scene, SceneGraph::DrawableGroup3D* const drawableGroup): WireframeObject{scene, drawableGroup} {
      _mesh = MeshTools::compile(Primitives::cubeWireframe());
    }
  };

  class WireframeGrid: public WireframeObject {
  public:
    explicit WireframeGrid(Scene3D* const scene, SceneGraph::DrawableGroup3D* const drawableGroup): WireframeObject{scene, drawableGroup} {
      using namespace Magnum::Math::Literals;

      _mesh = MeshTools::compile(Primitives::grid3DWireframe({ 10, 10 }));
      _obj3D->scale(Vector3(10.0f));
      _obj3D->rotateX(90.0_degf);
    }
  };

}
