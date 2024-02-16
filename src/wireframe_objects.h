#pragma once

#include <Corrade/Containers/Pointer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Grid.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/Trade/MeshData.h>
#include "flat_shade_object.h"
#include "gui/catpuccin.h"

namespace pc {

using namespace Magnum;

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

class WireframeObject {
public:
  explicit WireframeObject(Scene3D *const scene,
			   SceneGraph::DrawableGroup3D *const parent_group)
      : _parent_group(parent_group) {
    _object.reset(new Object3D{scene});
    _shader = Shaders::FlatGL3D{};
    _drawable.reset(new FlatShadeObject{*_object, _shader, Color3{1.0f}, _mesh,
					_parent_group});
  }

  WireframeObject &set_color(const Color3 &color) {
    _drawable->setColor(color);
    return *this;
  }
  WireframeObject &transform(const Matrix4 &matrix) {
    _object->transform(matrix);
    return *this;
  }
  WireframeObject &set_transformation(const Matrix4 &matrix) {
    _object->setTransformation(matrix);
    return *this;
  }

  void set_visible(bool visible) {
    if (_visible == visible) return;
    if (_visible && !visible) {
      _visible = false;
      _parent_group->remove(*_drawable.get());
      return;
    }
    if (!_visible && visible) {
      _visible = true;
      _parent_group->add(*_drawable.get());
    }
  }

protected:
  bool _visible = true;
  SceneGraph::DrawableGroup3D *const _parent_group;
  GL::Mesh _mesh{NoCreate};
  Shaders::FlatGL3D _shader{NoCreate};
  Containers::Pointer<Object3D> _object;
  Containers::Pointer<FlatShadeObject> _drawable;
};

class WireframeBox : public WireframeObject {
public:
  explicit WireframeBox(Scene3D *const scene,
                        SceneGraph::DrawableGroup3D *const parent_group)
      : WireframeObject{scene, parent_group} {
    _mesh = MeshTools::compile(Primitives::cubeWireframe());
  }
};

class WireframeGrid : public WireframeObject {
public:
  explicit WireframeGrid(Scene3D *const scene,
                         SceneGraph::DrawableGroup3D *const parent_group)
      : WireframeObject{scene, parent_group} {
    using namespace Magnum::Math::Literals;

    _mesh = MeshTools::compile(Primitives::grid3DWireframe({5, 5}));
    _object->scale(Vector3(1.0f));
    _object->rotateX(90.0_degf);
  }
};

} // namespace pc
