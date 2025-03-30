#pragma once
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/SceneGraph.h>
#include <Magnum/Shaders/PhongGL.h>
#include <Magnum/Trade/MeshData.h>

using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;
using Drawable3D = Magnum::SceneGraph::Drawable3D;
using DrawableGroup3D = Magnum::SceneGraph::DrawableGroup3D;

class SolidBox : public Object3D, public Drawable3D {
public:
  explicit SolidBox(Object3D *parent, DrawableGroup3D *group,
                    const Magnum::Color4 &color = {1.0f, 1.0f, 1.0f, 0.3f})
      : Object3D{parent}, Drawable3D{*this, group}, _color(color) {

    _shader = Magnum::Shaders::PhongGL{};
    _shader.setAmbientColor(Magnum::Color4(0.2f * _color.rgb(), _color.a()))
        .setDiffuseColor(_color)
        .setSpecularColor(Magnum::Color4{1.0f, 1.0f, 1.0f, 0.0f})
        .setShininess(50.0f);

    // for transparency
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);
    Magnum::GL::Renderer::setBlendFunction(
        Magnum::GL::Renderer::BlendFunction::SourceAlpha,
        Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
  }

  void setColor(Magnum::Color4 color) {
    _color = color;
    _shader.setAmbientColor(Magnum::Color4(0.2f * _color.rgb(), _color.a()))
        .setDiffuseColor(_color);
  }
  Magnum::Color4 getColor() const { return _color; }
  void setVisible(bool visible) { _visible = visible; }
  bool visible() const { return _visible; }

  void draw(const Magnum::Matrix4 &transformationMatrix,
            Magnum::SceneGraph::Camera3D &camera) override {
    if (!_visible) return;

    Magnum::GL::Renderer::setDepthMask(false);

    _shader.setTransformationMatrix(transformationMatrix)
        .setNormalMatrix(transformationMatrix.normalMatrix())
        .setProjectionMatrix(camera.projectionMatrix());

    static Magnum::GL::Mesh box_mesh =
        Magnum::MeshTools::compile(Magnum::Primitives::cubeSolid());
    _shader.draw(box_mesh);

    Magnum::GL::Renderer::setDepthMask(true);
  }

private:
  Magnum::Shaders::Phong _shader;
  Magnum::Color4 _color;
  bool _visible = true;
};
