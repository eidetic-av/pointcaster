#pragma once
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/SceneGraph.h>
#include <Magnum/Shaders/Phong.h>

using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;
using Drawable3D = Magnum::SceneGraph::Drawable3D;
using DrawableGroup3D = Magnum::SceneGraph::DrawableGroup3D;

class SolidBox : public Object3D, public Drawable3D {
public:
  explicit SolidBox(Object3D *parent, DrawableGroup3D *group,
                    const Magnum::Color4 &color = {1.0f, 1.0f, 1.0f, 0.3f})
      : Object3D{parent}, Drawable3D{*this, group}, _color(color) {
    // static for sharing resources across SolidBox instances
    static Magnum::GL::Mesh mesh;
    static Magnum::Shaders::Phong shader;
    static bool initialized = false;
    if (!initialized) {
      mesh = Magnum::MeshTools::compile(Magnum::Primitives::cubeSolid());

      shader = Magnum::Shaders::Phong{};
      shader.setAmbientColor(Magnum::Color4(0.2f * _color.rgb(), _color.a()))
          .setDiffuseColor(_color)
          .setSpecularColor(Magnum::Color4{1.0f, 1.0f, 1.0f, 0.0f})
          .setShininess(50.0f);

      // for transparency
      Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);
      Magnum::GL::Renderer::setBlendFunction(
          Magnum::GL::Renderer::BlendFunction::SourceAlpha,
          Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);

      initialized = true;
    }

    _mesh = &mesh;
    _shader = &shader;
  }

  void setColor(Magnum::Color4 color) {
    _color = color;
    _shader->setAmbientColor(Magnum::Color4(0.2f * _color.rgb(), _color.a()))
        .setDiffuseColor(_color);
  }

  void draw(const Magnum::Matrix4 &transformationMatrix,
            Magnum::SceneGraph::Camera3D &camera) override {
    _shader->setTransformationMatrix(transformationMatrix)
        .setNormalMatrix(transformationMatrix.normalMatrix())
        .setProjectionMatrix(camera.projectionMatrix());
    Magnum::GL::Renderer::setDepthMask(false);
    _shader->draw(*_mesh);
    Magnum::GL::Renderer::setDepthMask(true);
  }

private:
  Magnum::Color4 _color;
  Magnum::GL::Mesh *_mesh;
  Magnum::Shaders::Phong *_shader;
};
