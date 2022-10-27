#include "sphere_renderer.h"
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Shaders/Generic.h>

using namespace Magnum;
using namespace Magnum::Math::Literals;

bob::SphereRenderer::SphereRenderer() {
  mesh = MeshTools::compile(Primitives::icosphereSolid(5));
  // mesh = MeshTools::compile(Primitives::cubeSolid());
  color = Color3::fromHsv({35.0_degf, 1.0f, 1.0f});
  transform = Matrix4::scaling({1.0f, 1.0f, 1.0f}) * Matrix4::translation({0, 0, 0});
}

void bob::SphereRenderer::draw(SceneGraph::Camera3D &camera) {

  shader.setDiffuseColor(color)
      .setLightPosition({0, 0, 5.0f})
      .setTransformationMatrix(transform)
      .setNormalMatrix(transform.normalMatrix())
      .setProjectionMatrix(camera.projectionMatrix())
      .draw(mesh);
}