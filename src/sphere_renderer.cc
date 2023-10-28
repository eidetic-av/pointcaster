#include "sphere_renderer.h"
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Shaders/Generic.h>

using namespace Magnum;
using namespace Magnum::Math::Literals;

pc::SphereRenderer::SphereRenderer() {
  mesh = MeshTools::compile(Primitives::icosphereSolid(5));
  color = Color3::fromHsv({35.0_degf, 1.0f, 1.0f});
  transform = Matrix4::scaling({0.01f, 0.01f, 0.01f}) * Matrix4::translation({0, 0, 0});
}

void pc::SphereRenderer::draw(Magnum::SceneGraph::Camera3D &camera,
	  std::vector<pc::types::position> sphere_positions) {

  Magnum::Matrix4 t = transform;
  for (auto sphere_position : sphere_positions) {

    // convert short mm value to floats
    auto x = sphere_position.x / 1000.0f;
    auto y = sphere_position.y / 1000.0f;
    auto z = sphere_position.z / 1000.0f;
    t.translation() = {x, y, z};

    shader.setDiffuseColor(color)
	.setTransformationMatrix(t)
	.setNormalMatrix(t.normalMatrix())
	.setLightPositions({{0, 0, 5.0f, 0.0f}})
	.setProjectionMatrix(camera.projectionMatrix())
	.draw(mesh);
  }
}
