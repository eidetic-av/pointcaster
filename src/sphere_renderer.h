#pragma once

#include "structs.h"
#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Corrade/Containers/Pointer.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/Math/Color.h>

namespace pc {

  class SphereRenderer {
  public:
    pc::types::float3 position {0, 0, 0};
    float radius = 3.0f;

    Magnum::GL::Mesh mesh;
    Magnum::Shaders::PhongGL shader;
    Magnum::Color3 color;
    Magnum::Matrix4 transform;

    SphereRenderer();

    void draw(Magnum::SceneGraph::Camera3D &camera);
  };

}
