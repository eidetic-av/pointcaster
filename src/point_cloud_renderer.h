#pragma once

#include <memory>
#include <cstdint>
#include <vector>
#include <Corrade/Containers/Pointer.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Camera.h>
#include <pointclouds.h>
#include "structs.h"
#include "shaders/particle_sphere.h"

namespace bob {

  using namespace Magnum;
  using namespace bob::shaders;

  class PointCloudRenderer {
  public:
    PointCloudRenderer(float particleRadius);

    PointCloudRenderer& draw(Magnum::SceneGraph::Camera3D& camera, const Vector2i& viewportSize);

    bool isDirty() const { return _dirty; }

    PointCloudRenderer& setDirty() {
      _dirty = true;
      return *this;
    }

    Float particleRadius() const { return _particleRadius; }

    PointCloudRenderer& setParticleRadius(Float radius) {
      _particleRadius = radius;
      return *this;
    }

    bob::types::PointCloud points;

  private:
    bool _dirty = false;

    Float _particleRadius = 1.0f;
    GL::Buffer _positions_buffer;
    GL::Buffer _color_buffer;
    GL::Mesh _meshParticles;
    Containers::Pointer<ParticleSphereShader> _particleShader;
  };

}
