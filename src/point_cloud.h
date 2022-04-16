#pragma once

#include <cstdint>
#include <vector>
#include <Corrade/Containers/Pointer.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Camera.h>
#include "structs.h"

#include "shaders/particle_sphere.h"

namespace bob {

  using namespace Magnum;
  using namespace bob::shaders;

  struct PointCloud {
    std::vector<position> positions;
    // TODO this colors list really shouldn't be typed as 'float', as it's
    // really four packed bytes
    std::vector<float> colors;
    size_t size() { return positions.size(); }
    bool empty() { return positions.empty(); }
  };

  class PointCloudRenderer {
  public:
    PointCloudRenderer(float particleRadius);

    PointCloudRenderer& draw(Containers::Pointer<SceneGraph::Camera3D>& camera, const Vector2i& viewportSize);

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

    PointCloud _points;

  private:
    bool _dirty = false;

    Float _particleRadius = 1.0f;
    GL::Buffer _positions_buffer;
    GL::Buffer _color_buffer;
    GL::Mesh _meshParticles;
    Containers::Pointer<ParticleSphereShader> _particleShader;
  };

}
