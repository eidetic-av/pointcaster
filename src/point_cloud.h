#pragma once

#include <cstdint>
#include <vector>
#include <Corrade/Containers/Pointer.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Camera.h>

#include "shaders/particle_sphere.h"

namespace bob {

  using namespace Magnum;
  using namespace bob::shaders;

  template<typename pos_type = Vector3, typename color_type = Color3>
  struct PointCloud {
    std::vector<pos_type> positions;
    std::vector<color_type> colors;

    size_t size() { return positions.size(); }
    bool empty() { return positions.empty(); }

    PointCloud& operator=(PointCloud& other) {
      positions = other.positions;
      colors = other.colors;
      return *this;
    }
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

    // PointCloudRenderer& setPoints(PointCloud& points) {
    //   _points = points;
    //   // _points.positions.clear();
    //   // _points.positions.assign(points.positions.begin(), points.positions.end());
    //   // _points.positions.insert(points.positions.begin(), points.positions.end(), _points.positions.begin());
    //   return *this;
    // }

    PointCloud<Vector3, float> _points;

  private:
    bool _dirty = false;

    Float _particleRadius = 1.0f;
    GL::Buffer _positions_buffer;
    GL::Buffer _color_buffer;
    GL::Mesh _meshParticles;
    Containers::Pointer<ParticleSphereShader> _particleShader;
  };

}
