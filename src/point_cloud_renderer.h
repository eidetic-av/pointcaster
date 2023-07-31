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
#include "point_cloud_renderer_config.h"

namespace pc {

  using namespace Magnum;
  using namespace pc::shaders;

  class PointCloudRenderer {
  public:
    PointCloudRenderer();

    PointCloudRenderer& draw(Magnum::SceneGraph::Camera3D& camera,
			     const PointCloudRendererConfiguration& frame_config);

    bool isDirty() const { return _dirty; }

    PointCloudRenderer& setDirty() {
      _dirty = true;
      return *this;
    }

    pc::types::PointCloud points;

  private:
    bool _dirty = false;

    GL::Buffer _positions_buffer;
    GL::Buffer _color_buffer;
    GL::Mesh _meshParticles;
    Containers::Pointer<ParticleSphereShader> _particleShader;
  };
}
