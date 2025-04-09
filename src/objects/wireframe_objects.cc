#include "wireframe_objects.h"

namespace pc {

WireframeGrid::WireframeGrid(Scene3D *const scene,
                             SceneGraph::DrawableGroup3D *const parent_group)
    : WireframeObject{scene, parent_group} {
  using namespace Magnum::Math::Literals;

  const int cell_count = 6;
  const float cell_size_m = 1.0f;
  const Vector2i subdivisions{cell_count - 1, cell_count - 1};

  _mesh = MeshTools::compile(Primitives::grid3DWireframe(subdivisions));

  // rotate the upright grid to draw it as a floor
  _object->rotateX(90.0_degf);

  const float scale_factor = (cell_count * cell_size_m) / 2.0f;
  _object->scale(Vector3(scale_factor));
}

} // namespace pc