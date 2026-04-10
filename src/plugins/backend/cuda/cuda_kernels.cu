#include "cuda_kernels.h"

#include <iostream>

namespace pc::backend::cuda {
void transform_point_cloud(const TransformConfiguration &transform,
                           PointCloud &output_cloud,
                           const PointGeneratorFunction &generate_point) {
  std::cout << "Oi\n";
  //
}

} // namespace pc::backend::cuda