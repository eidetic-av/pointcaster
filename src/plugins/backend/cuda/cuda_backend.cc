#include "cuda_backend.h"
#include "cuda_backend_core.h"

#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <core/logger/logger.h>

namespace pc::backend {

void CudaBackend::init(const size_t point_count) {
  if (!cuda::init_device_memory(this, point_count)) {
    pc::logger()->error("Failed to initialse CUDA device memory");
  }
};

CudaBackend::~CudaBackend() {
  cuda::free_device_memory(this);
};

void CudaBackend::project_transform_frame_data(
    UShortDepthData input_depth_frame, RgbColorData input_rgb_frame,
    std::shared_ptr<PointCloud> output_cloud,
    const CameraIntrinsics &camera_intrinsics,
    std::span<std::byte> render_output) {
  cuda::project_transform_frame_data(this, input_depth_frame, input_rgb_frame,
                                     output_cloud, camera_intrinsics,
                                     render_output);
}

} // namespace pc::backend

CORRADE_PLUGIN_REGISTER(CudaBackend, pc::backend::CudaBackend,
                        "net.pointcaster.BackendPlugin/1.0")