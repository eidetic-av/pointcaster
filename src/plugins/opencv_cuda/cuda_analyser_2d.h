#pragma once

#include "../../analysis/analyser_2d_config.gen.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <opencv2/core/mat.hpp>


namespace pc::analysis {

class AbstractCudaAnalyser2D : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.CudaAnalyser2D/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    // <exe_dir>/plugins/opencv_cuda
    return {Corrade::InPlaceInit, {"plugins/opencv_cuda"}};
  }

  explicit AbstractCudaAnalyser2D(
      Corrade::PluginManager::AbstractManager &manager,
      Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {}

  virtual cv::Mat
  setup_input_frame_cuda(const cv::Mat &rgba_input,
                         const Analyser2DConfiguration &config) = 0;

  virtual std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>,
                     std::vector<uchar>>
  calculate_optical_flow_cuda(const cv::Mat &input_frame_1,
                              const cv::Mat &input_frame_2,
                              const OpticalFlowConfiguration &config) = 0;
};

} // namespace pc::analysis
