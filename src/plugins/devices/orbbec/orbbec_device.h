#pragma once

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>

namespace pc::devices {
class AbstractOrbbecDevice : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.OrbbecDevice/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    return {Corrade::InPlaceInit, {"../plugins/orbbec"}};
  }

  explicit AbstractOrbbecDevice(
      Corrade::PluginManager::AbstractManager &manager,
      Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {}

//   virtual cv::Mat
//   setup_input_frame_cuda(const cv::Mat &rgba_input,
//                          const Or &config) = 0;

//   virtual std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>,
//                      std::vector<uchar>>
//   calculate_optical_flow_cuda(const cv::Mat &input_frame_1,
//                               const cv::Mat &input_frame_2,
//                               const OpticalFlowConfiguration &config) = 0;
};

}