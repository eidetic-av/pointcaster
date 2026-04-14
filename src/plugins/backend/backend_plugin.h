#pragma once

#include "backend_types.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <config/transform_config.h>
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <functional>
#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <span>
#include <tuple>

namespace pc::backend {

class BackendPlugin : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.BackendPlugin/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    std::filesystem::path exe_dir(cpplocate::getModulePath());
    auto plugin_dir = exe_dir.parent_path() / "plugins" / "backend";

    std::vector<Corrade::Containers::String> search_paths;

    for (auto &recursive_file_node :
         std::filesystem::recursive_directory_iterator(plugin_dir)) {
      if (recursive_file_node.path().extension().string() == ".conf") {
        // for each subdirectory that contains a file named .conf,
        // treat it as a plugin directory and add it to our search paths...
        search_paths.emplace_back(
            recursive_file_node.path().parent_path().string());
      }
    }

#ifdef _WIN32
    // convert C:\style\path into posix /style/path used by corrade
    for (auto &path_entry : search_paths) {
      std::string result(path_entry.data());
      result.erase(0, 2);
      std::replace(result.begin(), result.end(), '\\', '/');
      path_entry = result;
    }
#endif

    // return results in a corrade array (required instead of vector)
    static Corrade::Containers::Array<Corrade::Containers::String> results;
    arrayClear(results);
    arrayReserve(results, search_paths.size());
    for (auto &path_string : search_paths) {
      arrayAppend(results, path_string);
    }

    return {Corrade::InPlaceInit, results};
  }

  explicit BackendPlugin(Corrade::PluginManager::AbstractManager &manager,
                         Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {}

  virtual ~BackendPlugin() = default;

  // since plugins are created by a factory, we implement a custom init() which
  // must be called
  virtual void init([[maybe_unused]] const size_t point_count) {};

  virtual void
  project_transform_frame_data(UShortDepthData input_depth_frame,
                               RgbColorData input_rgb_frame,
                               std::shared_ptr<PointCloud> output_cloud,
                               const CameraIntrinsics &camera_intrinsics) = 0;
};

} // namespace pc::backend