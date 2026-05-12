#pragma once

#include "operator_variants.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <Corrade/Tags.h>
#include <camera/camera_frame.h>
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <logger/logger.h>
#include <pointcaster/point_cloud.h>
#include <span>
#include <string_view>
#include <vector>

namespace pc {
class Workspace;
}

namespace pc::operators {

class OperatorPlugin : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.OperatorPlugin/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    std::filesystem::path exe_dir(cpplocate::getModulePath());
    auto plugin_dir = exe_dir.parent_path() / "plugins" / "operators";

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

  explicit OperatorPlugin(Corrade::PluginManager::AbstractManager &manager,
                          Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {}

  virtual ~OperatorPlugin() = default;

  virtual void init() {};

  virtual void process(const PointCloud &input, PointCloud &output,
                       const OperatorConfigurationVariant &config_variant) {
    _config = config_variant;
  };

  OperatorConfigurationVariant &config() { return _config; }
  const OperatorConfigurationVariant &config() const { return _config; }

  void update_config(const OperatorConfigurationVariant &config) {
    _config = config;
  }

  virtual void on_config_field_changed(std::string_view path = "") {}

  virtual bool plugin_null_state() const { return false; }

  // if an operator projects any camera / intermediary views into 2d that should
  // be displayed in the UI, declare them as member variables and return them
  // here
  virtual std::vector<camera::CameraFrameRef> camera_frames() { return {}; }

protected:
  OperatorConfigurationVariant _config;
};

} // namespace pc::operators
