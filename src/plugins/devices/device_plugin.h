#pragma once

#include "device_status.h"
#include "device_tree.h"
#include "device_variants.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <Corrade/Tags.h>
#include <atomic>
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <functional>
#include <logger/logger.h>
#include <memory>
#include <pipeline/concurrent_operator_pipeline.h>
#include <plugins/operators/operator_host.h>
#include <plugins/operators/operator_plugin.h>
#include <plugins/operators/operator_variants.h>
#include <pointcaster/point_cloud.h>
#include <pointcaster_api.h>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace pc {
class Workspace;
}

namespace pc::devices {

struct DiscoveredDevice {
  std::string label;
  std::string ip;
  std::string id;
};

class POINTCASTER_CLASS_API DevicePlugin
    : public Corrade::PluginManager::AbstractPlugin,
      public pc::operators::OperatorHost {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.DevicePlugin/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    std::filesystem::path exe_dir(cpplocate::getModulePath());
#ifdef _WIN32
    // flat windows layout: plugins/ sits beside the executable
    auto plugin_dir = exe_dir / "plugins" / "devices";
#else
    // linux bin/ layout: plugins/ sits beside the executable's parent dir
    auto plugin_dir = exe_dir.parent_path() / "plugins" / "devices";
#endif

    std::vector<Corrade::Containers::String> search_paths;

    if (std::filesystem::exists(plugin_dir)) {
      for (auto &recursive_file_node :
           std::filesystem::recursive_directory_iterator(plugin_dir)) {
        if (recursive_file_node.path().extension().string() == ".conf") {
          // for each subdirectory that contains a file named .conf,
          // treat it as a plugin directory and add it to our search paths...
          search_paths.emplace_back(
              recursive_file_node.path().parent_path().string());
        }
      }
    }

    // corrade requires at least one search path entry even when no plugins exist
    if (search_paths.empty()) {
      search_paths.emplace_back(plugin_dir.string());
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

  explicit DevicePlugin(Corrade::PluginManager::AbstractManager &manager,
                        Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {
    static bool loaded_discovery_instance = false;
    if (!loaded_discovery_instance) {
      _is_discovery_instance = true;
      loaded_discovery_instance = true;
    }
  }

  virtual ~DevicePlugin() = default;

  void init(Workspace &workspace) {
    OperatorHost::init(workspace);
    pc::logger()->debug("DevicePlugin::init this={}", fmt::ptr(this));
    // the parameterless init is what implementations override
    init();
  }

  void set_is_discovery_instance(bool v) { _is_discovery_instance = v; }
  bool is_discovery_instance() { return _is_discovery_instance; };

  virtual std::vector<DiscoveredDevice> discovered_devices() const {
    return {};
  };
  virtual void refresh_discovery() {};

  POINTCASTER_API bool active();
  POINTCASTER_API bool rendering();

  virtual void add_discovery_change_callback(std::function<void()>){};
  virtual bool has_discovery_change_callback() const { return false; };

  virtual DeviceStatus status() const = 0;

  void set_status_callback(std::function<void(DeviceStatus)> cb) {
    _status_callback = cb;
  }

  void set_point_cloud_updated_callback(std::function<void()> cb) {
    _point_cloud_updated_callback = cb;
  }

  DeviceConfigurationVariant &config() { return _config; }

  virtual void update_config(const DeviceConfigurationVariant &config) {
    _config = config;
    std::visit(
        [this](auto &device_config) {
          if constexpr (requires { device_config.operators; }) {
            sync_operators(device_config.operators);
          }
        },
        _config);
  }

  POINTCASTER_API virtual void
  on_config_field_changed([[maybe_unused]] std::string_view path = "");

  virtual bool plugin_null_state() const { return false; }

  virtual std::shared_ptr<PointCloud> point_cloud() = 0;

  std::shared_ptr<std::vector<std::byte>> render_data() {
    return _latest_render_data.load(std::memory_order_acquire);
  }

  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void restart() = 0;

  void reprocess() override {}

  virtual bool is_sequence() const { return false; }
  virtual size_t frame_count() const { return 1; }

  void notify_status_changed(DeviceStatus new_status) {
    if (_status_callback) _status_callback(new_status);
  }
  void notify_status_changed() { notify_status_changed(status()); }

  void notify_point_cloud_updated() {
    if (_point_cloud_updated_callback) _point_cloud_updated_callback();
  }

  DeviceConfigurationVariant &config_variant() { return _config; }

  std::vector<camera::CameraFrame> latest_camera_frames() const {
    if (_pipeline) return _pipeline->latest_camera_frames();
    return {};
  }

  pipeline::ConcurrentOperatorPipelineConfiguration &
  pipeline_config() override {
    return std::visit(
        [](auto &device_config)
            -> pipeline::ConcurrentOperatorPipelineConfiguration & {
          if constexpr (requires { device_config.operator_pipeline; }) {
            return device_config.operator_pipeline.value();
          } else {
            // TODO
            // groups have no operator pipeline yet...
            static pipeline::ConcurrentOperatorPipelineConfiguration fallback;
            return fallback;
          }
        },
        _config);
  }

protected:
  DeviceConfigurationVariant _config;
  std::function<void(DeviceStatus)> _status_callback;
  std::function<void()> _point_cloud_updated_callback;
  bool _is_discovery_instance = false;

  std::atomic<size_t> _process_tasks_in_flight{0};

  std::atomic<std::shared_ptr<std::vector<std::byte>>> _latest_render_data;

  void on_pipeline_output(std::shared_ptr<PointCloud>) override {
    notify_point_cloud_updated();
  }

private:
  virtual void init() {};
};

} // namespace pc::devices