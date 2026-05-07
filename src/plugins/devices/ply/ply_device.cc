#include "ply_device.h"
#include "plugins/devices/device_variants.h"
#include "plugins/devices/ply/ply_device_config.h"

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <cstring>
#include <happly.h>
#include <oneapi/tbb/parallel_for.h>
#include <pointcaster/point_cloud.h>
#include <ranges>
#include <workspace/workspace.h>

namespace pc::devices {

using pc::profiling::ProfilingZone;

void PlyDevice::init(Workspace &workspace) {
  _workspace = &workspace;

  auto &backend_manager = _workspace->backend_plugin_manager;
  _cpu_backend = backend_manager->instantiate("CpuBackend");

  pc::logger()->trace("PlyDevice created CPU backend");

  using Corrade::PluginManager::LoadState;

  for (const auto &plugin : backend_manager->pluginList()) {
    if (backend_manager->loadState(plugin) & LoadState::NotLoaded) continue;
    if (plugin == "CudaBackend") {
      _cuda_backend = backend_manager->instantiate(plugin);
      pc::logger()->trace("PlyDevice created CUDA backend");
    }
  }

  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.active && !config.file.file_path.empty()) {
    load_file(config.file.file_path);
  }
}

std::shared_ptr<PointCloud> PlyDevice::point_cloud() {
  return _current_point_cloud;
}

void PlyDevice::on_config_field_changed(std::string_view) {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.file.file_path != _loaded_file_path) {
    if (load_file(config.file.file_path)) {
      _loaded_file_path = config.file.file_path;
      pc::logger()->info("Loaded PLY file from '{}'", config.file.file_path);
    } else {
      pc::logger()->error("Failed to load PLY file from '{}'",
                          config.file.file_path);
      config.file.file_path = _loaded_file_path;
    }
  } else {
    apply_transform();
  }
}

bool PlyDevice::load_file(std::string_view url) {
  static const std::string file_prefix = "file:///";
  auto path = url.substr(file_prefix.size());

  const auto config = std::get<PlyDeviceConfiguration>(_config);

  pc::logger()->trace("Loading ply file: {}", path);

  std::optional<happly::PLYData> ply_in;

  try {
    ProfilingZone load_file_zone("PlyDevice::load_file");
    load_file_zone.text(path);
    ply_in.emplace(std::string(path));
  } catch (const std::runtime_error &e) {
    pc::logger()->error("Exception loading file: {}", e.what());
    return false;
  }

  constexpr auto vertex = "vertex";

  const auto x_values = ply_in->getElement(vertex).getProperty<short>("x");
  const auto y_values = ply_in->getElement(vertex).getProperty<short>("y");
  const auto z_values = ply_in->getElement(vertex).getProperty<short>("z");
  const auto r_values =
      ply_in->getElement(vertex).getProperty<unsigned char>("red");
  const auto g_values =
      ply_in->getElement(vertex).getProperty<unsigned char>("green");
  const auto b_values =
      ply_in->getElement(vertex).getProperty<unsigned char>("blue");

  const size_t point_count = x_values.size();

  auto input_cloud = std::make_shared<PointCloud>();
  input_cloud->resize(point_count);

  {
    ProfilingZone convert_zone("Convert and pack points");
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, point_count),
        [&](const tbb::blocked_range<size_t> &range) {
          for (size_t i = range.begin(), e = range.end(); i < e; ++i) {
            input_cloud->positions[i] = {x_values[i], y_values[i], z_values[i]};
            input_cloud->colors[i] = {r_values[i], g_values[i], b_values[i]};
          }
        });
  }

  _input_cloud = std::move(input_cloud);
  return true;
}

void PlyDevice::apply_transform() {
  if (!_input_cloud) return;

  const auto &config = std::get<PlyDeviceConfiguration>(_config);
  const auto point_count = _input_cloud->size();

  auto output = std::make_shared<PointCloud>();
  output->resize(point_count);

  // TODO cuda backend
  backend::BackendPlugin *backend = _cpu_backend.get();

  if (backend) {
    backend->transform_point_cloud(*_input_cloud, output, config.transform,
                                   config.color);
  }

  for (const auto &[operator_plugin, operator_config] :
       std::views::zip(operators, config.operators)) {
    operator_plugin->process(*output, *output, operator_config);
  }

  std::shared_ptr<std::vector<std::byte>> render_buffer;
  if (config.render && backend) {
    const auto final_count = output->size();
    render_buffer = std::make_shared<std::vector<std::byte>>(final_count * 16);
    backend->pack_render_buffer(*output, *render_buffer);
  }

  _current_point_cloud = std::move(output);
  if (render_buffer) {
    _latest_render_data.store(std::move(render_buffer),
                              std::memory_order_release);
  }
  notify_point_cloud_updated();
}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(PlyDevice, pc::devices::PlyDevice,
                        "net.pointcaster.DevicePlugin/1.0")