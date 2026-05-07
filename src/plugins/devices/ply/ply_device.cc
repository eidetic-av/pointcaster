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

namespace pc::devices {

using pc::profiling::ProfilingZone;

void PlyDevice::init(Workspace &workspace) {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.active) {
    if (!config.file.file_path.empty()) load_file(config.file.file_path);
  }
  _workspace = &workspace;
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
  }
}

bool PlyDevice::load_file(std::string_view url) {
  static const std::string file_prefix = "file:///";
  auto path = url.substr(file_prefix.size());

  const auto config = std::get<PlyDeviceConfiguration>(_config);

  pc::logger()->trace("Loading ply file: {}", path);

  size_t point_count;
  // std::vector<float> x_values, y_values, z_values;
  std::vector<short> x_values, y_values, z_values;
  std::vector<unsigned char> r_values, g_values, b_values;
  pc::PointCloud cloud{{}, {}};
  std::optional<happly::PLYData> ply_in;

  try {
    ProfilingZone load_file_zone("PlyDevice::load_file");
    load_file_zone.text(path);
    ply_in.emplace(std::string(path));
  } catch (const std::runtime_error &e) {
    pc::logger()->error("Exception loading file: {}", e.what());
    return false;
  }

  {
    ProfilingZone parse_ply_zone("Parse");
    constexpr auto vertex = "vertex";

    x_values = ply_in->getElement(vertex).getProperty<short>("x");
    y_values = ply_in->getElement(vertex).getProperty<short>("y");
    z_values = ply_in->getElement(vertex).getProperty<short>("z");

    r_values = ply_in->getElement(vertex).getProperty<unsigned char>("red");
    g_values = ply_in->getElement(vertex).getProperty<unsigned char>("green");
    b_values = ply_in->getElement(vertex).getProperty<unsigned char>("blue");

    point_count = x_values.size();
    cloud.positions.resize(point_count);
    cloud.colors.resize(point_count);
  }

  auto render_buffer =
      config.render ? std::make_shared<std::vector<std::byte>>(point_count * 16)
                    : nullptr;
  char *render_dest =
      render_buffer ? reinterpret_cast<char *>(render_buffer->data()) : nullptr;

  {
    ProfilingZone convert_zone("Convert and pack points");
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, point_count),
        [&](const tbb::blocked_range<size_t> &range) {
          for (size_t i = range.begin(), e = range.end(); i < e; ++i) {
            cloud.positions[i] = {x_values[i], y_values[i], z_values[i]};
            cloud.colors[i] = {r_values[i], g_values[i], b_values[i]};

            if (render_dest) {
              std::memcpy(render_dest + i * 16, &cloud.positions[i], 8);
              std::memcpy(render_dest + i * 16 + 8, &cloud.colors[i], 4);
              float idx = static_cast<float>(i);
              std::memcpy(render_dest + i * 16 + 12, &idx, 4);
            }
          }
        });
  }

  // TODO this can be reduced in parallel
  position_bounds bounds{};
  for (size_t i = 0; i < point_count; ++i) {
    auto &p = cloud.positions[i];
    bounds.min.x = std::min(bounds.min.x, p.x);
    bounds.min.y = std::min(bounds.min.y, p.y);
    bounds.min.z = std::min(bounds.min.z, p.z);
    bounds.max.x = std::max(bounds.max.x, p.x);
    bounds.max.y = std::max(bounds.max.y, p.y);
    bounds.max.z = std::max(bounds.max.z, p.z);
  }
  cloud.bounds = bounds;

  _current_point_cloud = std::make_shared<PointCloud>(std::move(cloud));

  if (render_buffer) {
    _latest_render_data.store(std::move(render_buffer),
                              std::memory_order_release);
  }

  notify_point_cloud_updated();
  return true;
}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(PlyDevice, pc::devices::PlyDevice,
                        "net.pointcaster.DevicePlugin/1.0")