#include "ply_device.h"
#include "plugins/devices/device_variants.h"
#include "plugins/devices/ply/ply_device_config.h"

#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <happly.h>
#include <oneapi/tbb/parallel_for.h>
#include <pointcaster/point_cloud.h>

namespace pc::devices {

using pc::profiling::ProfilingZone;

void PlyDevice::init() {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.active) {
    if (!config.file_path.empty()) load_file(config.file_path);
  }
}

void PlyDevice::on_config_field_changed(std::string_view) {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.file_path != _loaded_file_path) {
    if (load_file(config.file_path)) {
      _loaded_file_path = config.file_path;
      pc::logger()->info("Loaded PLY file from '{}'", config.file_path);
    } else {
      // revert the saved file-path if we failed to load the file
      config.file_path = _loaded_file_path;
      pc::logger()->error("Failed to load PLY file from '{}'",
                          config.file_path);
    }
  }
}

bool PlyDevice::load_file(std::string_view url) {

  static const std::string file_prefix = "file:///";
  auto path = url.substr(file_prefix.size() - 1);

  pc::logger()->trace("Loading ply file: {}", path);

  size_t point_count;
  std::vector<float> x_values, y_values, z_values;
  std::vector<unsigned char> r_values, g_values, b_values;
  pc::PointCloud cloud;
  std::optional<happly::PLYData> ply_in;

  try {
    ProfilingZone load_file_zone("OrbbecDevice::load_file");
    load_file_zone.text(path);
    ply_in.emplace(std::string(path));
  } catch (const std::runtime_error &e) {
    pc::logger()->error("Exception loading file: {}", e.what());
    return false;
  }

  {
    ProfilingZone parse_ply_zone("Parse");
    constexpr auto vertex = "vertex";

    x_values = ply_in->getElement(vertex).getProperty<float>("x");
    y_values = ply_in->getElement(vertex).getProperty<float>("y");
    z_values = ply_in->getElement(vertex).getProperty<float>("z");

    r_values = ply_in->getElement(vertex).getProperty<unsigned char>("red");
    g_values = ply_in->getElement(vertex).getProperty<unsigned char>("green");
    b_values = ply_in->getElement(vertex).getProperty<unsigned char>("blue");

    point_count = x_values.size();
    cloud.positions.resize(point_count);
    cloud.colors.resize(point_count);
  }

  {
    ProfilingZone convert_zone("Convert and pack points");
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, point_count),
        [&](const tbb::blocked_range<size_t> &range) {
          for (size_t i = range.begin(), e = range.end(); i < e; ++i) {
            cloud.positions[i] = {static_cast<short>(x_values[i] * 1000),
                                  static_cast<short>(y_values[i] * 1000),
                                  static_cast<short>(z_values[i] * 1000)};
            cloud.colors[i] = {r_values[i], g_values[i], b_values[i]};
          }
        });
  }

  _current_point_cloud = std::move(cloud);

  // TODO for sequences
  // mmap ply headers (maybe mio or llfio)
  // then Direct IO for data

  return true;
}
} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(PlyDevice, pc::devices::PlyDevice,
                        "net.pointcaster.DevicePlugin/1.0")