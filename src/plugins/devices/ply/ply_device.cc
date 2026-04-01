#include "ply_device.h"
#include "plugins/devices/device_variants.h"
#include "plugins/devices/ply/ply_device_config.h"

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <happly.h>
#include <llfio.hpp>
#include <oneapi/tbb/parallel_for.h>
#include <pointcaster/point_cloud.h>
#include <ranges>

namespace pc::devices {

using pc::profiling::ProfilingZone;

void PlyDevice::init() {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.active) {
    if (!config.file.file_path.empty())
      load_file(config.file.file_path);
  }
}

const PointCloud &PlyDevice::point_cloud() {
  const auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.file.load_sequence) {
    const auto &pc = _sequence_cloud_buffer[_current_sequence_frame_index];
    // pc::logger()->debug("pc.size(): {}", pc.size());
    return pc;
  } else {
    return _current_point_cloud;
  }
}

void PlyDevice::on_config_field_changed(std::string_view) {
  auto &config = std::get<PlyDeviceConfiguration>(_config);
  if (config.file.file_path != _loaded_file_path) {
    if (load_file(config.file.file_path)) {
      _loaded_file_path = config.file.file_path;
      pc::logger()->info("Loaded PLY file from '{}'",
                         config.file.file_path);
    } else {
      // revert the saved file-path if we failed to load the file
      pc::logger()->error("Failed to load PLY file from '{}'",
                          config.file.file_path);
      config.file.file_path = _loaded_file_path;
    }
  }
}

bool PlyDevice::load_file(std::string_view url) {
  static const std::string file_prefix = "file:///";
  auto path = url.substr(file_prefix.size() - 1);

  const auto config = std::get<PlyDeviceConfiguration>(_config);

  pc::logger()->trace("Loading ply file: {}", path);

  size_t point_count;
  std::vector<float> x_values, y_values, z_values;
  std::vector<unsigned char> r_values, g_values, b_values;
  pc::PointCloud cloud{{}, {}};
  std::optional<happly::PLYData> ply_in;

  try {
    // ProfilingZone load_file_zone("PlyDevice::load_file");
    // load_file_zone.text(path);
    if (config.file.load_sequence) {

      std::filesystem::path fs_path(path.data());
      auto directory_path = fs_path.parent_path().string();

      namespace llfio = LLFIO_V2_NAMESPACE;

      pc::logger()->debug("constructing directory_handle to '{}'...",
                          directory_path);

      llfio::directory_handle dh = llfio::directory({}, directory_path).value();

      pc::logger()->debug("constructed");

      std::vector<llfio::directory_handle::buffer_type> buffer(512);
      llfio::directory_handle::buffers_type entries(buffer);
      for (;;) {
        entries = dh.read({std::move(entries)}).value();
        if (entries.done()) {
          break;
        }
        buffer.resize(buffer.size() << 1);
        entries = {buffer, std::move(entries)};
      }

      pc::logger()->debug("all_entries_read");

      _sequence_file_entries.reserve(buffer.size());

      for (auto &entry : entries) {
        if (entry.leafname.extension().compare(".ply") != 0) continue;

        // if (!(entries.metadata() & llfio::stat_t::want::size)) {
        // llfio::file_handle fh =
        //     llfio::file(dh, entry.leafname,
        //                 llfio::file_handle::mode::attr_read)
        //         .value();
        // entry.stat.fill(fh, llfio::stat_t::want::size).value();

        // TODO these file handles i think are what we want to store in our
        // per ply device container when loading a sequence directory
        // -- then we
        // fh.read()
        // -- when we need the specific frame
        // and we have some mechanism that loads the next n entries into a
        // buffer to just load that file handle into cache / buffer or
        // something, maybe pre-fetch somehow
        // --

        // i think _sequence_files type must be a structure that contains the
        // file_mapping as well as the istream needed to construct a PLYData
        // into memory as per this answer constructing the imemstream
        // https://stackoverflow.com/questions/10839747/istream-vs-memory-mapping-a-file
        // but i feel like it might not work with binary (only ascii ply, we
        // will see) OR a member function OR free util function that does the
        // conversion from file_handle -> stream interface -> ply data,
        // reading it from disk or cache... yeah i think the util function
        // something like read_as_ply(llfio::file_handle) -> PLYData...

        // happly::PLYData()
        // }

        // TODO atm this actually returns an exception for: too many open files
        // so i actually DO need to stream the file_handle into the ring buffer,
        // not just the point cloud in the ring buffer and keep all file handles
        // in memory, caust it seems like thats actually not possible

        std::cout << "about to load " << entry.leafname << "\n";
        _sequence_file_entries.push_back(std::move(entry));

        // llfio::file(dh, entry.leafname, llfio::file_handle::mode::attr_read)
        //     .value();
        pc::logger()->debug("afterwards...");

        // std::cout << " maximum extent: " << entry.stat.st_size;
      }
      std::cout << std::endl;

      for (auto &entry : _sequence_file_entries) {
        std::cout << "pre-loaded: " << entry.leafname << "\n";
      }

      // need to read into the pointcloud here

      PointCloud pc{{}, {}};

      const auto &target_entry = _sequence_file_entries[0];

      auto handle = llfio::file({}, target_entry.leafname,
                                llfio::file_handle::mode::attr_read)
                        .value();

      // TODO

      // const auto file_size = handle.read(0, {});
      // pc::logger()->debug("file_size: {}", file_size.value());

      // _sequence_cloud_buffer[_current_sequence_frame_index];

      // using namespace boost::interprocess;tr

      // file_mapping file(path.data(), read_only);

      // // read until the end of the header?

      // mapped_region region(file, read_only);

      // void *addr = region.get_address();
      // std::size_t size = region.get_size();

    } else {
      ply_in.emplace(std::string(path));
    }

  } catch (const std::runtime_error &e) {
    pc::logger()->error("Exception loading file: {}", e.what());
    return false;
  }

  {
    // ProfilingZone parse_ply_zone("Parse");
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
    // ProfilingZone convert_zone("Convert and pack points");
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