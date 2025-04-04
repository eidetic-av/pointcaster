#include "ply_sequence_player.h"
#include "../../logger.h"
#include "../../parameters.h"
#include "../../uuid.h"
#include <algorithm>
#include <execution>
#include <filesystem>
#include <functional>
#include <happly.h>
#include <nfd.hpp>
#include <ranges>
#include <tbb/tbb.h>

namespace pc::devices {

PlySequencePlayerConfiguration PlySequencePlayer::load_directory() {
  PlySequencePlayerConfiguration player_config{.id = pc::uuid::word()};
  {
    NFD::Guard file_dialog_guard;
    NFD::UniquePath file_dialog_out_path;
    auto file_dialog_result = NFD::PickFolder(file_dialog_out_path);
    if (file_dialog_result == NFD_OKAY) {
      player_config.directory = std::string(file_dialog_out_path.get());
    }
  }
  return player_config;
}

PlySequencePlayer::PlySequencePlayer(PlySequencePlayerConfiguration &config)
    : DeviceBase<PlySequencePlayerConfiguration>(config) {
  if (!config.directory.empty()) {
    if (!std::filesystem::exists(config.directory)) {
      pc::logger->error("Can't find source directory '{}'", config.directory);
      return;
    }
    pc::logger->info("Loading PLY sequence from directory '{}'",
                     config.directory);
    for (const auto &file :
         std::filesystem::directory_iterator(config.directory)) {
      if (file.path().extension() != ".ply") continue;
      _file_paths.emplace_back(file.path().string());
    }
  }
  {
    std::lock_guard lock(PlySequencePlayer::devices_access);
    PlySequencePlayer::attached_devices.push_back(std::ref(*this));
  }
  _io_thread =
      std::make_unique<std::jthread>([this](std::stop_token stop_token) {
        try {
          _max_point_count = load_all_frames(stop_token);
          if (_max_point_count > 0) init_device_memory(_max_point_count);
          else throw std::runtime_error("Empty PLY files or load interrupted.");
        } catch (const std::exception &e) {
          pc::logger->error("Failed to load PLY sequence");
          pc::logger->error("Encountered exception: {}", e.what());
        }
      });
  // parameters::set_parameter_minmax(std::format("{}.current_frame", id()), 0,
  //                                  frame_count());
}

PlySequencePlayer::~PlySequencePlayer() {
  if (_io_thread && _io_thread->joinable()) _io_thread->request_stop();
  free_device_memory();
  std::lock_guard lock(PlySequencePlayer::devices_access);
  std::erase_if(PlySequencePlayer::attached_devices,
                [this](auto &device) { return &(device.get()) == this; });
}

size_t PlySequencePlayer::frame_count() const { return _file_paths.size(); }

size_t PlySequencePlayer::current_frame() const {
  return config().current_frame;
}

DeviceStatus PlySequencePlayer::status() const {
  if (_device_memory_ready) return DeviceStatus::Loaded;
  return _max_point_count == 0 ? DeviceStatus::Missing : DeviceStatus::Inactive;
}

void PlySequencePlayer::tick(float delta_time) {
  auto& conf = config();
  if (!conf.playing) return;
  _frame_accumulator += delta_time * conf.frame_rate;
  const size_t frames_to_advance = static_cast<size_t>(_frame_accumulator);
  _frame_accumulator -= frames_to_advance;
  const size_t total_frames = frame_count();
  auto current_frame_index = conf.current_frame;
  if (total_frames > 0) {
    auto new_frame_index = (current_frame_index + frames_to_advance) % total_frames;
    conf.current_frame = new_frame_index;
    _buffer_updated = new_frame_index != current_frame_index;
  }
}

size_t PlySequencePlayer::load_all_frames(std::stop_token stop_token) {
  size_t max_point_count = 0;
  for (const auto &file_path : _file_paths) {

    if (stop_token.stop_requested()) return 0;

    happly::PLYData ply_in(file_path);
    pc::types::PointCloud cloud;

    constexpr auto vertex_element_name = "vertex";
    auto x_values =
        ply_in.getElement(vertex_element_name).getProperty<float>("x");
    auto y_values =
        ply_in.getElement(vertex_element_name).getProperty<float>("y");
    auto z_values =
        ply_in.getElement(vertex_element_name).getProperty<float>("z");
    auto r_values = ply_in.getElement(vertex_element_name)
                        .getProperty<unsigned char>("red");
    auto g_values = ply_in.getElement(vertex_element_name)
                        .getProperty<unsigned char>("green");
    auto b_values = ply_in.getElement(vertex_element_name)
                        .getProperty<unsigned char>("blue");

    const auto point_count = x_values.size();
    cloud.positions.resize(point_count);
    cloud.colors.resize(point_count);

    if (point_count > max_point_count) max_point_count = point_count;

    const auto point_indices =
        std::views::iota(0, static_cast<int>(point_count));

    std::for_each(std::execution::par, point_indices.begin(),
                  point_indices.end(), [&, config = this->config()](int i) {
                    // positions m to mm
                    const auto x = static_cast<short>(x_values[i] * 1000);
                    const auto y = static_cast<short>(y_values[i] * 1000);
                    const auto z = static_cast<short>(z_values[i] * 1000);
                    cloud.positions[i] = {x, y, z};
                    // color rgb to bgr
                    cloud.colors[i] = {b_values[i], g_values[i], r_values[i]};
                  });

    _pointcloud_buffer.push_back(cloud);
  }
  return max_point_count;
}

} // namespace pc::devices