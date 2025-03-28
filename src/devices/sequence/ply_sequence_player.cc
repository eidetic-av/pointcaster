#include "ply_sequence_player.h"
#include "../../parameters.h"
#include "../../uuid.h"
#include <algorithm>
#include <execution>
#include <filesystem>
#include <happly.h>
#include <nfd.hpp>
#include <tbb/tbb.h>
#include <ranges>
#include <functional>

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
    : _config(config), _current_frame(0), sequence_length_seconds(0) {
  if (!config.directory.empty()) {
    if (!std::filesystem::exists(config.directory)) {
      pc::logger->error("Can't find source directory '{}'", config.directory);
      return;
    }
    for (const auto &file :
         std::filesystem::directory_iterator(config.directory)) {
      if (file.path().extension() != ".ply") continue;
      _file_paths.emplace_back(file.path().string());
    }
    constexpr float frame_rate = 30.0f;
    sequence_length_seconds = frame_count() / frame_rate;
    load_all_frames();
  }

  pc::parameters::declare_parameters(_config.id, _config);
}

size_t PlySequencePlayer::frame_count() const { return _file_paths.size(); }

size_t PlySequencePlayer::current_frame() const { return _current_frame; }

pc::types::PointCloud PlySequencePlayer::point_cloud() const {
  pc::types::PointCloud cloud = _pointcloud_buffer.at(
      std::min(_current_frame, _pointcloud_buffer.size()));
  const auto point_indices =
      std::views::iota(0, static_cast<int>(cloud.size()));
  std::for_each(std::execution::par, point_indices.begin(), point_indices.end(),
                [&, config = _config](int i) {
                  // transform positions as per config
                  const auto &pos = cloud.positions[i];
                  cloud.positions[i] = {
                      static_cast<short>(pos.x + config.translate.x),
                      static_cast<short>(pos.y + config.translate.y),
                      static_cast<short>(pos.z + config.translate.z)};
                });
  return cloud;
}

void PlySequencePlayer::tick(float delta_time) {
  constexpr float frame_rate = 30.0f;
  _frame_accumulator += delta_time * frame_rate;
  const size_t frames_to_advance = static_cast<size_t>(_frame_accumulator);
  _frame_accumulator -= frames_to_advance;
  const size_t total_frames = frame_count();
  if (total_frames > 0) {
    _current_frame = (_current_frame + frames_to_advance) % total_frames;
  }
}

void PlySequencePlayer::load_all_frames() {
   for (const auto& file_path : _file_paths) {

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

      auto point_count = x_values.size();
      cloud.positions.resize(point_count);
      cloud.colors.resize(point_count);

      const auto point_indices =
          std::views::iota(0, static_cast<int>(point_count));

      std::for_each(
          std::execution::par, point_indices.begin(), point_indices.end(),
          [&, config = _config](int i) {
            // positions m to mm
            const auto& translate = config.translate;
            const auto x = static_cast<short>(x_values[i] * 1000 + translate.x);
            const auto y = static_cast<short>(y_values[i] * 1000 + translate.y);
            const auto z = static_cast<short>(z_values[i] * 1000 + translate.z);
            cloud.positions[i] = {x, y, z};
            // color rgb to bgr
            cloud.colors[i] = {b_values[i], g_values[i], r_values[i]};
          });

      _pointcloud_buffer.push_back(cloud);

      pc::logger->info(file_path);
   }
}

} // namespace pc::devices