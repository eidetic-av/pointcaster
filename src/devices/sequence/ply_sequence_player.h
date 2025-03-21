#pragma once

#include "../../structs.h"
#include "../device.h"
#include "ply_sequence_player_configuration.gen.h"
#include <optional>
#include <string_view>
#include <memory>
#include <thread>
#include <functional>

namespace pc::devices {

class PlySequencePlayer {
public:
  static PlySequencePlayerConfiguration load_directory();

  // TODO get rid of this stuff and handle references properly
  inline static std::vector<std::reference_wrapper<PlySequencePlayer>> players;

  float sequence_length_seconds;

  explicit PlySequencePlayer(PlySequencePlayerConfiguration &config);

  PlySequencePlayerConfiguration& config() { return _config; }

  void tick(float delta_time);

  size_t frame_count() const;
  size_t current_frame() const;
  pc::types::PointCloud point_cloud() const;

private:
  PlySequencePlayerConfiguration _config;
  std::unique_ptr<std::jthread> _io_thread;
  std::vector<std::string> _file_paths;

  std::vector<pc::types::PointCloud> _pointcloud_buffer;
  
  float _frame_accumulator;
  size_t _current_frame;

  void load_all_frames();
};

} // namespace pc::devices