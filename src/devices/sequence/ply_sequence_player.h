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

class PlySequencePlayer : public DeviceBase<PlySequencePlayerConfiguration> {
public:
  inline static std::vector<std::reference_wrapper<PlySequencePlayer>> attached_devices;
  inline static std::mutex devices_access;

  static PlySequencePlayerConfiguration load_directory();

  float sequence_length_seconds;

  explicit PlySequencePlayer(PlySequencePlayerConfiguration &config);
  ~PlySequencePlayer();

  PlySequencePlayer(const PlySequencePlayer &) = delete;
  PlySequencePlayer &operator=(const PlySequencePlayer &) = delete;
  PlySequencePlayer(PlySequencePlayer &&) = delete;
  PlySequencePlayer &operator=(PlySequencePlayer &&) = delete;

  void tick(float delta_time);

  size_t frame_count() const;
  size_t current_frame() const;
  pc::types::PointCloud point_cloud(pc::operators::OperatorList operators = {}) override;

private:
  std::unique_ptr<std::jthread> _io_thread;
  std::vector<std::string> _file_paths;

  std::vector<pc::types::PointCloud> _pointcloud_buffer;
  
  float _frame_accumulator;
  size_t _current_frame;

  void load_all_frames();
};

} // namespace pc::devices