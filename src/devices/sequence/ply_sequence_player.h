#pragma once

#include "../../structs.h"
#include "../device.h"
#include "ply_sequence_player_configuration.gen.h"
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>


namespace pc::devices {

struct PlySequencePlayerImplDeviceMemory;

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

  bool draw_controls() override;

private:
  std::unique_ptr<std::jthread> _io_thread;
  std::vector<std::string> _file_paths;
  size_t _max_point_count;
  float _frame_accumulator;

  std::vector<pc::types::PointCloud> _pointcloud_buffer;
  std::mutex _pointcloud_buffer_access;
  std::atomic_bool _buffer_updated;

  pc::types::PointCloud _current_point_cloud;

  PlySequencePlayerImplDeviceMemory *_device_memory;
  std::atomic_bool _device_memory_ready{false};

  bool init_device_memory();
  void free_device_memory();

  void load_all_frames();
};

} // namespace pc::devices