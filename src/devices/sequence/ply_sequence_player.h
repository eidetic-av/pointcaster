#pragma once

#include "../../structs.h"
#include "../device.h"
#include "ply_sequence_player_config.gen.h"
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <optional>
#include <string_view>
#include <thread>

namespace pc::devices {

struct PlySequencePlayerImplDeviceMemory;

class PlySequencePlayer : public DeviceBase<PlySequencePlayerConfiguration> {
public:
  inline static std::vector<std::reference_wrapper<PlySequencePlayer>>
      attached_devices;
  inline static std::mutex devices_access;

  static PlySequencePlayerConfiguration load_directory();

  explicit PlySequencePlayer(PlySequencePlayerConfiguration &config);
  ~PlySequencePlayer();

  PlySequencePlayer(const PlySequencePlayer &) = delete;
  PlySequencePlayer &operator=(const PlySequencePlayer &) = delete;
  PlySequencePlayer(PlySequencePlayer &&) = delete;
  PlySequencePlayer &operator=(PlySequencePlayer &&) = delete;

  void tick(float delta_time);

  size_t frame_count() const;
  size_t current_frame() const;

  DeviceStatus status() const override;
  pc::types::PointCloud point_cloud() override;

  bool draw_controls() override;

private:
  std::vector<std::string> _file_paths;
  float _frame_accumulator{0};
  std::atomic_size_t _max_point_count{0};

  struct InMemoryFrame {
    std::atomic_size_t frame_index{std::numeric_limits<size_t>::max()};
    std::atomic_bool loaded{false};
    pc::types::PointCloud point_cloud;
  };

  std::vector<InMemoryFrame> _frame_buffer;
  std::atomic_size_t _current_frame;
  std::atomic_size_t _last_scheduled_frame;
  std::atomic_bool _buffer_initialised;
  pc::types::PointCloud _last_point_cloud{};

  tbb::task_arena _ply_loader_arena;
  std::atomic_size_t _ply_load_command_id{0};

  PlySequencePlayerImplDeviceMemory *_device_memory{nullptr};
  std::mutex _device_memory_access;
  std::atomic_bool _device_memory_ready{false};

  bool _playing{true};
  bool _advance_once{false};

  bool init_device_memory(size_t point_count);
  void free_device_memory();
  void ensure_device_memory_capacity(size_t point_count);

  void set_frame_buffer_capacity(size_t capacity);

  void schedule_load(size_t frame_index);
  void load_single_frame(size_t frame_index, size_t buffer_index);

};

} // namespace pc::devices