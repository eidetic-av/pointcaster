#pragma once

#include "../../structs.h"
#include "../device.h"
#include "ply_sequence_player_config.gen.h"
#include <functional>
#include <memory>
#include <mutex>
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

  size_t _cpu_buffer_capacity;
  std::vector<std::optional<pc::types::PointCloud>> _frame_buffer;
  std::mutex _frame_buffer_access;
  std::unique_ptr<std::atomic_bool[]> _buffer_index_ready;
  std::atomic_size_t _current_frame;
  std::atomic_size_t _last_index{0};
  std::atomic_size_t _loaded_frame_offset;
  std::jthread _loader_thread;
  std::condition_variable _loader_should_buffer;

  PlySequencePlayerImplDeviceMemory *_device_memory{nullptr};
  std::mutex _device_memory_access;
  std::atomic_bool _device_memory_ready{false};

  size_t _last_selected_frame{0};
  bool _playing{true};
  bool _advance_once{false};

  bool init_device_memory(size_t point_count);
  void free_device_memory();
  void ensure_device_memory_capacity(size_t point_count);

  void set_frame_buffer_capacity(size_t capacity);

  void clear_buffer_ready_flags();
  size_t fill_buffer();

};

} // namespace pc::devices