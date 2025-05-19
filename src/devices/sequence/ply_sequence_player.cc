#include "ply_sequence_player.h"
#include "../../logger.h"
#include "../../parameters.h"
#include "../../profiling.h"
#include "../../uuid.h"
#include <algorithm>
#include <atomic>
#include <execution>
#include <filesystem>
#include <functional>
#include <happly.h>
#include <memory>
#include <nfd.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <thread>

namespace pc::devices {

static pc::types::PointCloud
load_ply_to_pointcloud(const std::string &file_path) {
  using namespace pc::profiling;
  ProfilingZone profiling_zone("PLYSequencePlayer::load_ply_to_pointcloud");
  profiling_zone.text(file_path);
  pc::logger->debug("Loading: {}", file_path);

  ProfilingZone load_and_parse_zone("Load and parse");
  happly::PLYData ply_in(file_path);
  pc::types::PointCloud cloud;

  constexpr auto vertex = "vertex";
  const auto x_values = ply_in.getElement(vertex).getProperty<float>("x");
  const auto y_values = ply_in.getElement(vertex).getProperty<float>("y");
  const auto z_values = ply_in.getElement(vertex).getProperty<float>("z");
  const auto r_values = ply_in.getElement(vertex).getProperty<unsigned char>("red");
  const auto g_values = ply_in.getElement(vertex).getProperty<unsigned char>("green");
  const auto b_values = ply_in.getElement(vertex).getProperty<unsigned char>("blue");

  const auto point_count = x_values.size();
  cloud.positions.resize(point_count);
  cloud.colors.resize(point_count);

  {
    ProfilingZone convert_zone("Convert and pack points");
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, point_count),
        [&](const tbb::blocked_range<size_t> &range) {
          for (size_t i = range.begin(), e = range.end(); i < e; ++i) {
            cloud.positions[i] = {static_cast<short>(x_values[i] * 1000),
                                  static_cast<short>(y_values[i] * 1000),
                                  static_cast<short>(z_values[i] * 1000)};
            cloud.colors[i] = {b_values[i], g_values[i], r_values[i]};
          }
        });
  }

  return cloud;
}

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
    : DeviceBase<PlySequencePlayerConfiguration>(config),
      _cpu_buffer_capacity(static_cast<size_t>(config.buffer_capacity)),
      _current_frame(config.current_frame),
      _loaded_frame_offset(config.current_frame) {

  _frame_buffer.resize(_cpu_buffer_capacity);

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
    pc::logger->info("Found {} frames", _file_paths.size());
  }

  constexpr auto default_point_buffer_size = 200'000;
  init_device_memory(default_point_buffer_size);

  _buffer_index_ready =
      std::make_unique<std::atomic_bool[]>(_cpu_buffer_capacity);
  clear_buffer_ready_flags();


  _loader_thread = std::jthread([this](std::stop_token stop_token) {
    using namespace pc::profiling;
    bool needs_initial_fill = true;
    while (!stop_token.stop_requested()) {
      std::unique_lock buffer_lock(_frame_buffer_access);
      ProfilingZone load_buffer_zone("_loader_thread");
      {
        ProfilingZone wait_zone("Spin");
        _loader_should_buffer.wait(buffer_lock, [&] {
          return stop_token.stop_requested() || needs_initial_fill ||
                 _current_frame < _loaded_frame_offset ||
                 _current_frame >= _loaded_frame_offset + _cpu_buffer_capacity;
        });
      }
      size_t wanted_frame = _current_frame.load();
      if (wanted_frame < _loaded_frame_offset ||
          wanted_frame >= _loaded_frame_offset + _cpu_buffer_capacity) {
        _loaded_frame_offset = wanted_frame;
        clear_buffer_ready_flags();
      }
      {
        ProfilingZone fill_zone("Fill");
        const auto max_point_count = fill_buffer();
        const auto previous_max_point_count =
            _max_point_count.load(std::memory_order_relaxed);
        if (previous_max_point_count < max_point_count) {
          _max_point_count.store(max_point_count, std::memory_order_relaxed);
          ensure_device_memory_capacity(max_point_count);
        }
        needs_initial_fill = false;
      }
    }
  });

  {
    std::lock_guard lock(PlySequencePlayer::devices_access);
    PlySequencePlayer::attached_devices.push_back(std::ref(*this));
  }
  // // parameters::set_parameter_minmax(std::format("{}.current_frame", id()),
  // 0,
  // //                                  frame_count());
}

PlySequencePlayer::~PlySequencePlayer() {
  if (_loader_thread.joinable()) {
    _loader_thread.request_stop();
    _loader_should_buffer.notify_all();
  }
  free_device_memory();
  std::lock_guard lock(PlySequencePlayer::devices_access);
  std::erase_if(PlySequencePlayer::attached_devices,
                [this](auto &device) { return &(device.get()) == this; });
}

void PlySequencePlayer::set_frame_buffer_capacity(size_t capacity) {
  using namespace pc::profiling;
  ProfilingZone set_capacity_zone("PlySequencePlayer::set_frame_buffer_capacity");
  std::lock_guard lock(_frame_buffer_access);
  if (_loaded_frame_offset >= capacity) {
    _loaded_frame_offset = 0;
  }
  _cpu_buffer_capacity = capacity;
  _frame_buffer.clear();
  _frame_buffer.resize(capacity);
  _buffer_index_ready = std::make_unique<std::atomic_bool[]>(capacity);
  clear_buffer_ready_flags();
  _loader_should_buffer.notify_one();
}

void PlySequencePlayer::clear_buffer_ready_flags() {
    for (size_t i = 0; i < _cpu_buffer_capacity; i++) {
      _buffer_index_ready[i].store(false, std::memory_order_relaxed);
    }
}

size_t PlySequencePlayer::fill_buffer() {
    const size_t total_frames = frame_count();
    if (total_frames == 0) return 0;

    const size_t window = std::min(_cpu_buffer_capacity, total_frames);
    const size_t frame_offset = _loaded_frame_offset;

    // load the ply to our CPU pointcloud frame buffer
    // - use parallel reduce and have the reduce keep track
    // of the maximum point count
    // (this is used to make sure our GPU device memory is sized appropriately)

    // contrain all concurrent calls to fill_buffer to the same
    // number of threads.
    static tbb::task_arena fill_buffer_arena {
      std::max<int>(1, std::thread::hardware_concurrency() / 2)
    };

    size_t max_point_count = 0;
    fill_buffer_arena.execute([&] {
      max_point_count = tbb::parallel_reduce(
          tbb::blocked_range<size_t>(0, window), size_t(0),
          [&](auto const &range,
              size_t thread_local_max_point_count) -> size_t {
            for (size_t i = range.begin(), e = range.end(); i < e; ++i) {
              if (!_buffer_index_ready[i].load(std::memory_order_acquire)) {
                const size_t frame_idx = (frame_offset + i) % total_frames;
                auto cloud = load_ply_to_pointcloud(_file_paths[frame_idx]);
                thread_local_max_point_count =
                    std::max(thread_local_max_point_count, cloud.size());
                _frame_buffer[i].emplace(std::move(cloud));
                _buffer_index_ready[i].store(true, std::memory_order_release);
              }
            }
            return thread_local_max_point_count;
          },
          [](size_t a, size_t b) -> size_t { return std::max(a, b); });
    });

    return max_point_count;
}

size_t PlySequencePlayer::frame_count() const { return _file_paths.size(); }

size_t PlySequencePlayer::current_frame() const {
  return config().current_frame;
}

DeviceStatus PlySequencePlayer::status() const {
  if (_device_memory_ready) return DeviceStatus::Loaded;
  return DeviceStatus::Inactive;
}

void PlySequencePlayer::tick(float delta_time) {
  auto &conf = config();
  if (!conf.playing) return;
  _frame_accumulator += delta_time * conf.frame_rate;
  const size_t frames_to_advance = static_cast<size_t>(_frame_accumulator);
  _frame_accumulator -= frames_to_advance;
  const size_t total_frames = frame_count();
  auto current_frame_index = conf.current_frame;
  if (total_frames > 0) {
    auto new_frame_index =
        (current_frame_index + frames_to_advance) % total_frames;
    conf.current_frame = new_frame_index;
    _current_frame.store(new_frame_index, std::memory_order_relaxed);
    if (conf.buffer_capacity != _cpu_buffer_capacity) {
      set_frame_buffer_capacity(conf.buffer_capacity);
    }
    _loader_should_buffer.notify_one();
  }
}

} // namespace pc::devices