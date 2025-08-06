#include "ply_sequence_player.h"
#include "../../logger.h"
#include "../../parameters.h"
#include "../../profiling.h"
#include "../../uuid.h"
#include "ply_sequence_player_config.gen.h"
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <functional>
#include <happly.h>
#include <limits>
#include <memory>
#include <nfd.hpp>
#include <optional>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <utility>

namespace pc::devices {

static pc::types::PointCloud
load_ply_to_pointcloud(const std::string &file_path) {
  using namespace pc::profiling;
  ProfilingZone profiling_zone("PLYSequencePlayer::load_ply_to_pointcloud");
  profiling_zone.text(file_path);
  pc::logger->debug("Loading: {}", file_path);

  size_t point_count;
  std::vector<float> x_values, y_values, z_values;
  std::vector<unsigned char> r_values, g_values, b_values;
  pc::types::PointCloud cloud;
  std::optional<happly::PLYData> ply_in;

  {
    ProfilingZone load_file_zone("Load file");
    ply_in.emplace(file_path);
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
      _current_frame(config.current_frame), _last_scheduled_frame(0),
      _ply_loader_arena(config.io_threads), _buffer_initialised{false} {

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

  // TODO
  // set_frame_buffer_capacity(config.buffer_capacity);
  set_frame_buffer_capacity(_file_paths.size());

  constexpr auto default_point_buffer_size = 350'000;
  init_device_memory(default_point_buffer_size);

  {
    std::lock_guard lock(PlySequencePlayer::devices_access);
    PlySequencePlayer::attached_devices.push_back(std::ref(*this));
  }
  // // parameters::set_parameter_minmax(std::format("{}.current_frame", id()),
  // 0,
  // //                                  frame_count());
}

PlySequencePlayer::~PlySequencePlayer() {
  free_device_memory();
  std::lock_guard lock(PlySequencePlayer::devices_access);
  std::erase_if(PlySequencePlayer::attached_devices,
                [this](auto &device) { return &(device.get()) == this; });
}

void PlySequencePlayer::set_frame_buffer_capacity(size_t capacity) {
  std::vector<InMemoryFrame> new_buffer(capacity);
  std::swap(_frame_buffer, new_buffer);
}

void PlySequencePlayer::schedule_load(size_t frame_index) {
  size_t command_id = _ply_load_command_id.load();
  const auto i = frame_index % config().buffer_capacity;
  auto &frame = _frame_buffer[i];
  if (frame.frame_index.load() == frame_index && frame.loaded.load()) {
    return;
  }
  frame.loaded.store(false);
  frame.frame_index.store(frame_index);
  _ply_loader_arena.enqueue([this, frame_index, i, command_id] {
    if (_ply_load_command_id.load() != command_id) { return; }
    load_single_frame(frame_index, i);
    if (!_buffer_initialised.load(std::memory_order_acquire)) {
      const size_t frames_loaded =
          std::ranges::count_if(_frame_buffer, [](auto &f) {
            return f.loaded.load(std::memory_order_relaxed);
          });
      const size_t consider_initialised_frame_count =
          std::min<size_t>(30, _file_paths.size());
      if (frames_loaded >= consider_initialised_frame_count) {
        _buffer_initialised.store(true, std::memory_order_release);
      }
    }
  });
}

void PlySequencePlayer::load_single_frame(size_t frame_index,
                                          size_t buffer_index) {
  using namespace pc::profiling;
  ProfilingZone load_zone("load_single_frame");

  auto cloud = load_ply_to_pointcloud(_file_paths[frame_index]);
  const size_t point_count = cloud.size();

  const size_t current_max_points =
      _max_point_count.load(std::memory_order_relaxed);
  if (point_count > current_max_points) {
    _max_point_count.store(point_count, std::memory_order_relaxed);
    ensure_device_memory_capacity(point_count);
  }

  auto &frame = _frame_buffer[buffer_index];
  frame.point_cloud = std::move(cloud);
  frame.loaded.store(true);
}

size_t PlySequencePlayer::frame_count() const { return _file_paths.size(); }

size_t PlySequencePlayer::current_frame() const {
  return config().current_frame;
}

DeviceStatus PlySequencePlayer::status() const {
  if (_device_memory_ready) return DeviceStatus::Loaded;
  return DeviceStatus::Inactive;
}

std::optional<std::reference_wrapper<pc::types::PointCloud>> PlySequencePlayer::last_point_cloud() {
  if (_last_point_cloud.empty()) return std::nullopt;
  return std::ref(_last_point_cloud);
};

void PlySequencePlayer::tick(float delta_time) {
  auto &conf = config();
  auto force_update = false;

  static std::map<std::string, PlySequencePlayerConfiguration> last_configs;
  auto &last_config = last_configs[conf.id];
  if (last_config.id.empty()) last_config = conf;

  if (conf.streaming.use_static_channel) {
    // if the configuration is different from last time we ticked, static
    // streamers need to update to their clients
    force_update = last_config.current_frame != conf.current_frame ||
                   last_config.streaming.use_static_channel !=
                       conf.streaming.use_static_channel ||
                   last_config.transform != conf.transform ||
                   last_config.color != conf.color;
    conf.streaming.requires_update |= force_update;
  }

  last_config = conf;

  if (!force_update && !conf.playing && !_playing && !_advance_once) { return; }

  // advance accumulator
  _frame_accumulator += delta_time * conf.frame_rate;
  const size_t frames_to_advance = static_cast<size_t>(_frame_accumulator);
  _frame_accumulator -= frames_to_advance;

  const size_t total_frames = frame_count();
  const auto buffer_capacity = static_cast<size_t>(conf.buffer_capacity);
  if (_frame_buffer.size() < buffer_capacity) {
    set_frame_buffer_capacity(buffer_capacity);
  }
  const auto prefetch_count = std::min<size_t>(buffer_capacity, total_frames);
  const size_t start_frame =
      static_cast<size_t>(conf.start_frame) % total_frames;

  // compute where we are inside the buffer (0 … capacity-1)
  size_t rel_current;
  {
    size_t cf = conf.current_frame;
    rel_current = (cf + total_frames - start_frame) % total_frames;
    rel_current = std::min(rel_current, prefetch_count - 1);
  }

  // advance relative index
  size_t raw_rel = rel_current + (conf.playing ? frames_to_advance : 0);
  size_t rel_next;
  if (conf.playing && conf.looping) {
    rel_next = raw_rel % prefetch_count;
  } else {
    rel_next = std::min(raw_rel, prefetch_count - 1);
  }

  // map back to absolute file index
  size_t next_frame_index = (start_frame + rel_next) % total_frames;

  if (next_frame_index < start_frame) {
    next_frame_index = start_frame;
  }

  // cancel if we’ve wrapped or rewound in the ring
  size_t last_rel = _last_scheduled_frame.load(std::memory_order_relaxed);
  if (rel_next <= last_rel) {
    _ply_load_command_id.fetch_add(1, std::memory_order_relaxed);
    _last_scheduled_frame.store(rel_next - 1, std::memory_order_relaxed);
  }

  conf.current_frame = next_frame_index;
  _current_frame.store(next_frame_index, std::memory_order_relaxed);

  // schedule the next window in the ring
  size_t buffer_start =
      _last_scheduled_frame.load(std::memory_order_relaxed) + 1;
  size_t buffer_end = rel_next + prefetch_count;
  for (size_t off = buffer_start; off <= buffer_end; ++off) {
    size_t rel = off % buffer_capacity;
    size_t abs = (start_frame + rel) % total_frames;
    schedule_load(abs);
  }
  _last_scheduled_frame.store(buffer_end, std::memory_order_relaxed);

  _playing = conf.playing;
  _advance_once = false;
}

} // namespace pc::devices