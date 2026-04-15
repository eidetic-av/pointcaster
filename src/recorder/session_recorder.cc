#include "session_recorder.h"
#include "workspace/workspace.h"
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <plugins/backend/cpu/cpu_backend.h>
#include <plugins/devices/device_plugin.h>
#include <profiling/profiling_zone.h>
#include <ranges>
#include <thread>
#include <unordered_map>
#include <variant>

namespace pc::recorder {

using namespace pc::profiling;

SessionRecorder::SessionRecorder(Workspace *workspace)
    : _workspace(workspace) {

      };

void SessionRecorder::start_recording() {
  _current_frame.store(0, std::memory_order_relaxed);
  _queue_depth.store(0, std::memory_order_relaxed);
  _dropped_frames.store(0, std::memory_order_relaxed);
  _recording.store(true, std::memory_order_relaxed);
  if (_recording_changed_callback) _recording_changed_callback(_recording);
  _recorder_thread = std::jthread([this](std::stop_token stop_token) { //
    recorder_thread_work(stop_token);
  });
}

void SessionRecorder::stop_recording() {
  _recording.store(false, std::memory_order_relaxed);
  if (_recording_changed_callback) _recording_changed_callback(_recording);
}

bool SessionRecorder::is_recording() {
  return _recording.load(std::memory_order_relaxed);
}

bool SessionRecorder::is_file_writing() {
  return _file_writing.load(std::memory_order_relaxed);
}

size_t SessionRecorder::current_recording_frame() const {
  return _current_frame.load(std::memory_order_relaxed);
}

float SessionRecorder::current_recording_seconds() const {
  return current_recording_frame() / 30.0f;
}

size_t SessionRecorder::writer_queue_depth() const {
  return _queue_depth.load(std::memory_order_relaxed);
}

size_t SessionRecorder::dropped_frames() const {
  return _dropped_frames.load(std::memory_order_relaxed);
}

void SessionRecorder::set_recording_changed_callback(
    std::function<void(bool)> callback) {
  _recording_changed_callback = callback;
}

void SessionRecorder::recorder_thread_work(std::stop_token stop_token) {
  using namespace std::chrono;
  using namespace std::chrono_literals;

  constexpr auto fps = 30;
  constexpr auto frame_duration = 1'000'000us / fps;

  auto next_frame = steady_clock::now();

  // keep a map of device names so we don't need to allocate each frame when we
  // need the device name to write a frame to disk (if recording a sequence of
  // individual files like with PLY)
  std::unordered_map<devices::DevicePlugin *, std::string> device_names;
  for (auto &device : _workspace->devices) {
    std::visit(
        [&](auto &device_config) {
          std::string device_id = device_config.id;
          // sanitize the device_id for our filenames, only allowing
          // alphanumeric and dash and underscore
          std::ranges::replace_if(
              device_id,
              [](char c) { return !std::isalnum(c) && c != '-' && c != '_'; },
              '_');

          // and create directories for each device
          std::filesystem::create_directories(
              std::format("{}/{}", _output_directory, device_id));

          // then populate the id map
          device_names.emplace(device.get(), std::move(device_id));
        },
        device->config());
  }

  // start a file writer thread that waits for this recorder thread to dump
  // device frames into the queue it pops from
  auto file_writer_thread = std::jthread([this](std::stop_token stop_token) {
    file_writer_thread_work(stop_token);
  });

  pc::logger()->trace("started recording");

  while (!stop_token.stop_requested() && is_recording()) {
    next_frame += frame_duration;

    if (_queue_depth.load(std::memory_order_relaxed) < frame_queue_max) {
      std::vector<DeviceFrame> device_frames;
      for (auto &device : _workspace->devices) {
        device_frames.push_back({.device_name = device_names[device.get()],
                                 .data = device->point_cloud()});
      }
      _frame_queue.enqueue(
          Frame{.frame_index = _current_frame.load(std::memory_order_relaxed),
                .devices = std::move(device_frames)});
      _queue_depth.fetch_add(1);
    } else {
      _dropped_frames.fetch_add(1);
    }

    _current_frame.fetch_add(1);

    std::this_thread::sleep_until(next_frame);
  }

  pc::logger()->trace("stopped recording");

  // we still wait for the file_writer_thread to complete...
  // the jthread will request stop here, and that thread will move on to drain
  // the frames to disk and eventually exit
}

void SessionRecorder::file_writer_thread_work(std::stop_token stop_token) {
  using namespace std::chrono_literals;

  _file_writing.store(true, std::memory_order_relaxed);

  const auto write_files = [&](Frame frame) {
    for (auto &device_frame : frame.devices) {
      auto file_path = std::format("{}/{}/{:05d}.ply", _output_directory,
                                   device_frame.device_name, frame.frame_index);

      if (_parallelise_writes) {
        _writes_in_flight.fetch_add(1);
        backend::CpuBackend::thread_pool.detach_task(
            [this, path = std::move(file_path),
             cloud = std::move(device_frame.data)] {
              if (_output_file_type == OutputFileType::PLY) {
                write_ply(path, cloud);
              }
              _writes_in_flight.fetch_sub(1);
              _writes_in_flight.notify_all();
            });
      } else {
        if (_output_file_type == OutputFileType::PLY) {
          write_ply(file_path, device_frame.data);
        }
      }
    }
  };

  while (!stop_token.stop_requested()) {
    Frame frame;
    if (_frame_queue.wait_dequeue_timed(frame, 100ms)) {
      _queue_depth.fetch_sub(1);
      write_files(frame);
    }
  }

  // drain remaining frames after stop is requested
  Frame frame;
  while (_frame_queue.try_dequeue(frame)) {
    _queue_depth.fetch_sub(1);
    write_files(frame);
  }

  while (_writes_in_flight.load() > 0) {
    _writes_in_flight.wait(_writes_in_flight.load());
  }
  _file_writing.store(false, std::memory_order_relaxed);
}

void SessionRecorder::write_ply(const std::string &file_path,
                                const std::shared_ptr<PointCloud> cloud) {

  ProfilingZone write_ply_zone("SessionRecorder::write_ply");

  std::ofstream output_file(file_path, std::ios::binary);
  if (!output_file) {
    pc::logger()->error("failed to open {}", file_path);
    return;
  }

  const auto point_count = cloud->size();

  output_file << "ply\n"
              << "format binary_little_endian 1.0\n"
              << "element vertex " << point_count << "\n"
              << "property short x\n"
              << "property short y\n"
              << "property short z\n"
              << "property uchar red\n"
              << "property uchar green\n"
              << "property uchar blue\n"
              << "end_header\n";

  // 6 bytes pos (skip padding) + 3 bytes color (skip alpha)
  constexpr size_t point_stride = 9;
  std::vector<std::byte> data_buffer(point_count * point_stride);
  auto *data_ptr = data_buffer.data();

  for (size_t i = 0; i < point_count; ++i) {
    std::memcpy(data_ptr + i * point_stride, &cloud->positions[i].x, 6);
    std::memcpy(data_ptr + i * point_stride + 6, &cloud->colors[i].r, 3);
  }

  output_file.write(reinterpret_cast<const char *>(data_buffer.data()),
                    data_buffer.size());
}

} // namespace pc::recorder