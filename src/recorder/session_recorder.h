#pragma once

#include <atomic>
#include <functional>
#include <logger/logger.h>
#include <memory>
#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <readerwriterqueue/readerwriterqueue.h>
#include <string_view>
#include <thread>

namespace pc {
class Workspace;
}

namespace pc::recorder {

enum class OutputFileType { PLY };

class SessionRecorder {
public:
  explicit SessionRecorder(Workspace *workspace);

  void start_recording();
  void stop_recording();

  bool is_recording();
  bool is_file_writing();

  size_t current_recording_frame() const;
  float current_recording_seconds() const;

  size_t writer_queue_depth() const;
  size_t dropped_frames() const;

  void set_recording_changed_callback(std::function<void(bool)> callback);

private:
  Workspace *_workspace;
  std::function<void(bool)> _recording_changed_callback{};

  std::atomic<bool> _recording = false;
  std::atomic<bool> _file_writing = false;
  std::atomic<size_t> _current_frame{0};

  // TODO these should be inside a configuration structure
  std::string _output_directory = "C:/pointcaster/recordings";
  OutputFileType _output_file_type = OutputFileType::PLY;
  bool _parallelise_writes = true;

  struct DeviceFrame {
    std::string_view device_name;
    std::shared_ptr<PointCloud> data;
  };
  struct Frame {
    size_t frame_index;
    std::vector<DeviceFrame> devices;
  };
  static constexpr size_t frame_queue_max{100};
  moodycamel::BlockingReaderWriterQueue<Frame> _frame_queue{frame_queue_max};

  std::atomic<size_t> _queue_depth{0};
  std::atomic<size_t> _dropped_frames{0};

  std::atomic<size_t> _writes_in_flight{0};

  std::jthread _recorder_thread;

  void recorder_thread_work(std::stop_token stop_token);
  void file_writer_thread_work(std::stop_token stop_token);

  static void write_ply(const std::string &file_path,
                        const std::shared_ptr<PointCloud> cloud);
};

} // namespace pc::recorder