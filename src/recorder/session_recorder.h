#pragma once

#include <functional>
#include <logger/logger.h>

namespace pc {
class Workspace;
}

namespace pc::recorder {

class SessionRecorder {
public:
  explicit SessionRecorder(Workspace *workspace);

  bool recording() { return _recording; }

  void start_recording() {
    _recording = true;
    if (_recording_changed_callback) _recording_changed_callback(_recording);
    pc::logger()->debug("started recording");
  }

  void stop_recording() {
    _recording = false;
    if (_recording_changed_callback) _recording_changed_callback(_recording);
    pc::logger()->debug("stopped recording");
  }

  void set_recording_changed_callback(std::function<void(bool)> callback) {
    _recording_changed_callback = callback;
  }

private:
  Workspace *_workspace;
  bool _recording = false;
  std::function<void(bool)> _recording_changed_callback{};
};

} // namespace pc::recorder