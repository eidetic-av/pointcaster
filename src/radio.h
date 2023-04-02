#include <memory>
#include <thread>
#include <vector>
#include <utility>

namespace bob::pointcaster {

struct RadioConfiguration {
  int port;
  bool compress_frames;
  bool capture_stats;

  bool operator==(const RadioConfiguration other) const {
    return port == other.port && compress_frames == other.compress_frames &&
	   capture_stats == other.capture_stats;
  }
  bool operator!=(const RadioConfiguration other) const {
    return !operator==(other);
  }
};

class Radio {
public:
  Radio();
  Radio(const RadioConfiguration config);

  void draw_imgui_window();

private:
  // default config
  RadioConfiguration _current_config{
      .port = 9999, .compress_frames = true, .capture_stats = false};

  std::unique_ptr<std::jthread> _radio_thread;

  std::unique_ptr<std::jthread> make_thread(const RadioConfiguration config);
  void set_config(const RadioConfiguration config);
};
} // namespace bob::pointcaster
