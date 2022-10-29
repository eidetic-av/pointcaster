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

  ~Radio();

  void drawImGuiWindow();

private:
  // default config
  RadioConfiguration current_config{
      .port = 9999, .compress_frames = true, .capture_stats = false};

  std::unique_ptr<std::jthread> radio_thread;

  std::unique_ptr<std::jthread> makeThread(const RadioConfiguration config);
  void setConfig(const RadioConfiguration config);
};
} // namespace bob::pointcaster
