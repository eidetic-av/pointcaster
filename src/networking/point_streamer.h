#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace pc {
class Workspace;
}

namespace pc::networking {

class PointStreamer {
public:
  explicit PointStreamer(Workspace &workspace);

  bool has_listeners(const std::string &channel_address) const;

  std::shared_ptr<const std::unordered_map<std::string, int>>
  subscriber_counts() const {
    return _subscriber_counts.load(std::memory_order_acquire);
  }

private:
  std::atomic<std::shared_ptr<const std::unordered_map<std::string, int>>>
      _subscriber_counts;
  std::jthread _worker;
};

} // namespace pc::networking
