#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <util/string_map.h>

namespace pc {
class Workspace;
} // namespace pc

namespace pc::networking {

class PointStreamer {
public:
  explicit PointStreamer(Workspace &workspace);

  bool has_listeners(const std::string &channel_address) const;

  std::shared_ptr<const StringMap<int>> subscriber_counts() const;

private:
  std::atomic<std::shared_ptr<const StringMap<int>>> _subscriber_counts;
  std::jthread _worker;
};

} // namespace pc::networking
