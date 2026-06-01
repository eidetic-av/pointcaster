#pragma once

#include <thread>

namespace pc {
class Workspace;
}

namespace pc::networking {

class PointStreamer {
public:
  explicit PointStreamer(Workspace &workspace);

private:
  std::jthread _worker;
};

} // namespace pc::networking