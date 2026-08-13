#pragma once

#include <thread>

namespace pc {
class Workspace;
}

namespace pc::publishers {
class OscSender {
public:
  explicit OscSender(Workspace &workspace);

private:
  std::jthread _worker;
};
} // namespace pc::publishers
