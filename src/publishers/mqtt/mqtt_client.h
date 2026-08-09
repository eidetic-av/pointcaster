#pragma once

#include <thread>

namespace pc {
class Workspace;
}

namespace pc::publishers {
class MqttClient {
public:
  explicit MqttClient(Workspace &workspace);

private:
  std::jthread _worker;
};
} // namespace pc::publishers