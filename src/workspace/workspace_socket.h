#pragma once
#include <config/config_registry.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <zmq.hpp>

namespace pc {

class Workspace;

class WorkspaceSocket {
public:
  static WorkspaceSocket create_subscriber(const std::string_view filter = "");

  using SocketUpdate = std::pair<std::string, ConfigValue>;

  std::optional<SocketUpdate> receive();

private:
  std::unique_ptr<zmq::socket_t> _socket;
  explicit WorkspaceSocket(std::string_view filter);
};
} // namespace pc