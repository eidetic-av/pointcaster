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

  // block to dequeue an update, with a timeout of 100ms, after which nullopt is
  // returned
  std::optional<SocketUpdate> receive();

  // dequeue an update immediately without blocking, or return nullopt if no
  // update is available
  std::optional<SocketUpdate> try_receive();

private:
  std::unique_ptr<zmq::socket_t> _socket;
  explicit WorkspaceSocket(std::string_view filter);
};
} // namespace pc