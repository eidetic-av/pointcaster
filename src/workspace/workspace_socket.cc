#include "workspace_socket.h"
#include "workspace.h"

#include <memory>
#include <networking/zmq_context.h>
#include <zmq.hpp>
#include <zpp_bits.h>

namespace pc {

WorkspaceSocket
WorkspaceSocket::create_subscriber(const std::string_view filter) {
  return WorkspaceSocket(filter);
}

WorkspaceSocket::WorkspaceSocket(const std::string_view filter) {
  auto &ctx = networking::zmq_context();
  _socket.reset(new zmq::socket_t{ctx, zmq::socket_type::sub});
  _socket->set(zmq::sockopt::subscribe, filter);
  // thread wait time to receive a msg
  _socket->set(zmq::sockopt::rcvtimeo, 100);
  _socket->connect("inproc://workspace");
}

namespace {
std::optional<WorkspaceSocket::SocketUpdate>
deserialize_config_value(zmq::message_t &msg) {
  const std::span<const std::byte> bytes{
      static_cast<const std::byte *>(msg.data()), msg.size()};
  zpp::bits::in deserialize{bytes};

  std::pair<std::string, ConfigValue> kvp;
  const auto result = deserialize(kvp);
  if (zpp::bits::failure(result)) {
    pc::logger()->warn("Failed to deserialize a published value");
    return std::nullopt;
  }
  return kvp;
}
} // namespace

std::optional<WorkspaceSocket::SocketUpdate> WorkspaceSocket::receive() {
  zmq::message_t msg;
  const auto received = _socket->recv(msg);
  if (!received || msg.size() == 0) return std::nullopt;
  return deserialize_config_value(msg);
}

std::optional<WorkspaceSocket::SocketUpdate> WorkspaceSocket::try_receive() {
  zmq::message_t msg;
  const auto received = _socket->recv(msg, zmq::recv_flags::dontwait);
  if (!received || msg.size() == 0) return std::nullopt;
  return deserialize_config_value(msg);
}

} // namespace pc