#include "sync_server.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../network_globals.h"
#include "../parameters.h"
#include "../publisher/publisher.h"
#include "../string_utils.h"
#include <system_error>
#include <zpp_bits.h>

namespace pc::client_sync {

using namespace pc::parameters;

SyncServer::SyncServer(SyncServerConfiguration &config) : _config(config) {
  _config.port = std::clamp(_config.port, 1024, 49151);
  declare_parameters("sync", _config);
  publisher::add(this);

  _sync_thread =
      std::make_unique<std::jthread>([this](std::stop_token stop_token) {
        auto socket = zmq::socket_t(network_globals::get_zmq_context(),
                                    zmq::socket_type::router);
        const auto socket_address = fmt::format("tcp://*:{}", _config.port);

        pc::logger->info("Starting thread for sync server at '{}'",
                         socket_address);
        try {
          socket.bind(socket_address);
        } catch (const zmq::error_t &e) {
          pc::logger->error(e.what());
          return;
        }

        using namespace std::chrono_literals;
        static constexpr auto send_queue_wait_time = 5ms;
        static constexpr auto recv_wait_ms = 5;

        socket.set(zmq::sockopt::rcvtimeo, recv_wait_ms);

        while (!stop_token.stop_requested()) {

          // receive any incoming message requests
          zmq::message_t msg;
          if (socket.recv(msg)) {
            bool more = socket.get(zmq::sockopt::rcvmore);
            if (more) {
              // if it's a multipart message, the first part is the sender's id
              auto msg_buf = static_cast<const char *>(msg.data());
              std::string client_id(msg_buf, msg_buf + msg.size());

              if (socket.recv(msg)) {
                auto unpacked = msgpack::unpack(
                    static_cast<const char *>(msg.data()), msg.size());
                const auto &obj = unpacked.get();

                MessageType message_type;
                if (obj.type == msgpack::type::BIN) {
                  std::array<uint8_t, 1> message_type_buffer;
                  obj.convert(message_type_buffer);
                  message_type =
                      static_cast<MessageType>(message_type_buffer[0]);
                } else if (obj.type == msgpack::type::ARRAY) {
                  message_type = static_cast<MessageType>(
                      obj.via.array.ptr[0].as<uint8_t>());
                }
                pc::logger->debug("message_type_byte: {}",
                                  fmt::format("{:02x}", (char)message_type));

                if (message_type == MessageType::Connected) {
                  _connected_client_ids.insert(client_id);
                  pc::logger->info("Sync client '{}' connected", client_id);
                  continue;
                } else if (message_type == MessageType::ClientHeartbeat) {
                  pc::logger->debug("Heartbeat from: {}", client_id);
                  _connected_client_ids.insert(client_id);
                  enqueue_signal(MessageType::ClientHeartbeatResponse);
                  continue;
                } else if (message_type == MessageType::ParameterRequest) {
                  auto parameter_id =
                      unpacked.get().via.array.ptr[1].as<std::string_view>();
                  pc::logger->debug("'{}' requesting: '{}'", client_id,
                                    parameter_id);

                  if (parameter_bindings.contains(parameter_id)) {
                    auto value = parameter_bindings.at(parameter_id).value;

                    if (std::holds_alternative<FloatReference>(value)) {
                      enqueue_parameter_update(ParameterUpdate{
                          std::string{parameter_id},
                          std::get<FloatReference>(value).get()});
                    } else if (std::holds_alternative<IntReference>(value)) {
                      enqueue_parameter_update(
                          ParameterUpdate{std::string{parameter_id},
                                          std::get<IntReference>(value).get()});
                    } else if (std::holds_alternative<Float3Reference>(value)) {
                      enqueue_parameter_update(ParameterUpdate{
                          std::string{parameter_id},
                          std::get<Float3Reference>(value).get()});
                    }
                  }
                }
              };
            }
          }

          // send any outgoing messages
          SyncMessage message;
          while (_messages_to_send.wait_dequeue_timed(message,
                                                      send_queue_wait_time)) {

            if (_connected_client_ids.empty()) { continue; }

            // serialize the outgoing message
            auto [buffer, out] = zpp::bits::data_out();
            out(message);

            for (const auto &client_id : _connected_client_ids) {
              socket.send(zmq::message_t{client_id}, zmq::send_flags::sndmore);
              socket.send(zmq::message_t{buffer.data(), buffer.size()},
                           zmq::send_flags::none);
            }
          }
        }

        pc::logger->info("Stopped thread for sync server");
      });
}

void SyncServer::enqueue_parameter_update(ParameterUpdate update) {
  _messages_to_send.enqueue(update);
}

void SyncServer::enqueue_signal(MessageType signal) {
  _messages_to_send.enqueue(signal);
}

SyncServer::~SyncServer() { _sync_thread->request_stop(); }

void SyncServer::draw_imgui_window() {

  ImGui::Begin("Sync", nullptr);

  if (pc::gui::draw_parameters("sync",
                               struct_parameters.at(std::string{"sync"}))) {
    _config.port = std::clamp(_config.port, 1024, 49151);
  }

  ImGui::End();
}

} // namespace pc::client_sync
