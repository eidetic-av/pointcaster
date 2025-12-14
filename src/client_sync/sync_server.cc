#include "sync_server.h"
#include "../devices/device.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../network_globals.h"
#include "../parameters.h"
#include "../profiling.h"
#include "../publisher/publisher.h"
#include <zpp_bits.h>

namespace pc::client_sync {

using namespace pc::parameters;

SyncServer::SyncServer(SyncServerConfiguration &config) : _config(config) {
  _config.port = std::clamp(_config.port, 1024, 49151);
  declare_parameters("workspace", "sync", _config);
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
            // it should be a multipart message, because the first part is the
            // sender's id
            auto id_ptr = static_cast<const char *>(msg.data());
            std::string client_id(id_ptr, id_ptr + msg.size());
            // now get the actual msg
            bool more = socket.get(zmq::sockopt::rcvmore);
            if (!more || !socket.recv(msg)) {
              pc::logger->error("Client '{}' sent malformed message", client_id);
            } else {
              // unpack the sync message variant
              std::vector<std::byte> buffer(msg.size());
              std::memcpy(buffer.data(), msg.data(), msg.size());
              zpp::bits::in in(buffer);
              SyncMessage message;
              if (failure(in(message))) {
                pc::logger->error("Failed to deserialize client sync message");
              } else {
                // in a server receiving a SyncMessage, it has to be a
                // MessageType not a ParameterUpdate, so if std::get throws here
                // that's fine, its a serious problem
                auto message_type = std::get<MessageType>(message);
                pc::logger->debug("message_type_byte: {}",
                                  fmt::format("{:02x}", (char)message_type));
                if (message_type == MessageType::Connected) {
                  _connected_client_ids.insert(client_id);
                  pc::logger->info("Sync client '{}' connected", client_id);
                  // TODO the new client needs static point cloud frames,
                  // but atm we just send to *all* clients
                  pc::devices::send_all_static_point_clouds();
                  for (auto &client_connect_callback : on_client_connected) {
                    client_connect_callback();
                  }
                } else if (message_type == MessageType::ClientHeartbeat) {
                  pc::logger->debug("Heartbeat from: {}", client_id);
                  _connected_client_ids.insert(client_id);
                  enqueue_signal(MessageType::ClientHeartbeatResponse);
                } else if (message_type == MessageType::ParameterRequest) {

                  // TODO
                  pc::logger->debug("Got parameter request");

                  // when we get a parameter request, publish everything that is
                  // set to be published!

                  // std::for_each(parameter_states.begin(),
                  //               parameter_states.end(), [&](auto &&kvp) {
                  //                 const auto &[id, state] = kvp;
                  //                 if (state == ParameterState::Publish) {
                  //                   auto it = parameter_bindings.find(id);
                  //                   if (it != parameter_bindings.end()) {
                  //                     std::visit(
                  //                         [&](auto &&value) {
                  //                           using P = std::decay_t<
                  //                               decltype(value.get())>;
                  //                           if constexpr (
                  //                               is_publishable_container_v<P> ||
                  //                               pc::types::ScalarType<P> ||
                  //                               pc::types::VectorType<P>) {
                  //                             publisher::publish_all(
                  //                                 id, value.get());
                  //                             pc::logger->debug("Sent: ", id);
                  //                           }
                  //                         },
                  //                         it->second.value);
                  //                   }
                  //                 }
                  //               });

                  // auto parameter_id =
                  //     unpacked.get().via.array.ptr[1].as<std::string_view>();
                  // pc::logger->debug("'{}' requesting: '{}'", client_id,
                  //                   parameter_id);

                  // if (parameter_bindings.contains(parameter_id)) {
                  //   auto value = parameter_bindings.at(parameter_id).value;

                  //   if (std::holds_alternative<FloatReference>(value)) {
                  //     enqueue_parameter_update(ParameterUpdate{
                  //         std::string{parameter_id},
                  //         std::get<FloatReference>(value).get()});
                  //   } else if (std::holds_alternative<IntReference>(value)) {
                  //     enqueue_parameter_update(
                  //         ParameterUpdate{std::string{parameter_id},
                  //                         std::get<IntReference>(value).get()});
                  //   } else if (std::holds_alternative<Float3Reference>(value)) {
                  //     enqueue_parameter_update(ParameterUpdate{
                  //         std::string{parameter_id},
                  //         std::get<Float3Reference>(value).get()});
                  //   }
                  // }
                }
              }
            }
          }
          // send any outgoing messages
          SyncMessage message;
          while (_messages_to_send.wait_dequeue_timed(message,
                                                      send_queue_wait_time)) {

            if (_connected_client_ids.empty()) { continue; }

            using namespace pc::profiling;
            ProfilingZone serialize_and_send_zone("serialize_and_send");

            // serialize the outgoing message
            auto [buffer, out] = zpp::bits::data_out();
            const auto serialize_result = out(message);
            if (failure(serialize_result)) {
              pc::logger->error("Failed to serialize sync message.");
            } else {
              // and send
              for (const auto &client_id : _connected_client_ids) {
                using namespace zmq;
                socket.send(message_t{client_id}, send_flags::sndmore);
                socket.send(message_t{buffer}, send_flags::none);
              }
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

void SyncServer::publish_endpoint(EndpointUpdate update) {
  _messages_to_send.enqueue(update);
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
