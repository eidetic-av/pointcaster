#include "sync_server.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../parameters.h"
#include "../publisher/publisher.h"
#include "../string_utils.h"
#include <msgpack.hpp>

namespace pc::client_sync {

using namespace pc::parameters;

msgpack::sbuffer serialize_parameter_update(const ParameterUpdate &update) {
  static const auto param_update_msgtype =
      static_cast<uint8_t>(MessageType::ParameterUpdate);
  msgpack::sbuffer buffer;

  std::visit(
      [&buffer, &update](const auto &val) {
        using T = std::decay_t<decltype(val)>;

	if constexpr (types::ScalarType<T>) {
          std::tuple<uint8_t, std::string_view, T> update_value{
	      param_update_msgtype, update.id, val};
	  msgpack::pack(buffer, update_value);
	} else if constexpr (types::VectorType<T>) {
	  std::array<typename T::vector_type, types::VectorSize<T>::value>
	      array;
	  for (std::size_t el = 0; el < types::VectorSize<T>::value; el++) {
	    array[el] = val[el];
	  }
	  std::tuple<uint8_t, std::string_view, decltype(array)> update_value{
	      param_update_msgtype, update.id, array};
	  msgpack::pack(buffer, update_value);
        } else if constexpr (std::is_same_v<T, std::array<float, 3>>) {
	  std::tuple<uint8_t, std::string_view, std::array<float, 3>>
	      update_value{param_update_msgtype, update.id, val};
	  msgpack::pack(buffer, update_value);
        } else {
          pc::logger->debug("{}", typeid(T).name());
        }
      },
      update.value);

  return buffer;
}

SyncServer::SyncServer(SyncServerConfiguration &config) : _config(config) {
  _config.port = std::clamp(_config.port, 1024, 49151);
  declare_parameters("sync", _config);
  publisher::add(this);

  // TODO: I think zmq sockets are not thread safe
  // So we need to switch the listener thread out for a poller impl.

  _socket_context = std::make_unique<zmq::context_t>();
  _socket = std::make_unique<zmq::socket_t>(*_socket_context,
					    zmq::socket_type::router);

  const auto socket_address = fmt::format("tcp://*:{}", _config.port);
  _socket->bind(socket_address);

  _publish_thread = std::make_unique<std::jthread>([socket_address,
						    this](std::stop_token st) {

    pc::logger->info("Starting sync server publish thread for {}",
		     socket_address);

    using namespace std::chrono_literals;
    static constexpr auto thread_wait_time = 50ms;

    while (!st.stop_requested()) {

      ServerToClientMessage message;

      while (
	  _messages_to_send.wait_dequeue_timed(message, thread_wait_time)) {

        if (_connected_client_ids.empty()) {
	  continue;
	}

	msgpack::sbuffer buffer;

	if (std::holds_alternative<ParameterUpdate>(message)) {
	  buffer =
	      serialize_parameter_update(std::get<ParameterUpdate>(message));
	} else if (std::holds_alternative<MessageType>(message)) {
	  const auto signal_message =
	      static_cast<uint8_t>(std::get<MessageType>(message));
	  msgpack::pack(buffer, std::array<uint8_t, 1>{signal_message});
        }

        for (const auto &client_id : _connected_client_ids) {
	  _socket->send(zmq::message_t{client_id}, zmq::send_flags::sndmore);
	  _socket->send(zmq::message_t{buffer.data(), buffer.size()},
			zmq::send_flags::none);
        }
      }
    }

    pc::logger->info("Stopped sync server publish thread");
  });

  _listener_thread = std::make_unique<std::jthread>([socket_address,
						     this](std::stop_token st) {
    pc::logger->info("Starting sync server listener thread for {}",
		     socket_address);

    static constexpr auto recv_wait_ms = 50;
    _socket->set(zmq::sockopt::rcvtimeo, recv_wait_ms);

    zmq::message_t msg;
    while (!st.stop_requested()) {
      if (_socket->recv(msg)) {

        bool more = _socket->get(zmq::sockopt::rcvmore);
	if (more) {
	  // if it's a multipart message, the first part is the sender's id
	  auto msg_buf = static_cast<const char *>(msg.data());
          std::string client_id(msg_buf, msg_buf + msg.size());

          if (_socket->recv(msg)) {
	    auto unpacked = msgpack::unpack(
		static_cast<const char *>(msg.data()), msg.size());
	    const auto& obj = unpacked.get();

	    MessageType message_type;
	    if (obj.type == msgpack::type::BIN) {
	      std::array<uint8_t, 1> message_type_buffer;
	      obj.convert(message_type_buffer);
	      message_type = static_cast<MessageType>(message_type_buffer[0]);
            } else if (obj.type == msgpack::type::ARRAY) {
	      message_type = static_cast<MessageType>(obj.via.array.ptr[0].as<uint8_t>());
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
            }
            //
            else if (message_type == MessageType::ParameterRequest) {
              auto parameter_id =
		  unpacked.get().via.array.ptr[1].as<std::string_view>();
              pc::logger->debug("'{}' requesting: '{}'", client_id, parameter_id);

              if (parameter_bindings.contains(parameter_id)) {
		auto value = parameter_bindings.at(parameter_id).value;

		if (std::holds_alternative<FloatReference>(value)) {
		  enqueue_parameter_update(
		      ParameterUpdate{std::string{parameter_id},
				      std::get<FloatReference>(value).get()});
                } else if (std::holds_alternative<IntReference>(value)) {
                  enqueue_parameter_update(ParameterUpdate{std::string{parameter_id},
					  std::get<IntReference>(value).get()});
                } else if (std::holds_alternative<Float3Reference>(value)) {
		  enqueue_parameter_update(
		      ParameterUpdate{std::string{parameter_id},
				      std::get<Float3Reference>(value).get()});
                }

              }
            }
          };
        }
      };
    }

    pc::logger->info("Stopped sync server listener thread");
  });
}

void SyncServer::enqueue_parameter_update(ParameterUpdate update) {
  _messages_to_send.enqueue(update);
}

void SyncServer::enqueue_signal(MessageType signal) {
  _messages_to_send.enqueue(signal);
}

SyncServer::~SyncServer() { _publish_thread->request_stop(); }

void SyncServer::draw_imgui_window() {

  ImGui::Begin("Sync", nullptr);

  if (pc::gui::draw_parameters("sync",
                               struct_parameters.at(std::string{"sync"}))) {
    _config.port = std::clamp(_config.port, 1024, 49151);
  }

  ImGui::End();
}

} // namespace pc::client_sync
