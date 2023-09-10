#pragma once

#include "midi_client_config.h"
#include <libremidi/client.hpp>
#include <libremidi/libremidi.hpp>

#include <atomic>
#include <expected>
#include <future>
#include <map>
#include <memory>
#include <mqtt/async_client.h>
#include <msgpack.hpp>
#include <mutex>
#include <string_view>

namespace pc::midi {

class MidiClient {
public:
  static std::shared_ptr<MidiClient> &instance() {
    if (!_instance)
      throw std::logic_error(
          "No MidiClient instance created. Call MidiClient::create() first.");
    return _instance;
  }

  static void create(MidiClientConfiguration &config) {
    std::lock_guard lock(_mutex);
    _instance = make_instance(config);
  }

  MidiClient(const MidiClient &) = delete;
  MidiClient &operator=(const MidiClient &) = delete;

  ~MidiClient();

  void draw_imgui_window();

private:
  static std::shared_ptr<MidiClient> _instance;
  static std::mutex _mutex;

  static std::shared_ptr<MidiClient>
  make_instance(MidiClientConfiguration &config) {
    return std::shared_ptr<MidiClient>(new MidiClient(config));
  }

  explicit MidiClient(MidiClientConfiguration &config);

  MidiClientConfiguration &_config;
  std::unordered_map<std::string, libremidi::observer> _midi_observers;
  std::unordered_map<std::string, libremidi::midi_in> _midi_inputs;

  void handle_input_added(const libremidi::input_port &port,
                          const libremidi::API &api);
  void handle_input_removed(const libremidi::input_port &port);

  void handle_message(const std::string &port_name,
                      const libremidi::message &msg);
};

} // namespace pc::midi
