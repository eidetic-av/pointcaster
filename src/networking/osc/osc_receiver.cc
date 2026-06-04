#include "osc_receiver.h"

#include "config/config_registry.h"
#include "workspace/workspace.h"
#include "workspace/workspace_config.h"

#include <core/logger/logger.h>
#include <lo/lo.h>
#include <lo/lo_cpp.h>

#include <algorithm>
#include <cstring>

namespace pc::networking::osc {

OscReceiver::OscReceiver(Workspace& workspace) : _workspace(workspace) {
  if (config().enable.value()) start();
}

OscReceiver::~OscReceiver() {
  stop();
}

OscReceiverConfiguration& OscReceiver::config() {
  return _workspace.config.osc_receiver.value();
}

void OscReceiver::reconfigure() {
  const bool running = (_server != nullptr);
  const bool should_run = config().enable.value();

  if (running && !should_run) {
    stop();
  } else if (!running && should_run) {
    start();
  }
}

void OscReceiver::start() {
  const int port = std::clamp(config().port.value(), 1024, 49151);
  _server = std::make_unique<lo::ServerThread>(port);

  _server->add_method(nullptr, nullptr,
    [this](const char* path, const char* types, lo_arg** argv, int /*argc*/) {
      dispatch(path, types, reinterpret_cast<void**>(argv));
    });

  _server->start();
  pc::logger()->info("OSC Receiver listening on port {}", port);
}

void OscReceiver::stop() {
  if (!_server) return;
  _server->stop();
  _server.reset();
  pc::logger()->info("OSC Receiver stopped");
}

void OscReceiver::dispatch(const char* address, const char* types, void** argv) {
  if (!address || !types || address[0] != '/') return;

  pc::logger()->trace("received OSC at: {}", address);

  // strip leading '/'
  const std::string_view path{address + 1};
  auto* args = reinterpret_cast<lo_arg**>(argv);

  ConfigValue value;
  bool matched = true;

  if (std::strcmp(types, "f") == 0) {
    value = args[0]->f;
  } else if (std::strcmp(types, "i") == 0) {
    value = args[0]->i;
  } else if (std::strcmp(types, "d") == 0) {
    value = static_cast<double>(args[0]->d);
  } else if (std::strcmp(types, "s") == 0) {
    value = std::string{&args[0]->s};
  } else if (std::strcmp(types, "T") == 0) {
    value = true;
  } else if (std::strcmp(types, "F") == 0) {
    value = false;
  } else {
    matched = false;
  }

  if (!matched) {
    pc::logger()->debug("OSC unhandled type '{}' at '{}'", types, address);
    return;
  }

  if (!_workspace.config_registry.set(path, value)) {
    pc::logger()->debug("OSC no field registered for '{}'", path);
  }
}

} // namespace pc::networking::osc