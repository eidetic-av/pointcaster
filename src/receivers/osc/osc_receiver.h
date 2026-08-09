#pragma once

#include "osc_receiver_config.h"
#include <memory>

namespace lo {
class ServerThread;
}

namespace pc {
class Workspace;
}

namespace pc::receivers {

class OscReceiver {
public:
  enum class State { Stopped, Running, Failed };

  explicit OscReceiver(Workspace& workspace);
  ~OscReceiver();

  void reconfigure();

  State state() const { return _state; }

private:
  Workspace& _workspace;
  std::unique_ptr<lo::ServerThread> _server;
  State _state = State::Stopped;
  int _active_port = 0;

  OscReceiverConfiguration& config();

  void start(int port);
  void stop();
  void dispatch(const char* address, const char* types, void** argv);
};

} // namespace pc::receivers
