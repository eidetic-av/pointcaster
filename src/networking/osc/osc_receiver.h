#pragma once

#include "osc_receiver_config.h"
#include <memory>

namespace lo {
class ServerThread;
}

namespace pc {
class Workspace;
}

namespace pc::networking::osc {

class OscReceiver {
public:
  explicit OscReceiver(Workspace& workspace);
  ~OscReceiver();

  // TODO atm called when config changes
  // but config changes so much causes this to reconfigure a lot
  void reconfigure();

private:
  Workspace& _workspace;
  std::unique_ptr<lo::ServerThread> _server;

  OscReceiverConfiguration& config();

  void start();
  void stop();
  void dispatch(const char* address, const char* types, void** argv);
};

} // namespace pc::networking::osc
