#pragma once

#include "../client_sync/sync_server.h"
#include "../operators/session_operator_host.h"
#include "radio_config.gen.h"
#include <memory>
#include <thread>

namespace pc::radio {

class Radio {
public:
  Radio(RadioConfiguration &config,
        pc::operators::SessionOperatorHost &session_operator_host, pc::client_sync::SyncServer& sync_server);

  void draw_imgui_window();

private:
  RadioConfiguration &_config;
  pc::operators::SessionOperatorHost &_session_operator_host;
  pc::client_sync::SyncServer& _sync_server;
  std::unique_ptr<std::jthread> _radio_thread;
};
} // namespace pc::radio
