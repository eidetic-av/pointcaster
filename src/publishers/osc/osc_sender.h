#pragma once

#include "../workspace_listener.h"
#include "osc_sender_config.h"
#include <memory>

namespace pc::publishers {

class OscConnection;

class OscSender {
public:
  explicit OscSender(Workspace &workspace);
  ~OscSender();

  OscSender(const OscSender &) = delete;
  OscSender &operator=(const OscSender &) = delete;
  OscSender(OscSender &&) = delete;
  OscSender &operator=(OscSender &&) = delete;

  OscSenderConfiguration config(Workspace &workspace) const;

  void handle_update(const std::string_view path, const ConfigValue &value,
                     const OscSenderConfiguration &config_snapshot);

  void handle_config_change(std::string_view path,
                            const OscSenderConfiguration &config_snapshot);

private:
  std::unique_ptr<OscConnection> _connection;

  // must stay last
  WorkspaceListener<OscSender, OscSenderConfiguration> _listener;
};
} // namespace pc::publishers
