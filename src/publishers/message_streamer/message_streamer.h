#pragma once

#include "../workspace_listener.h"
#include "message_streamer_config.h"

namespace pc::publishers {
class MessageStreamer {
public:
  MessageStreamer(Workspace &workspace);

  MessageStreamerConfiguration config(Workspace &workspace) const;

  void handle_update(const std::string_view path, const ConfigValue &value,
                     const MessageStreamerConfiguration &config_snapshot) const;

private:
  // must stay last
  WorkspaceListener<MessageStreamer, MessageStreamerConfiguration> _listener;
};
} // namespace pc::publishers
