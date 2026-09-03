#pragma once

#include "../workspace_listener.h"
#include "message_streamer_config.h"

#include <util/string_map.h>

namespace zmq {
class socket_t;
}

namespace pc::publishers {
class MessageStreamer {
public:
  MessageStreamer(Workspace &workspace);

  MessageStreamerConfiguration config(Workspace &workspace) const;

  void handle_update(const std::string_view path, const ConfigValue &value,
                     const MessageStreamerConfiguration &);

  void
  handle_config_change(std::string_view path,
                       const MessageStreamerConfiguration &config_snapshot);

  void tick(const MessageStreamerConfiguration &config_snapshot);

  struct Subscribers {
    StringMap<int> count_for_path;
    bool dirty = false;
  } subscribers;

private:
  std::unique_ptr<zmq::socket_t> _socket = nullptr;

  // must stay last, contains a thread
  WorkspaceListener<MessageStreamer, MessageStreamerConfiguration> _listener;
};
} // namespace pc::publishers
