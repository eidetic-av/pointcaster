#include "message_streamer.h"
#include <core/logger/logger.h>

// TODO can i do this?? it seems circular? it probs isnt because message
// streamer is forward declared inside workspace...
#include <mutex>
#include <workspace/workspace.h>

namespace pc::publishers {

MessageStreamer::MessageStreamer(Workspace &workspace)
    : _listener(*this, workspace) {
  pc::logger()->debug("MessageStreamer constructor");
}

void MessageStreamer::handle_update(
    const std::string_view path, const ConfigValue &value,
    const MessageStreamerConfiguration &config_snapshot) const {

  pc::logger()->debug("aw yeah messagestreamer");
}

MessageStreamerConfiguration
MessageStreamer::config(Workspace &workspace) const {
  std::scoped_lock lock(workspace.config_access);
  return workspace.config.publishers.value().message_streamer.value();
}

} // namespace pc::publishers