#include "stream_channels.h"

#include <plugins/devices/device_tree.h>
#include <session/session.h>
#include <variant>
#include <workspace/workspace.h>

namespace pc::networking {

std::vector<StreamChannelSource>
collect_stream_channel_sources(Workspace &workspace) {
  std::vector<StreamChannelSource> sources;
  sources.reserve(workspace.config.sessions.size() + workspace.devices.size());

  for (const auto &session_config : workspace.config.sessions) {
    const auto &label = session_config.label.value();
    std::string address = !label.empty() ? label : session_config.id;

    std::shared_ptr<PointCloud> cloud;
    if (auto it = workspace.sessions.find(session_config.id);
        it != workspace.sessions.end() && it->second) {
      cloud = it->second->point_cloud();
    }
    sources.push_back({std::move(address), std::move(cloud)});
  }

  for (auto &device : workspace.devices) {
    if (!device) continue;
    std::string id;
    std::visit([&](const auto &cfg) { id = cfg.id; }, device->config());
    sources.push_back(
        {devices::device_address(workspace.config, id), device->point_cloud()});
  }

  return sources;
}

} // namespace pc::networking
