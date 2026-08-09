#pragma once

#include <memory>
#include <pointcaster/point_cloud.h>
#include <string>
#include <vector>

namespace pc {
class Workspace;
}

namespace pc::networking {

struct StreamChannelSource {
  std::string address;
  std::shared_ptr<PointCloud> cloud;
};

std::vector<StreamChannelSource>
collect_stream_channel_sources(Workspace &workspace);

} // namespace pc::networking
