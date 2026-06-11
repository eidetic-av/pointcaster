#pragma once

#include <config/transform_config.h>
#include <pointcaster_api.h>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <workspace/workspace_config.h>

namespace pc::devices {

POINTCASTER_API bool effective_render(const pc::WorkspaceConfiguration &config,
                                      const std::string &node_id);

POINTCASTER_API bool effective_active(const pc::WorkspaceConfiguration &config,
                                      const std::string &node_id);

// this takes a device or group id and will give the accumulated transform for
// its ancestors... that is what to offset its own local transform by...
POINTCASTER_API pc::float4x4
effective_world_transform(const pc::WorkspaceConfiguration &config,
                          const std::string &node_id);

POINTCASTER_API int group_index_by_id(const pc::WorkspaceConfiguration &config,
                                      const std::string &id);

} // namespace pc::devices