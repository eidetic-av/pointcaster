#pragma once

#include <config/transform_config.h>
#include <pointcaster_api.h>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <workspace/workspace_config.h>

namespace pc::devices {

// whether a node is switched on for a given session
POINTCASTER_API bool
effective_session_enabled(const pc::WorkspaceConfiguration &config,
                          const pc::SessionConfiguration &session_config,
                          const std::string &node_id);

POINTCASTER_API bool in_any_session(const pc::WorkspaceConfiguration &config,
                                    const std::string &node_id);

POINTCASTER_API std::vector<std::string>
all_node_ids(const pc::WorkspaceConfiguration &config);

// this takes a device or group id and will give the accumulated transform for
// its ancestors... that is what to offset its own local transform by...
POINTCASTER_API pc::float4x4
effective_world_transform(const pc::WorkspaceConfiguration &config,
                          const std::string &node_id);

POINTCASTER_API int group_index_by_id(const pc::WorkspaceConfiguration &config,
                                      const std::string &id);

// builds a "<ancestor_label_or_id>/.../<device_label_or_id>" address for a
// device or group, walking up the parent_id chain through device_groups
POINTCASTER_API std::string
device_address(const pc::WorkspaceConfiguration &config,
               const std::string &device_id);

// returns the IDs of all non-group devices whose parent_id chain passes
// through the given group_id (includes devices in nested child groups)
POINTCASTER_API std::vector<std::string>
device_ids_in_group(const pc::WorkspaceConfiguration &config,
                    const std::string &group_id);

} // namespace pc::devices