#pragma once

#include <string>
#include <variant>
#include <workspace/workspace_config.h>

namespace pc::devices {

inline int group_index_by_id(const pc::WorkspaceConfiguration &config,
                             const std::string &id) {
  for (int i = 0; i < int(config.device_groups.size()); ++i)
    if (config.device_groups[size_t(i)].id == id) return i;
  return -1;
}

// effective = the node's own flag AND every ancestor group's flag.
// a group's state therefore gates everything beneath it, without
// mutating the children's own flags.
template <typename Flag>
bool effective_group_value(const pc::WorkspaceConfiguration &config,
                           const std::string &node_id, Flag flag) {
  bool own = true;
  std::string parent;
  bool found = false;

  if (const int gi = group_index_by_id(config, node_id); gi >= 0) {
    own = flag(config.device_groups[size_t(gi)]);
    parent = config.device_groups[size_t(gi)].parent_id.value();
    found = true;
  } else {
    for (const auto &device_variant : config.devices) {
      std::visit(
          [&](const auto &device_config) {
            if (device_config.id == node_id) {
              own = flag(device_config);
              parent = device_config.parent_id.value();
              found = true;
            }
          },
          device_variant);
      if (found) break;
    }
  }

  if (!found) return true;
  if (!own) return false;
  for (std::string p = parent; !p.empty();) {
    const int gi = group_index_by_id(config, p);
    if (gi < 0) break;
    if (!flag(config.device_groups[size_t(gi)])) return false;
    p = config.device_groups[size_t(gi)].parent_id.value();
  }
  return true;
}

inline bool effective_render(const pc::WorkspaceConfiguration &config,
                             const std::string &id) {
  return effective_group_value(config, id,
                               [](const auto &c) { return c.render.value(); });
}

inline bool effective_active(const pc::WorkspaceConfiguration &config,
                             const std::string &id) {
  return effective_group_value(config, id,
                               [](const auto &c) { return c.active.value(); });
}

} // namespace pc::devices