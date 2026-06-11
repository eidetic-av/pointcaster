#include "device_tree.h"

#include <config/transform_config.h>
#include <core/util/geometry_utils.h>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <workspace/workspace_config.h>

namespace pc::devices {

namespace {

bool effective_group_boolean(const pc::WorkspaceConfiguration &config,
                             const std::string &node_id, auto get_config_bool) {
  bool own = true;
  std::string parent;
  bool found = false;

  if (const int gi = group_index_by_id(config, node_id); gi >= 0) {
    own = get_config_bool(config.device_groups[size_t(gi)]);
    parent = config.device_groups[size_t(gi)].parent_id.value();
    found = true;
  } else {
    for (const auto &device_variant : config.devices) {
      std::visit(
          [&](const auto &device_config) {
            if (device_config.id == node_id) {
              own = get_config_bool(device_config);
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
    if (!get_config_bool(config.device_groups[size_t(gi)])) return false;
    p = config.device_groups[size_t(gi)].parent_id.value();
  }
  return true;
}

} // namespace

bool effective_render(const pc::WorkspaceConfiguration &config,
                      const std::string &node_id) {
  return effective_group_boolean(
      config, node_id, [](const auto &c) { return c.render.value(); });
}

bool effective_active(const pc::WorkspaceConfiguration &config,
                      const std::string &node_id) {
  return effective_group_boolean(
      config, node_id, [](const auto &c) { return c.active.value(); });
}

pc::float4x4 effective_world_transform(const pc::WorkspaceConfiguration &config,
                                       const std::string &node_id) {
  std::string parent;
  bool found = false;

  if (const int group_index = group_index_by_id(config, node_id);
      group_index >= 0) {
    parent = config.device_groups[size_t(group_index)].parent_id.value();
    found = true;
  } else {
    for (const auto &device_variant : config.devices) {
      std::visit(
          [&](const auto &device_config) {
            if (device_config.id == node_id) {
              parent = device_config.parent_id.value();
              found = true;
            }
          },
          device_variant);
      if (found) break;
    }
  }
  if (!found) return {};

  pc::float4x4 world; // identity
  for (std::string p = parent; !p.empty();) {
    const int group_index = group_index_by_id(config, p);
    if (group_index < 0) break;
    world = pc::multiply(
        pc::to_float4x4(
            config.device_groups[size_t(group_index)].transform.value()),
        world);
    p = config.device_groups[size_t(group_index)].parent_id.value();
  }
  return world;
}

int group_index_by_id(const pc::WorkspaceConfiguration &config,
                      const std::string &id) {
  for (int i = 0; i < int(config.device_groups.size()); ++i)
    if (config.device_groups[size_t(i)].id == id) return i;
  return -1;
}

} // namespace pc::devices