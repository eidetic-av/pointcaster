#pragma once

#include "../camera/camera_config.gen.h"
#include "../devices/device_config.gen.h"
#include "../operators/operator_host_config.gen.h"
#include <string>

namespace pc::graph {

enum class LinkDirection { In, Out, InOut };

using NodeDataVariant =
    std::variant<operators::OperatorConfigurationVariant,
                 std::reference_wrapper<pc::camera::CameraConfiguration>>;

struct Node {
  int id;
  LinkDirection link_direction;
  NodeDataVariant node_data;
  int in_connection{-1};            // single input connection
  std::vector<int> out_connections; // multiple output connections
};

struct Link {
  int id;
  int output_id;
  int input_id;
};

template <typename T> struct is_reference_wrapper : std::false_type {};

template <typename U>
struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

} // namespace pc::graph
