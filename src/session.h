#pragma once

#include "camera/camera_config.gen.h"
#include "operators/operator_friendly_names.h"
#include "operators/operator_host_config.gen.h"
#include "string_map.h"
#include <format>
#include <map>
#include <string>
#include <vector>


namespace pc {

// these are necessary for the python pre-processor for now, doesn't handle nested types <<<>>> well
using CameraMap = std::map<std::string, camera::CameraConfiguration>;
using SerializedColor = std::vector<float>;
using ActiveDevicesMap = std::map<std::string, bool>;

struct PointcasterSession {
  std::string id;
  std::string label;

  camera::CameraConfiguration camera; // @optional
  ActiveDevicesMap active_devices; // @optional

  operators::OperatorHostConfiguration session_operator_host; // @optional
};

inline std::map<std::string, std::string> session_label_from_id{};
inline std::map<std::string, std::string> session_id_from_parameter_id{};

} // namespace pc
