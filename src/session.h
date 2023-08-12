#pragma once

#include "camera/camera_config.h"
#include "mqtt/mqtt_client_config.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pc {

struct PointCasterSession {
  std::string id;
  bool auto_connect_sensors = false;

  // GUI
  std::vector<char> imgui_layout;
  bool fullscreen = false;
  bool hide_ui = false;
  bool show_log = false;
  bool show_devices_window = true;
  bool show_controllers_window = false;
  bool show_radio_window = false;
  bool show_snapshots_window = false;
  bool show_global_transform_window = false;
  bool show_stats = false;
  bool show_mqtt_window = false;

  std::vector<camera::CameraConfiguration> cameras;

  MqttClientConfiguration mqtt_config;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PointCasterSession, id, auto_connect_sensors,
                                   imgui_layout, fullscreen, hide_ui, show_log,
                                   show_devices_window, show_controllers_window,
                                   show_radio_window, show_snapshots_window,
                                   show_global_transform_window, show_stats,
                                   show_mqtt_window, cameras, mqtt_config);

} // namespace pc
