#pragma once

#include "camera/camera_config.gen.h"
#include "devices/device_config.gen.h"
#include "devices/usb_config.gen.h"
#include "midi/midi_client_config.gen.h"
#include "osc/osc_client_config.gen.h"
#include "osc/osc_server_config.gen.h"
#include "mqtt/mqtt_client_config.gen.h"
#include "operators/operator_host_config.gen.h"
#include "radio/radio_config.gen.h"
#include "serialization.h"
#include <atomic>
#include <map>
#include <string>
#include <vector>

namespace pc {

struct PointCasterSessionLayout {
  bool fullscreen = false; // @optional
  bool hide_ui = false; // @optional
  bool show_log = true; // @optional
  bool show_devices_window = false; // @optional
  bool show_radio_window = false; // @optional
  bool show_snapshots_window = false; // @optional
  bool show_global_transform_window = false; // @optional
  bool show_stats = false; // @optional
};

using DeviceMap = std::map<std::string, devices::DeviceConfiguration>;
using CameraMap = std::map<std::string, camera::CameraConfiguration>;

struct PointCasterSession {
  std::string id;

  DeviceMap devices; // @optional
  CameraMap cameras; // @optional

  std::optional<radio::RadioConfiguration> radio;
  std::optional<devices::UsbConfiguration> usb;
  std::optional<mqtt::MqttClientConfiguration> mqtt;
  std::optional<midi::MidiClientConfiguration> midi;
  std::optional<osc::OscClientConfiguration> osc_client;
  std::optional<osc::OscServerConfiguration> osc_server;

  PointCasterSessionLayout layout; // @optional

  operators::OperatorHostConfiguration session_operator_host; // @optional
};

} // namespace pc
