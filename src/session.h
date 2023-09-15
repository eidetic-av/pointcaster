#pragma once

#include "camera/camera_config.h"
#include "devices/device_config.h"
#include "mqtt/mqtt_client_config.h"
#include "midi/midi_client_config.h"

#include <serdepp/serde.hpp>

#include <string>
#include <vector>
#include <map>

namespace pc {

struct PointCasterSessionLayout {
  bool fullscreen = false;
  bool hide_ui = false;
  bool show_log = true;
  bool show_devices_window = true;
  bool show_radio_window = false;
  bool show_snapshots_window = false;
  bool show_global_transform_window = false;
  bool show_stats = false;

  DERIVE_SERDE(PointCasterSessionLayout,
               (&Self::fullscreen, "fullscreen")
               (&Self::hide_ui, "hide_ui")
               (&Self::show_log, "show_log")
               (&Self::show_devices_window, "show_devices_window")
               (&Self::show_radio_window, "show_radio_window")
               (&Self::show_snapshots_window, "show_snapshots_window")
               (&Self::show_global_transform_window, "show_global_transform_window")
               (&Self::show_stats, "show_stats"))
};

struct PointCasterSession {
  std::string id;
  bool load_devices_at_launch = true;
  MqttClientConfiguration mqtt;
  midi::MidiClientConfiguration midi;
  std::map<std::string, devices::DeviceConfiguration> devices;
  std::map<std::string, camera::CameraConfiguration> cameras;
  PointCasterSessionLayout layout;

  DERIVE_SERDE(
      PointCasterSession, (&Self::id, "id")
      (&Self::mqtt, "mqtt")
      (&Self::midi, "midi")
      [attributes(make_optional)] (&Self::devices, "devices")
      [attributes(make_optional)] (&Self::cameras, "cameras")
      (&Self::load_devices_at_launch, "load_devices_at_launch")
      (&Self::layout, "layout"))
};

} // namespace pc
