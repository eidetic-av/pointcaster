#pragma once

#include "client_sync/sync_server_config.gen.h"
#include "devices/device_variants.h"
#include "devices/usb_config.gen.h"
#include "gui/layout.gen.h"
#include "midi/midi_device_config.gen.h"
#include "mqtt/mqtt_client_config.gen.h"
#include "osc/osc_client_config.gen.h"
#include "osc/osc_server_config.gen.h"
#include "radio/radio_config.gen.h"
#include "session.gen.h"
#include "serialization.h"
#include <serdepp/adaptor/toml11.hpp>

namespace pc {

using PublishedParameterList = std::vector<std::string>;
using PushedParameterList = std::vector<std::string>;
using OperatorNameMap = std::map<std::string, operators::uid>;
using OperatorColorMap = std::map<std::string, SerializedColor>;

struct DeviceEntry {
  std::string variant_type;
  devices::DeviceConfigurationVariant variant;
};

struct PointcasterWorkspace {
  std::vector<PointcasterSession> sessions;
  int selected_session_index;

  std::optional<devices::UsbConfiguration> usb;
  std::optional<radio::RadioConfiguration> radio;
  std::optional<mqtt::MqttClientConfiguration> mqtt;
  std::optional<midi::MidiDeviceConfiguration> midi;
  std::optional<osc::OscClientConfiguration> osc_client;
  std::optional<osc::OscServerConfiguration> osc_server;
  std::optional<client_sync::SyncServerConfiguration> sync_server;

  std::optional<PublishedParameterList> published_params;
  PushedParameterList pushed_params; // @optional
  OperatorNameMap operator_names;   // @optional
  OperatorColorMap operator_colors; // @optional

  gui::PointcasterSessionLayout layout; // @optional

  bool maintain_window_focus = false; // @optional
  std::optional<int> maintain_window_focus_ms;
};

} // namespace pc