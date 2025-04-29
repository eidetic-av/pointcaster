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

namespace pc {

using DeviceMap = std::map<std::string, devices::DeviceConfigurationVariant>;
using PublishedParameterList = std::vector<std::string>;

using OperatorNameMap = std::map<std::string, operators::uid>;
using OperatorColorMap = std::map<std::string, SerializedColor>;


struct PointcasterWorkspace {
  std::vector<PointcasterSession> sessions;
  int selected_session_index;

  DeviceMap devices; // @optional

  std::optional<devices::UsbConfiguration> usb;
  std::optional<radio::RadioConfiguration> radio;
  std::optional<mqtt::MqttClientConfiguration> mqtt;
  std::optional<midi::MidiDeviceConfiguration> midi;
  std::optional<osc::OscClientConfiguration> osc_client;
  std::optional<osc::OscServerConfiguration> osc_server;
  std::optional<client_sync::SyncServerConfiguration> sync_server;

  std::optional<PublishedParameterList> published_params;
  OperatorNameMap operator_names;   // @optional
  OperatorColorMap operator_colors; // @optional

  gui::PointcasterSessionLayout layout; // @optional
};

} // namespace pc