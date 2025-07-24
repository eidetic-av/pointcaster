#pragma once

#include "pch.h"

#include "workspace.gen.h"
#include "devices/sequence/ply_sequence_player_config.gen.h"
#include "gui/catpuccin.h"
#include "parameters.h"
#include <imgui.h>
#include <imgui_stdlib.h>

#ifdef WIN32
#include <SDL2/SDL_Clipboard.h>
#else
#include <SDL2/SDL.h>
#endif

#ifdef WITH_OSC
#include "osc/osc_client.h"
#include "osc/osc_server.h"
#endif

#include "camera/camera_controller.h"
#include "client_sync/sync_server.h"
#include "devices/device.h"
#include "devices/k4a/k4a_device.h"
#include "devices/orbbec/orbbec_device.h"
#include "devices/sequence/ply_sequence_player.h"
#include "devices/sequence/ply_sequence_player_config.gen.h"
#include "devices/usb.h"
#include "fonts/IconsFontAwesome6.h"
#include "graph/operator_graph.h"
#include "gui/widgets.h"
#include "logger.h"
#include "main_thread_dispatcher.h"
#include "midi/midi_device.h"
#include "modes.h"
#include "mqtt/mqtt_client.h"
#include "mqtt/mqtt_client_config.gen.h"
#include "objects/wireframe_objects.h"
#include "operators/session_bounding_boxes.h"
#include "operators/session_operator_host.h"
#include "path.h"
#include "point_cloud_renderer/point_cloud_renderer.h"
#include "pointclouds.h"
#include "pointer.h"
#include "profiling.h"
#include "publisher/publisher.h"
#include "radio/radio.h"
#include "session.gen.h"
#include "shaders/texture_display.h"
#include "snapshots.h"
#include "sphere_renderer.h"
#include "structs.h"
#include "tween/tween_manager.h"
#include "type_utils.h"
#include "uuid.h"

// Temporary headers to be removed
#include <k4a/k4a.h>

namespace pc {

using namespace pc;
using namespace pc::camera;
using namespace pc::client_sync;
using namespace pc::devices;
using namespace pc::graph;
using namespace pc::midi;
using namespace pc::mqtt;
using namespace pc::operators;
using namespace pc::parameters;
using namespace pc::radio;
using namespace pc::snapshots;
using namespace pc::tween;
using namespace pc::types;

#ifdef WITH_OSC
using namespace pc::osc;
#endif

using namespace Magnum;
using namespace Math::Literals;

using pc::devices::Device;
using pc::devices::K4ADevice;
using pc::devices::OrbbecDevice;

using uint = unsigned int;
using Object3D = Magnum::SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = Magnum::SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

struct SphereInstanceData {
  Matrix4 transformationMatrix;
  Matrix3x3 normalMatrix;
  Color3 color;
};

class PointCaster : public Platform::Application {

public:
  explicit PointCaster(const Arguments &args);

  template <class Function, class... Args>
  void run_async(Function &&f, Args &&...args) {
    _async_tasks.emplace_back(f, args...);
  }

  PointcasterWorkspace workspace;
  std::vector<SessionOperatorHost> session_operator_hosts;

  std::filesystem::path last_modified_workspace_file;
  std::filesystem::file_time_type last_write_time;

  // TODO these shouldn't be part of the PointCaster app class,
  // they should exist inside their own device namespace
  void open_kinect_sensors();
  void open_orbbec_sensor(std::string_view ip);
  void open_ply_sequence();

  std::mutex _session_devices_mutex;

  Mode _current_mode{Mode::Normal};
  std::array<char, modeline_buffer_size> _modeline_input =
      std::array<char, modeline_buffer_size>({});

  std::vector<std::jthread> _async_tasks;

  std::unique_ptr<Scene3D> _scene;
  std::unique_ptr<SceneGraph::DrawableGroup3D> _scene_root;

  std::optional<Vector2i> _display_resolution;

  std::vector<std::unique_ptr<CameraController>> camera_controllers;
  std::optional<std::reference_wrapper<CameraController>>
      interacting_camera_controller;

  std::unique_ptr<PointCloudRenderer> _point_cloud_renderer;
  std::unique_ptr<SphereRenderer> _sphere_renderer;

  std::unique_ptr<WireframeGrid> _ground_grid;
  std::unique_ptr<OperatorGraph> _session_operator_graph;

  std::unique_ptr<Snapshots> _snapshots_context;

  std::unique_ptr<Radio> _radio;
  std::unique_ptr<MqttClient> _mqtt;
  std::unique_ptr<MidiDevice> _midi;
#ifdef WITH_OSC
  std::unique_ptr<OscClient> _osc_client;
  std::unique_ptr<OscServer> _osc_server;
#endif
  std::unique_ptr<SyncServer> _sync_server;

#ifndef WIN32
  std::unique_ptr<UsbMonitor> _usb_monitor;
  std::mutex _usb_config_mutex;
#endif

  ImGuiIntegration::Context _imgui_context{NoCreate};

  ImFont *_font;
  ImFont *_mono_font;
  std::shared_ptr<ImFont> _icon_font;
  std::shared_ptr<ImFont> _icon_font_small;

  /* Spheres rendering */
  GL::Mesh _sphere_mesh{NoCreate};
  GL::Buffer _sphere_instance_buffer{NoCreate};
  Shaders::PhongGL _sphere_shader{NoCreate};
  Containers::Array<SphereInstanceData> _sphere_instance_data;

  void save_workspace();
  void save_workspace(std::filesystem::path file_path);
  void load_workspace(std::filesystem::path file_path);

  void sync_session_instances();

  std::atomic_bool loading_device = false;
  void load_device(DeviceConfigurationVariant &config);
  void load_k4a_device(AzureKinectConfiguration &config,
                       std::string_view target_id = "");

  void render_cameras();
  void publish_parameters();

  void draw_menu_bar();
  void draw_control_bar();
  void draw_devices_window();
  void draw_onscreen_log();
  void draw_modeline();

  Vector2i _restore_window_size;
  Vector2i _restore_window_position;
  bool _full_screen;
  void set_full_screen(bool full_screen);

  void quit();
  void save_and_quit();

  Timeline _timeline;
  std::vector<float> frame_durations;
  void draw_stats(const float delta_time);

protected:
  void drawEvent() override;
  void viewportEvent(ViewportEvent &event) override;

  void keyPressEvent(KeyEvent &event) override;
  void keyReleaseEvent(KeyEvent &event) override;

  void find_mode_keypress(KeyEvent &event);

  void textInputEvent(TextInputEvent &event) override;
  void mousePressEvent(MouseEvent &event) override;
  void mouseReleaseEvent(MouseEvent &event) override;
  void mouseMoveEvent(MouseMoveEvent &event) override;
  void mouseScrollEvent(MouseScrollEvent &event) override;

  template <class MemberType, class ConfigType, class... Args>
  void try_init_unique_member(std::optional<ConfigType> &config,
                              std::unique_ptr<MemberType> &member_ptr,
                              Args &&...ctor_args) {
    bool failed = false;
    if (config) {
      try {
        member_ptr = std::make_unique<MemberType>(
            *config, std::forward<Args>(ctor_args)...);
      } catch (...) {
        pc::logger->error(
            "Failed to load '{}' type. Possibly invalid configuration file.",
            ConfigType::Name);
        failed = true;
      }
    }
    if (!config || failed) {
      config = ConfigType{};
      member_ptr = std::make_unique<MemberType>(
          *config, std::forward<Args>(ctor_args)...);
    }
  }
};

} // namespace pc