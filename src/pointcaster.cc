#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <span>
#include <thread>
#include <vector>

#include <zpp_bits.h>

#ifdef _WIN32
#include <WinSock2.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include <zmq.hpp>

#include "path.h"
#include "log.h"
#include "pointer.h"
#include "string_utils.h"

#include <Corrade/Utility/StlMath.h>
#include <Magnum/Image.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Version.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/ImGuiIntegration/Widgets.h>
#include "fonts/IconsFontAwesome6.h"

//#include <libremidi/libremidi.hpp>
#include <pointclouds.h>

#include "gui_helpers.h"
#include "camera/camera_controller.h"
#include "devices/device.h"
#include "devices/usb.h"
#include "wireframe_objects.h"
#include "point_cloud_renderer.h"
#include "sphere_renderer.h"
#include "radio.h"
#include "snapshots.h"
#include "uuid.h"

// TODO these need to be removed when initialisation loop is made generic
#include <k4a/k4a.h>
#include "devices/k4a/k4a_device.h"


#if WITH_SKYBRIDGE
#include "skybridge.h"
#endif

namespace pc {

using namespace pc;
using namespace pc::types;
using namespace pc::sensors;
using namespace pc::camera;
using namespace pc::radio;
using namespace pc::snapshots;
using namespace Magnum;
using namespace Math::Literals;

using pc::strings::concat;
using pc::sensors::Device;
using pc::sensors::K4ADevice;

using uint = unsigned int;

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

struct PointCasterSession {
  std::string id;

  // GUI
  std::vector<char> imgui_layout;
  bool show_sensors_window = true;
  bool show_controllers_window = false;
  bool show_radio_window = true;
  bool show_snapshots_window = false;
  bool show_global_transform_window = true;
  bool show_stats = true;
  bool auto_connect_sensors = false;

  std::vector<CameraConfiguration> cameras;

};

class PointCaster : public Platform::Application {
public:
  explicit PointCaster(const Arguments &args);

  template<class Function, class... Args>
  void run_async(Function&& f, Args&&... args) {
    _async_tasks.emplace_back(f, args...);
  }

protected:
  PointCasterSession _session;

  void serialize_session();
  void serialize_session(std::filesystem::path file_path);
  void deserialize_session(std::filesystem::path file_path);
  
  std::vector<std::jthread> _async_tasks;

  std::unique_ptr<Scene3D> _scene;
  std::unique_ptr<SceneGraph::DrawableGroup3D> _drawable_group;

  std::vector<std::unique_ptr<CameraController>> _cameras;
  int _hovering_camera_index = -1;

  std::unique_ptr<PointCloudRenderer> _point_cloud_renderer;
  std::unique_ptr<SphereRenderer> _sphere_renderer;

  std::unique_ptr<WireframeGrid> _ground_grid;

  std::unique_ptr<Radio> _radio;
  std::unique_ptr<Snapshots> _snapshots_context;

  std::unique_ptr<UsbMonitor> _usb_monitor;

  ImGuiIntegration::Context _imgui_context{NoCreate};

  void open_kinect_sensors();

  void fill_point_renderer();

  void draw_menu_bar();
  void draw_control_bar();
  void draw_camera_window();
  void draw_sensors_window();
  void draw_controllers_window();

  void saveAndQuit();

  //void initControllers();
  //void handleMidiLearn(const libremidi::message &message);

  Timeline timeline;
  std::vector<float> frame_durations;
  void draw_stats();

  void drawEvent() override;
  void viewportEvent(ViewportEvent &event) override;
  void keyPressEvent(KeyEvent &event) override;
  void keyReleaseEvent(KeyEvent &event) override;
  void textInputEvent(TextInputEvent &event) override;
  void mousePressEvent(MouseEvent &event) override;
  void mouseReleaseEvent(MouseEvent &event) override;
  void mouseMoveEvent(MouseMoveEvent &event) override;
  void mouseScrollEvent(MouseScrollEvent &event) override;
};

PointCaster::PointCaster(const Arguments &args)
    : Platform::Application(args, NoCreate) {
  pc::log.info("This is pointcaster");

  // Set up the window
  const Vector2 dpi_scaling = this->dpiScaling({});
  Configuration conf;
  conf.setTitle("pointcaster");
  conf.setSize({1900, 1200});
  // conf.setSize({1600, 1080});
  // conf.setSize({960, 640});
  conf.setSize(conf.size(), dpi_scaling);
  conf.setWindowFlags(Configuration::WindowFlag::Resizable);

  // Try 8x MSAA, fall back to zero if not possible.
  // Enable only 2x MSAA if we have enough DPI.
  GLConfiguration gl_conf;
  gl_conf.setSampleCount(dpi_scaling.max() < 2.0f ? 8 : 2);
  if (!tryCreate(conf, gl_conf)) create(conf, gl_conf.setSampleCount(0));

  // Set up ImGui
  ImGui::CreateContext();
  // Don't save imgui layout to a file, handle it manually
  ImGui::GetIO().IniFilename = nullptr;

  ImGui::StyleColorsDark();


#if WITH_SKYBRIDGE
  /* skybridge::initConnection(); */
#endif

  const auto size = Vector2(windowSize()) / dpiScaling();

  // load fonts from resources
  Utility::Resource rs("data");

  auto font = rs.getRaw("SpaceGrotesk");
  ImFontConfig font_config;
  font_config.FontDataOwnedByAtlas = false;
  const auto font_size = 14.0f;
  ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
      const_cast<char *>(font.data()), font.size(),
      14.0f * framebufferSize().x() / size.x(), &font_config);

  auto font_icons = rs.getRaw("FontAwesomeRegular");
  static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_16_FA, 0 };
  ImFontConfig icons_config;
  icons_config.MergeMode = true;
  icons_config.PixelSnapH = true;
  icons_config.FontDataOwnedByAtlas = false;
  const auto icon_font_size = font_size * 2.0f / 3.0f;
  icons_config.GlyphMinAdvanceX = icon_font_size;
  const auto icon_font_size_pixels = font_size * framebufferSize().x() / size.x();

  ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
      const_cast<char *>(font_icons.data()), font_icons.size(),
      icon_font_size_pixels, &icons_config, icons_ranges);

  // enable window docking
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // ImGui::GetIO().ConfigFlags |= ImGuiDockNodeFlags_PassthruCentralNode;
  // ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // for editing parameters with the keyboard
  auto backspace = ImGui::GetIO().KeyMap[ImGuiKey_Backspace];
  ImGui::GetIO().AddInputCharacter(backspace);
  ImGui::GetIO().AddKeyEvent(ImGuiKey_Backspace, true);
  ImGui::GetIO().AddKeyEvent(ImGuiKey_Backspace, false);

  _imgui_context = ImGuiIntegration::Context(
      *ImGui::GetCurrentContext(), Vector2(windowSize()) / dpiScaling(),
      windowSize(), framebufferSize());

  // Set up blending to be used by imgui
  Magnum::GL::Renderer::setBlendEquation(
      Magnum::GL::Renderer::BlendEquation::Add,
      Magnum::GL::Renderer::BlendEquation::Add);
  Magnum::GL::Renderer::setBlendFunction(
      Magnum::GL::Renderer::BlendFunction::SourceAlpha,
      Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);

  // Set up scene
  // TODO should drawable groups go inside each camera controller?
  _scene = std::make_unique<Scene3D>();
  _drawable_group = std::make_unique<SceneGraph::DrawableGroup3D>();
  _ground_grid = std::make_unique<WireframeGrid>(_scene.get(), _drawable_group.get());
  _ground_grid->transform(Matrix4::scaling(Vector3(1.0f)) *
			 Matrix4::translation(Vector3(0, 0, 0)));

  // Deserialize last session
  auto data_dir = path::get_or_create_data_directory();
  std::filesystem::path last_modified_session_file;
  std::filesystem::file_time_type last_write_time;

  for (const auto &entry : std::filesystem::directory_iterator(data_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pcs")
      continue;

    auto write_time = std::filesystem::last_write_time(entry);
    if (last_modified_session_file.empty() || write_time > last_write_time) {
      last_modified_session_file = entry.path();
      last_write_time = write_time;
    }
  }

  if (last_modified_session_file.empty()) {
    pc::log.info("No previous session file found. Creating new session.");
    _session = {.id = pc::uuid::word()};
    auto file_path = data_dir / (_session.id + ".pcs");
    serialize_session(file_path);
  } else {
    pc::log.info("Found previous session file");
    deserialize_session(last_modified_session_file);
  }

  // If there are no cameras in the scene, initialise at least one
  if (_cameras.empty()) {
    auto _default_camera_controller = std::make_unique<CameraController>(_scene.get());
    // TODO viewport size needs to be dynamic
    _default_camera_controller->camera().setViewport(
	GL::defaultFramebuffer.viewport().size());
    _cameras.push_back(std::move(_default_camera_controller));
  }

  const auto viewport_size = GL::defaultFramebuffer.viewport().size();

  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  // render particles as sprites
  GL::Renderer::enable(GL::Renderer::Feature::ProgramPointSize);

  // set background color
  GL::Renderer::setClearColor(0x0d1117_rgbf);

  _point_cloud_renderer = std::make_unique<PointCloudRenderer>(0.005f);
  _sphere_renderer = std::make_unique<SphereRenderer>();

  // Start the timer, loop at 144 Hz max
  setSwapInterval(1);
  setMinimalLoopPeriod(7);

  // Initialise our network radio for points
  _radio = std::make_unique<Radio>();

  // 
  _snapshots_context = std::make_unique<Snapshots>();

  if (_session.auto_connect_sensors) open_kinect_sensors();

  // Init our controllers
  //initControllers();

  // Init our sensors
  // TODO the following should be moved into pc::sensors ns and outside this
  // source file
  // std::lock_guard<std::mutex> lock(pc::sensors::devices_access);
  // pc::sensors::attached_devices.reset(new std::vector<pointer<Device>>);
  // create a callback for the USB handler thread
  // that will add new devices to our main sensor list
  registerUsbAttachCallback([&](auto attached_device) {
    pc::log.debug("attached usb callback");
    if (!_session.auto_connect_sensors) return;

    // freeUsb();
    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    // for (std::size_t i = 0; i < k4a::device::get_installed_count(); i++) {
    //   pointer<pc::sensors::Device> p;
    //   p.reset(new pc::sensors::K4ADevice());
    //   pc::sensors::attached_devices.push_back(std::move(p));
    // }
    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    // initUsb();
    // devices->push_back(attached_device);
  });
  registerUsbDetachCallback([&](auto detached_device) {
    // std::erase(*devices, detached_device);
    pc::log.debug("detached usb callback");
  });
  // init libusb and any attached devices
  _usb_monitor = std::make_unique<UsbMonitor>();

  // TODO replace the k4a routine with something generic in usb.cc
  // open each k4a on startup:
  // for (std::size_t i = 0; i < k4a::device::get_installed_count(); i++) {
  //   pointer<pc::sensors::Device> p;
  //   p.reset(new pc::sensors::K4ADevice());
  //   pc::sensors::attached_devices.push_back(std::move(p));
  // }

  timeline.start();
}

void PointCaster::saveAndQuit() {
  serialize_session();
  exit(0);
}

void PointCaster::serialize_session() {
  auto data_dir = path::get_or_create_data_directory();
  auto file_path = data_dir / (_session.id + ".pcs");
  serialize_session(file_path);
}

void PointCaster::serialize_session(std::filesystem::path file_path) {
  pc::log.info("Saving session to %s", file_path.string());

  // save imgui layout
  std::size_t imgui_layout_size;
  auto imgui_layout_data = ImGui::SaveIniSettingsToMemory(&imgui_layout_size);
  _session.imgui_layout = std::vector<char>(
      imgui_layout_data, imgui_layout_data + imgui_layout_size);

  // save camera configurations
  _session.cameras.clear();
  for (auto &camera : _cameras) {
    _session.cameras.push_back(camera->config());
  }

  std::vector<uint8_t> data;
  auto out = zpp::bits::out(data);
  auto success = out(_session);
  path::save_file(file_path, data);
}

void PointCaster::deserialize_session(std::filesystem::path file_path) {
  pc::log.info("Loading state from %s", file_path.string());
  auto buffer = path::load_file(file_path);
  auto in = zpp::bits::in(buffer);
  auto success = in(_session);
  pc::log.info("%s", std::to_string(_session.imgui_layout.size()));

  ImGui::LoadIniSettingsFromMemory(_session.imgui_layout.data(),
				   _session.imgui_layout.size());

  // get saved camera configurations to populate the cams list
  for (auto &saved_camera_config : _session.cameras) {
    auto saved_camera =
	std::make_unique<CameraController>(_scene.get(), saved_camera_config);
    saved_camera->camera().setViewport(
        GL::defaultFramebuffer.viewport().size());
    _cameras.push_back(std::move(saved_camera));
  }

  pc::log.info("Loaded session '%s'", file_path.filename().string());
}

void PointCaster::fill_point_renderer() {
  auto points = pc::sensors::synthesized_point_cloud();
  points += snapshots::pointCloud();

  if (!points.empty()) {
    _point_cloud_renderer->points = std::move(points);
    _point_cloud_renderer->setDirty();
  }
}

void PointCaster::draw_menu_bar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Quit", "q")) saveAndQuit();
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Window")) {

      constexpr auto window_item = [](const char * item_name,
				      const char * shortcut_key,
				      bool & window_toggle) {
	ImGui::BeginDisabled();
	ImGui::Checkbox(concat("##Toggle_Window_", item_name).data(), &window_toggle);
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::MenuItem(item_name, shortcut_key)) window_toggle = !window_toggle;
      };

      window_item("Transform", "t", _session.show_global_transform_window);
      window_item("Sensors", "s", _session.show_sensors_window);
      window_item("Controllers", "c", _session.show_controllers_window);
      window_item("RenderStats", "f", _session.show_stats);

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

void PointCaster::draw_control_bar() {

  constexpr auto control_bar_flags =
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_MenuBar;

  if (ImGui::BeginViewportSideBar("##ControlBar", ImGui::GetMainViewport(),
				  ImGuiDir_Up, ImGui::GetFrameHeight(),
				  control_bar_flags)) {
    if (ImGui::BeginMenuBar()) {
      if (ImGui::Button(ICON_FA_FLOPPY_DISK)) {
	serialize_session();
      }

      if (ImGui::Button(ICON_FA_FOLDER_OPEN)) {
      }
      ImGui::EndMenuBar();
    }

    ImGui::End();
  }

  // ImGui::SetNextWindowPos({0, 20});

  // // Extend width to viewport width
  // ImGui::SetNextWindowSize({100, 100});

  // constexpr ImGuiWindowFlags control_bar_flags =
  //     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
  //     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoScrollbar |
  //     ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoDecoration;

  // if (ImGui::Begin("ControlBar", nullptr, control_bar_flags)) {
  //   ImGui::Text("eyo");
  //   ImGui::SameLine();
  //   ImGui::Button("oh");
  //   ImGui::SameLine();
  //   ImGui::Button("woh");
  //   ImGui::End();
  // }
}

void PointCaster::draw_camera_window() {

  _hovering_camera_index = -1;

  ImGuiWindowClass docking_viewport_class = {};

  ImGuiID id =
      ImGui::DockSpaceOverViewport(nullptr,
                                   ImGuiDockNodeFlags_NoDockingInCentralNode |
                                       ImGuiDockNodeFlags_PassthruCentralNode,
                                   nullptr);
  ImGuiDockNode* node = ImGui::DockBuilderGetCentralNode(id);

  ImGuiWindowClass central_always = {};
  central_always.DockNodeFlagsOverrideSet |=
      ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_NoDockingOverMe;
  ImGui::SetNextWindowClass(&central_always);
  ImGui::SetNextWindowDockID(node->ID, ImGuiCond_Always);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});

  if (ImGui::Begin("CamerasRoot")) {
    ImGui::PopStyleVar();

    constexpr auto camera_tab_bar_flags =
        ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_NoTooltip;

    if (ImGui::BeginTabBar("Cameras"), camera_tab_bar_flags) {

      // button for creating a new camera
      auto new_camera_index = -1;
      if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing)) {
        auto new_camera = std::make_unique<CameraController>(_scene.get());
        _cameras.push_back(std::move(new_camera));
        new_camera_index = _cameras.size() - 1;
        pc::log.info("new camera: %i", new_camera_index);
      }

      for (int i = 0; i < _cameras.size(); i++) {
        const auto &camera = _cameras.at(i);

        ImGuiTabItemFlags tab_item_flags = ImGuiTabItemFlags_None;
        if (new_camera_index == i)
          tab_item_flags |= ImGuiTabItemFlags_SetSelected;

        if (ImGui::BeginTabItem(camera->name().c_str(), nullptr,
                                tab_item_flags)) {
          ImGui::BeginChild("Frame");

          const auto window_size = ImGui::GetWindowSize();
          const auto frame_size =
              Vector2i{(int)window_size.x, (int)window_size.y};

          camera->setupFramebuffer(frame_size);
          camera->bindFramebuffer();

          _point_cloud_renderer->draw(camera->camera(), frame_size);
          _sphere_renderer->draw(camera->camera());

          camera->camera().draw(*_drawable_group);

          ImGuiIntegration::image(
              camera->outputFrame(),
              {(float)frame_size.x(), (float)frame_size.y()});

          if (ImGui::IsItemHovered())
            _hovering_camera_index = i;

          ImGui::EndChild();
          ImGui::EndTabItem();
        }
      }
      ImGui::EndTabBar();
    }

    ImGui::End();
    GL::defaultFramebuffer.bind();
  }
}

void PointCaster::open_kinect_sensors() {
    run_async([&]() {
      for (std::size_t i = 0; i < k4a::device::get_installed_count(); i++) {
	auto p = std::make_shared<K4ADevice>();
	Device::attached_devices.push_back(std::move(p));
      }
    });
}

void PointCaster::draw_sensors_window() {
  ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Sensors", nullptr);

  ImGui::Checkbox("USB auto-connect", &_session.auto_connect_sensors);

  if (ImGui::Button("Open")) open_kinect_sensors();

  if (ImGui::Button("Close")) {
    run_async([]() { Device::attached_devices.clear(); });
  }

  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);

  for (auto& device : Device::attached_devices) {
    if (!device->is_sensor) return;
    if (ImGui::CollapsingHeader(device->name.c_str(), nullptr))
      device->Device::draw_imgui_controls();
    ImGui::Spacing();
    ImGui::Text(device->name.c_str());
    device->Device::draw_imgui_controls();
    ImGui::Spacing();
    ImGui::Separator();
  }
  ImGui::PopItemWidth();
  ImGui::End();
}

void PointCaster::draw_stats() {
  ImGui::PushID("FrameStats");
  ImGui::SetNextWindowPos({50.0f, 200.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({200.0f, 100.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Frame Stats", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);

  // calculate the mean, min and max frame times from our last 60 frames
  const auto frame_duration = timeline.previousFrameDuration();
  frame_durations.push_back(frame_duration);
  constexpr auto frames_to_average = 60 * 2; // 2 seconds
  if (frame_durations.size() < frames_to_average) {
    ImGui::Text("Gathering data...");
  } else {
    frame_durations.erase(frame_durations.begin()); // pop_front
    const float avg_duration =
      std::reduce(frame_durations.begin(), frame_durations.end()) / frame_durations.size();
    const auto minmax_duration =
      std::minmax_element(frame_durations.begin(), frame_durations.end());

    if (ImGui::CollapsingHeader("Rendering", true)) {
      ImGui::Text("Frame Duration");
      ImGui::BeginTable("duration", 2);
      ImGui::TableNextColumn();
      ImGui::Text("Average");
      ImGui::TableNextColumn();
      ImGui::Text("%.2fms", avg_duration * 1000);
      ImGui::TableNextColumn();
      ImGui::Text("Min");
      ImGui::TableNextColumn();
      ImGui::Text("%.2fms", *minmax_duration.first * 1000);
      ImGui::TableNextColumn();
      ImGui::Text("Max");
      ImGui::TableNextColumn();
      ImGui::Text("%.2fms", *minmax_duration.second * 1000);
      ImGui::EndTable();
      ImGui::Spacing();
      ImGui::Text("%.0f FPS", 1000.0f / (avg_duration * 1000));
    }
  }

  ImGui::PopItemWidth();
  ImGui::End();
  ImGui::PopID();
}
//
//void PointCaster::initControllers() {
//  using namespace libremidi;
//  std::thread midi_startup([&]() {
//    midi_in midi;
//    auto port_count = midi.get_port_count();
//    pc::log.info("Detected %d MIDI ports", port_count);
//    midi.open_port(0);
//    midi.set_callback([&](const message &message) {
//      if (gui::midi_learn_mode) {
//	handleMidiLearn(message);
//	return;
//      }
//      // parse the midi message
//      auto channel = message.get_channel();
//      auto type = message.get_message_type();
//      float value;
//      uint controller_number;
//      if (type == message_type::PITCH_BEND) {
//	controller_number = 999;
//	int first_byte = message[1];
//	int second_byte = message[2];
//	int pb_value = (second_byte * 128) + first_byte;
//	value = pb_value / 128.f / 128.f;
//      } else if (type == message_type::CONTROL_CHANGE) {
//	controller_number = message[1];
//	int control_change_value = message[2];
//	value = control_change_value / 127.f;
//      }
//
//      // check if we have assigned the midi message to any parameters
//      gui::AssignedMidiParameter learned_parameter;
//      for (auto assigned_parameter : gui::assigned_midi_parameters) {
//	if (assigned_parameter.channel != channel) continue;
//	if (assigned_parameter.controller_number != controller_number) continue;
//	// copy it locally
//	learned_parameter = assigned_parameter;
//	break;
//      }
//      if (learned_parameter.parameter.value == nullptr) return;
//      // if we have assigned this parameter, apply the change
//      auto min = learned_parameter.parameter.range_min;
//      auto max = learned_parameter.parameter.range_max;
//      auto output = min + value * (max - min);
//      // TODO this is only implemented for float parameters
//      if (learned_parameter.parameter.param_type == gui::ParameterType::Float) {
//	auto value_ptr = reinterpret_cast<float*>(learned_parameter.parameter.value);
//	*value_ptr = output;
//      }
//    });
//  });
// midi_startup.detach();
//}

void PointCaster::draw_controllers_window() {
  ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Controllers", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);
  if (gui::midi_learn_mode) ImGui::BeginDisabled();
  if (ImGui::Button("Midi Learn")) {
    gui::midi_learn_mode = true;
    ImGui::BeginDisabled();
  }
  if (gui::midi_learn_mode) {
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) gui::midi_learn_mode = false;
  }
  // for (auto device : *devices) {
  //   if (ImGui::CollapsingHeader(device->name.c_str(), nullptr))
  //     device->Device::drawImGuiControls();
  // }
  ImGui::PopItemWidth();
  ImGui::End();
}

//void PointCaster::handleMidiLearn(const libremidi::message &message) {
//  using namespace libremidi;
//
//  if (!gui::midi_learn_parameter) return;
//
//  auto channel = message.get_channel();
//  auto type = message.get_message_type();
//
//  if (type == message_type::CONTROL_CHANGE) {
//    gui::assigned_midi_parameters.push_back(gui::AssignedMidiParameter {
//	*gui::midi_learn_parameter, channel, message[1]});
//    gui::midi_learn_parameter.reset();
//    gui::midi_learn_mode = false;
//    return;
//  }
//
//  if (type == message_type::PITCH_BEND) {
//    // use controller number 999 for pitch bends
//    gui::assigned_midi_parameters.push_back(gui::AssignedMidiParameter {
//	*gui::midi_learn_parameter, channel, 999});
//    gui::midi_learn_parameter.reset();
//    gui::midi_learn_mode = false;
//  }
//
//}

auto output_count = 0;

void PointCaster::drawEvent() {

  fill_point_renderer();
  
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
                               GL::FramebufferClear::Depth);
  _imgui_context.newFrame();

  // Enable text input, if needed/
  if (ImGui::GetIO().WantTextInput && !isTextInputActive())
    startTextInput();
  else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
    stopTextInput();

  // Draw gui windows

  // ImGui::DockSpaceOverViewport();

  draw_menu_bar();
  draw_control_bar();

  draw_camera_window();

  if (_session.show_sensors_window) draw_sensors_window();
  if (_session.show_controllers_window) draw_controllers_window();
  if (_session.show_stats) draw_stats();
  if (_session.show_radio_window) _radio->draw_imgui_window();
  if (_session.show_snapshots_window) _snapshots_context->draw_imgui_window();
  if (_session.show_global_transform_window) pc::sensors::draw_global_controls();

  _imgui_context.updateApplicationCursor(*this);

  // Render ImGui window
  GL::Renderer::enable(GL::Renderer::Feature::Blending);
  GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);

  _imgui_context.drawFrame();

  if (ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }

  GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::disable(GL::Renderer::Feature::Blending);

  // TODO the timeline should not be linked to GUI drawing
  // so this needs to be run elsewhere, but still needs to
  // be locked to the same FPS
  timeline.nextFrame();

  // The context is double-buffered, swap buffers
  swapBuffers();

  // Run the next frame immediately
  redraw();
}

void PointCaster::viewportEvent(ViewportEvent &event) {
  // resize main framebuffer
  GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

  // relayout imgui
  _imgui_context.relayout(Vector2{event.windowSize()} / event.dpiScaling(),
			  event.windowSize(), event.framebufferSize());

  // recompute the camera's projection matrix
  // camera->setViewport(event.framebufferSize());
}

void PointCaster::keyPressEvent(KeyEvent &event) {
  switch (event.key()) {
  case KeyEvent::Key::Q:
    saveAndQuit();
    break;
  case KeyEvent::Key::S:
    _session.show_sensors_window = !_session.show_sensors_window;
    break;
  case KeyEvent::Key::C:
    _session.show_controllers_window = !_session.show_controllers_window;
    if (!_session.show_controllers_window) gui::midi_learn_mode = false;
    break;
  case KeyEvent::Key::F:
    _session.show_stats = !_session.show_stats;
    break;
  case KeyEvent::Key::R:
    _session.show_radio_window = !_session.show_radio_window;
    break;
  case KeyEvent::Key::T:
    _session.show_global_transform_window = !_session.show_global_transform_window;
    break;
  default:
    if (_imgui_context.handleKeyPressEvent(event))
      event.setAccepted(true);
  }
}

void PointCaster::keyReleaseEvent(KeyEvent &event) {
  if (_imgui_context.handleKeyReleaseEvent(event))
    event.setAccepted(true);
}

void PointCaster::textInputEvent(TextInputEvent &event) {
  if (_imgui_context.handleTextInputEvent(event))
    event.setAccepted(true);
}

void PointCaster::mousePressEvent(MouseEvent &event) {
  if (_imgui_context.handleMousePressEvent(event)) {
    event.setAccepted(true);
    return;
  }
}

void PointCaster::mouseReleaseEvent(MouseEvent &event) {
  if (_imgui_context.handleMouseReleaseEvent(event)) {
    event.setAccepted(true);
  }
}

// TODO
// all camera related mouse events need to happen on top of the selected camera
// window

void PointCaster::mouseMoveEvent(MouseMoveEvent &event) {
  if (_imgui_context.handleMouseMoveEvent(event) && _hovering_camera_index == -1) {
    event.setAccepted(true);
    return;
  }

  // rotate
  if (event.buttons() == MouseMoveEvent::Button::Left)
    _cameras.at(_hovering_camera_index)->mouseRotate(event);
  // translate
  else if (event.buttons() == MouseMoveEvent::Button::Right)
    _cameras.at(_hovering_camera_index)->mouseTranslate(event);

  event.setAccepted();
}

void PointCaster::mouseScrollEvent(MouseScrollEvent &event) {
  if (_imgui_context.handleMouseScrollEvent(event) && _hovering_camera_index == -1) {
    /* Prevent scrolling the page */
    event.setAccepted(true);
    return;
  }

  const Float delta = event.offset().y();
  if (Math::abs(delta) < 1.0e-2f) return;

  if (event.modifiers() ==
      Magnum::Platform::Sdl2Application::InputEvent::Modifier::Alt) {
    _cameras.at(_hovering_camera_index)->zoomPerspective(event);

  } else {
    _cameras.at(_hovering_camera_index)->dolly(event);
  }
}


} // namespace pc

MAGNUM_APPLICATION_MAIN(pc::PointCaster);
