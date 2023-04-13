#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <random>
#include <vector>
#include <set>
#include <queue>
#include <thread>
#include <iostream>
#include <fstream>

#include <zpp_bits.h>

#ifdef _WIN32
#include <WinSock2.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>

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
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <imgui.h>

//#include <libremidi/libremidi.hpp>
#include <pointclouds.h>

#include "camera_controller.h"
#include "gui_helpers.h"
#include "devices/device.h"
#include "devices/usb.h"
#include "wireframe_objects.h"
#include "point_cloud_renderer.h"
#include "sphere_renderer.h"
#include "radio.h"
#include "snapshots.h"

// TODO these need to be removed when initialisation loop is made generic
#include <k4a/k4a.h>
#include "devices/k4a/k4a_device.h"


#if WITH_SKYBRIDGE
#include "skybridge.h"
#endif

namespace bob {

// I guess we probably shouldn't be using namespace like this except for the
// literals... should probs replace thes with using aliases
using namespace bob;
using namespace bob::pointcaster;
using namespace bob::pointcaster::snapshots;
using namespace bob::types;
using namespace bob::sensors;
using namespace Magnum;
using namespace Math::Literals;

using bob::strings::concat;
using bob::sensors::Device;
using bob::sensors::K4ADevice;

using uint = unsigned int;

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

struct PointCasterAppState {
  bool show_sensors_window = true;
  bool show_controllers_window = false;
  bool show_radio_window = true;
  bool show_snapshots_window = true;
  bool show_global_transform_window = true;
  bool show_stats = true;

  bool auto_connect_sensors = false;
};

class PointCaster : public Platform::Application {
public:
  explicit PointCaster(const Arguments &args);

  template<class Function, class... Args>
  void run_async(Function&& f, Args&&... args) {
    _async_tasks.emplace_back(f, args...);
  }

protected:
  PointCasterAppState _state;
  std::vector<std::jthread> _async_tasks;

  std::unique_ptr<Scene3D> _scene;
  std::unique_ptr<SceneGraph::DrawableGroup3D> _drawable_group;

  std::unique_ptr<CameraController> _camera_controller;

  std::unique_ptr<PointCloudRenderer> _point_cloud_renderer;
  std::unique_ptr<SphereRenderer> sphere_renderer;

  std::unique_ptr<WireframeGrid> ground_grid;

  std::unique_ptr<Radio> radio;
  std::unique_ptr<Snapshots> snapshots_context;
  ImGuiIntegration::Context imgui_context{NoCreate};

  std::unique_ptr<UsbMonitor> usb_monitor;

  bool mouse_pressed = false;

  void drawMenuBar();
  void quit();

  void drawSensorsWindow();
  //void initControllers();
  void drawControllersWindow();
  //void handleMidiLearn(const libremidi::message &message);


  Timeline timeline;
  std::vector<float> frame_durations;
  void drawStats();

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
  bob::log.info("This is pointcaster");

#if WITH_SKYBRIDGE
  /* skybridge::initConnection(); */
#endif

  // Set up the window
  const Vector2 dpi_scaling = this->dpiScaling({});
  Configuration conf;
  conf.setTitle("pointcaster");
  // conf.setSize({1600, 1200});
  // conf.setSize({1600, 1080});
  conf.setSize({960, 640});
  conf.setSize(conf.size(), dpi_scaling);
  conf.setWindowFlags(Configuration::WindowFlag::Resizable);
  GLConfiguration gl_conf;
  gl_conf.setSampleCount(dpi_scaling.max() < 2.0f ? 8 : 2);
  if (!tryCreate(conf, gl_conf)) create(conf, gl_conf.setSampleCount(0));

  // Set up ImGui
  ImGui::CreateContext();
  ImGui::StyleColorsDark();

  // load SpaceGrotesk font for imgui
  ImFontConfig font_config;
  font_config.FontDataOwnedByAtlas = false;
  const auto size = Vector2(windowSize()) / dpiScaling();
  Utility::Resource rs("data");
  auto font = rs.getRaw("SpaceGrotesk");
  ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
      const_cast<char *>(font.data()), Int(font.size()),
      14.0f * framebufferSize().x() / size.x(), &font_config);

  imgui_context = ImGuiIntegration::Context(
      *ImGui::GetCurrentContext(), Vector2(windowSize()) / dpiScaling(),
      windowSize(), framebufferSize());

  // Set up blending to be used by imgui
  Magnum::GL::Renderer::setBlendEquation(
      Magnum::GL::Renderer::BlendEquation::Add,
      Magnum::GL::Renderer::BlendEquation::Add);
  Magnum::GL::Renderer::setBlendFunction(
      Magnum::GL::Renderer::BlendFunction::SourceAlpha,
      Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);

  // Set up scene and camera
  _scene = std::make_unique<Scene3D>();
  _drawable_group = std::make_unique<SceneGraph::DrawableGroup3D>();

  const auto viewport_size = GL::defaultFramebuffer.viewport().size();

  _camera_controller = std::make_unique<CameraController>(*_scene);
  _camera_controller->camera()
      .setViewport(GL::defaultFramebuffer.viewport().size());

  ground_grid = std::make_unique<WireframeGrid>(_scene.get(), _drawable_group.get());
  ground_grid->transform(Matrix4::scaling(Vector3(0.25f)) *
			 Matrix4::translation(Vector3(0, 0, 0)));

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
  radio = std::make_unique<Radio>();

  // 
  snapshots_context = std::make_unique<Snapshots>();

  // Init our controllers
  //initControllers();

  // Init our sensors
  // TODO the following should be moved into bob::sensors ns and outside this
  // source file
  // std::lock_guard<std::mutex> lock(bob::sensors::devices_access);
  // bob::sensors::attached_devices.reset(new std::vector<pointer<Device>>);
  // create a callback for the USB handler thread
  // that will add new devices to our main sensor list
  registerUsbAttachCallback([&](auto attached_device) {
    bob::log.debug("attached usb callback");
    if (!_state.auto_connect_sensors) return;

    // freeUsb();
    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    // for (std::size_t i = 0; i < k4a::device::get_installed_count(); i++) {
    //   pointer<bob::sensors::Device> p;
    //   p.reset(new bob::sensors::K4ADevice());
    //   bob::sensors::attached_devices.push_back(std::move(p));
    // }
    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    // initUsb();
    // devices->push_back(attached_device);
  });
  registerUsbDetachCallback([&](auto detached_device) {
    // std::erase(*devices, detached_device);
    bob::log.debug("detached usb callback");
  });
  // init libusb and any attached devices
  usb_monitor = std::make_unique<UsbMonitor>();

  // TODO replace the k4a routine with something generic in usb.cc

  // for (std::size_t i = 0; i < k4a::device::get_installed_count(); i++) {
  //   pointer<bob::sensors::Device> p;
  //   p.reset(new bob::sensors::K4ADevice());
  //   bob::sensors::attached_devices.push_back(std::move(p));
  // }

  timeline.start();
}

void PointCaster::quit() {
  Device::attached_devices.clear(); 
  exit(0);
}

void PointCaster::drawMenuBar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Quit", "q")) quit();
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

      window_item("Transform", "t", _state.show_global_transform_window);
      window_item("Sensors", "s", _state.show_sensors_window);
      window_item("Controllers", "c", _state.show_controllers_window);
      window_item("RenderStats", "f", _state.show_stats);

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

void PointCaster::drawSensorsWindow() {
  ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Sensors", nullptr);

  ImGui::Checkbox("USB auto-connect", &_state.auto_connect_sensors);

  if (ImGui::Button("Open")) {
    run_async([&]() {
      for (std::size_t i = 0; i < k4a::device::get_installed_count(); i++) {
	auto p = std::make_shared<K4ADevice>();
	Device::attached_devices.push_back(std::move(p));
      }
    });
  }

  if (ImGui::Button("Close")) {
    run_async([]() {
      Device::attached_devices.clear();
    });
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

void PointCaster::drawStats() {
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
//    bob::log.info("Detected %d MIDI ports", port_count);
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

void PointCaster::drawControllersWindow() {
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
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
                               GL::FramebufferClear::Depth);
  imgui_context.newFrame();

  auto points = bob::sensors::synthesized_point_cloud();
  points += snapshots::pointCloud();

  if (!points.empty()) {
    _point_cloud_renderer->points = std::move(points);
    _point_cloud_renderer->setDirty();
  }
  _point_cloud_renderer->draw(_camera_controller->camera(), framebufferSize());

  sphere_renderer->draw(_camera_controller->camera());

  _camera_controller->camera().draw(*_drawable_group);

  // Enable text input, if needed/
  if (ImGui::GetIO().WantTextInput && !isTextInputActive())
    startTextInput();
  else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
    stopTextInput();

  // Draw gui windows
  drawMenuBar();
  if (_state.show_sensors_window) drawSensorsWindow();
  if (_state.show_controllers_window) drawControllersWindow();
  if (_state.show_stats) drawStats();
  if (_state.show_radio_window) radio->draw_imgui_window();
  if (_state.show_snapshots_window) snapshots_context->drawImGuiWindow();
  if (_state.show_global_transform_window) bob::sensors::draw_global_controls();

  imgui_context.updateApplicationCursor(*this);

  // Render ImGui window
  GL::Renderer::enable(GL::Renderer::Feature::Blending);
  GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);

  imgui_context.drawFrame();

  GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::disable(GL::Renderer::Feature::Blending);

  // The context is double-buffered, swap buffers
  swapBuffers();

  // Run the next frame immediately
  redraw();
  timeline.nextFrame();
}

void PointCaster::viewportEvent(ViewportEvent &event) {
  // resize main framebuffer
  GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

  // relayout imgui
  imgui_context.relayout(Vector2{event.windowSize()} / event.dpiScaling(),
			  event.windowSize(), event.framebufferSize());

  // recompute the camera's projection matrix
  // camera->setViewport(event.framebufferSize());
}

void PointCaster::keyPressEvent(KeyEvent &event) {
  switch (event.key()) {
  case KeyEvent::Key::Q:
    quit();
    break;
  case KeyEvent::Key::S:
    _state.show_sensors_window = !_state.show_sensors_window;
    break;
  case KeyEvent::Key::C:
    _state.show_controllers_window = !_state.show_controllers_window;
    if (!_state.show_controllers_window) gui::midi_learn_mode = false;
    break;
  case KeyEvent::Key::F:
    _state.show_stats = !_state.show_stats;
    break;
  case KeyEvent::Key::R:
    _state.show_radio_window = !_state.show_radio_window;
    break;
  case KeyEvent::Key::T:
    _state.show_global_transform_window = !_state.show_global_transform_window;
    break;
  default:
    if (imgui_context.handleKeyPressEvent(event))
      event.setAccepted(true);
  }
}

void PointCaster::keyReleaseEvent(KeyEvent &event) {
  if (imgui_context.handleKeyReleaseEvent(event))
    event.setAccepted(true);
}

void PointCaster::textInputEvent(TextInputEvent &event) {
  if (imgui_context.handleTextInputEvent(event))
    event.setAccepted(true);
}

void PointCaster::mousePressEvent(MouseEvent &event) {
  if (imgui_context.handleMousePressEvent(event)) {
    event.setAccepted(true);
    return;
  }
  mouse_pressed = true;
}

void PointCaster::mouseReleaseEvent(MouseEvent &event) {
  mouse_pressed = false;
  if (imgui_context.handleMouseReleaseEvent(event)) {
    event.setAccepted(true);
  }
}

void PointCaster::mouseMoveEvent(MouseMoveEvent &event) {
  if (imgui_context.handleMouseMoveEvent(event)) {
    event.setAccepted(true);
    return;
  }

  // rotate
  if (event.buttons() == MouseMoveEvent::Button::Left)
    _camera_controller->rotate(event.relativePosition());
  // translate
  else if (event.buttons() == MouseMoveEvent::Button::Right)
    _camera_controller->move(event.relativePosition());

  event.setAccepted();
}

void PointCaster::mouseScrollEvent(MouseScrollEvent &event) {
  const Float delta = event.offset().y();
  if (Math::abs(delta) < 1.0e-2f)
    return;

  _camera_controller->zoom(delta);

  if (imgui_context.handleMouseScrollEvent(event)) {
    /* Prevent scrolling the page */
    event.setAccepted();
    return;
  }
}


} // namespace bob

MAGNUM_APPLICATION_MAIN(bob::PointCaster);
