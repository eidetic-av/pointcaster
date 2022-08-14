#include <chrono>
#include <limits>
#include <mutex>
#include <random>
#include <vector>
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
#include <spdlog/spdlog.h>
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>

#include "pointer.h"

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

#include "gui_helpers.h"
#include "devices/device.h"
// #include "devices/usb.h"
#include "wireframe_objects.h"
#include "point_cloud_renderer.h"
#include "radio.h"

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
using namespace bob::types;
using namespace bob::sensors;
using namespace Magnum;
using namespace Math::Literals;

using uint = unsigned int;

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

class PointCaster : public Platform::Application {
public:
  explicit PointCaster(const Arguments &args);

protected:
  pointer<Radio> radio;
  ImGuiIntegration::Context imgui_context{NoCreate};

  pointer<Scene3D> scene;
  pointer<SceneGraph::DrawableGroup3D> drawable_group;

  // camera helpers
  Vector3 default_cam_position{0.0f, 1.5f, 8.0f};
  Vector3 default_cam_target{0.0f, 1.0f, 0.0f};
  Vector2i prev_mouse_position;
  Vector3 rotation_point, translation_point;
  Float last_depth;
  pointer<Object3D> object_camera;
  pointer<SceneGraph::Camera3D> camera;

  pointer<PointCloudRenderer> point_cloud_renderer;

  // Ground grid
  pointer<WireframeGrid> grid;

  // helper functions for camera movement
  Float depthAt(const Vector2i &window_position);
  Vector3 unproject(const Vector2i &window_position, Float depth) const;
  bool mouse_pressed = false;

  void drawMenuBar();
  void quit();

  bool show_sensors_window = true;
  void drawSensorsWindow();

  bool show_controllers_window = false;
  //void initControllers();
  void drawControllersWindow();
  //void handleMidiLearn(const libremidi::message &message);

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
  spdlog::info("This is Box of Birds PointCaster");

#if WITH_SKYBRIDGE
  /* skybridge::initConnection(); */
#endif

  // Set up the window
  const Vector2 dpi_scaling = this->dpiScaling({});
  Configuration conf;
  conf.setTitle("pointcaster");
  //conf.setSize({1600, 1200});
  //conf.setSize({1600, 960});
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
  scene.reset(new Scene3D{});
  drawable_group.reset(new SceneGraph::DrawableGroup3D{});

  object_camera.reset(new Object3D({scene.get()}));
  object_camera->setTransformation(
      Matrix4::lookAt(Vector3(0, 1.5, 8), Vector3(0, 1, 0), Vector3(0, 1, 0)));

  const auto viewport_size = GL::defaultFramebuffer.viewport().size();
  camera.reset(new SceneGraph::Camera3D(*object_camera));
  camera->setProjectionMatrix(Matrix4::perspectiveProjection(
	  45.0_degf, Vector2{viewport_size}.aspectRatio(), 0.01f, 1000.0f))
      .setViewport(viewport_size);

  // set default camera parameters
  default_cam_position = Vector3(1.5f, 3.3f, 6.0f);
  default_cam_target = Vector3(1.5f, 1.3f, 0.0f);
  object_camera->setTransformation(Matrix4::lookAt(
      default_cam_position, default_cam_target, Vector3(0, 1, 0)));

  // initialise depth to the value at scene center
  last_depth = ((camera->projectionMatrix() * camera->cameraMatrix())
		     .transformPoint({}).z() + 1.0f) * 0.5f;

  // Set up ground grid
  grid.reset(new WireframeGrid(scene.get(), drawable_group.get()));
  grid->transform(Matrix4::scaling(Vector3(0.5f)) *
		   Matrix4::translation(Vector3(3, 0, -5)));

  // Enable depth test, render particles as sprites
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::ProgramPointSize);

  // set background color
  GL::Renderer::setClearColor(0x0d1117_rgbf);

  // TODO renderer class should be RAII
  // initialise our main pc renderer
  point_cloud_renderer.reset(new PointCloudRenderer(0.005f));
  point_cloud_renderer->points = PointCloud{};
  point_cloud_renderer->setDirty();


  // Start the timer, loop at 144 Hz max
  setSwapInterval(1);
  setMinimalLoopPeriod(7);

  // Initialise our network radio for points
  // const int broadcast_port = 9999;
  // radio.reset(new Radio(broadcast_port));

  // Init our controllers
  //initControllers();

  // Init our sensors
  // TODO the following should be moved into bob::sensors ns and outside this
  // source file
  // std::lock_guard<std::mutex> lock(bob::sensors::devices_access);
  // bob::sensors::attached_devices.reset(new std::vector<pointer<Device>>);
  // // create a callback for the USB handler thread
  // // that will add new devices to our main sensor list
  // registerUsbAttachCallback([&](Device* attached_device) {
  //   devices->push_back(attached_device);
  // });
  // registerUsbDetachCallback([&](Device* detached_device) {
  //   std::erase(*devices, detached_device);
  // });
  // init libusb and any attached devices
  // initUsb();

  // TODO replace the k4a routine with something generic in usb.cc
  for (std::size_t i = 0; i < k4a::device::get_installed_count(); i++) {
    pointer<bob::sensors::Device> p;
    p.reset(new bob::sensors::K4ADevice());
    bob::sensors::attached_devices.push_back(std::move(p));
  }

  spdlog::debug("{} devices attached", bob::sensors::attached_devices.size());
}

void PointCaster::quit() {
  // freeUsb();
  exit(0);
}

void PointCaster::drawMenuBar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Quit", "q")) quit();
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Window")) {
      ImGui::BeginDisabled();
      ImGui::Checkbox("##Window_Sensors", &show_sensors_window);
      ImGui::EndDisabled();
      ImGui::SameLine();
      if (ImGui::MenuItem("Sensors", "s"))
	show_sensors_window = !show_sensors_window;

      ImGui::BeginDisabled();
      ImGui::Checkbox("##Window_Controllers", &show_controllers_window);
      ImGui::EndDisabled();
      ImGui::SameLine();
      if (ImGui::MenuItem("Controllers", "c")) {
	show_controllers_window = !show_controllers_window;
	if (!show_controllers_window) gui::midi_learn_mode = false;
      }

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

void PointCaster::drawSensorsWindow() {
  ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Sensors", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);
  for (auto& device : bob::sensors::attached_devices) {
    if (ImGui::CollapsingHeader(device->name.c_str(), nullptr))
      device->Device::drawImGuiControls();
    ImGui::Separator();
  }
  ImGui::PopItemWidth();
  ImGui::End();
}
//
//void PointCaster::initControllers() {
//  using namespace libremidi;
//  std::thread midi_startup([&]() {
//    midi_in midi;
//    auto port_count = midi.get_port_count();
//    spdlog::info("Detected {} MIDI ports", port_count);
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

  // Enable text input, if needed/
  if (ImGui::GetIO().WantTextInput && !isTextInputActive())
    startTextInput();
  else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
    stopTextInput();

  drawMenuBar();

  auto points = bob::sensors::synthesizedPointCloud();
  if (!points.empty()) {
    point_cloud_renderer->points = std::move(points);
    point_cloud_renderer->setDirty();
  }
  point_cloud_renderer->draw(camera, framebufferSize());

  camera->draw(*drawable_group);

  // Draw gui windows
  if (show_sensors_window) drawSensorsWindow();
  if (show_controllers_window) drawControllersWindow();

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
}

Float PointCaster::depthAt(const Vector2i &window_position) {
  /* First scale the position from being relative to window size to being
     relative to framebuffer size as those two can be different on HiDPI
     systems */
  const Vector2i position =
      window_position * Vector2(framebufferSize()) / Vector2(windowSize());
  const Vector2i fbPosition(position.x(),
			    GL::defaultFramebuffer.viewport().sizeY() -
				position.y() - 1);

  GL::defaultFramebuffer.mapForRead(
      GL::DefaultFramebuffer::ReadAttachment::Front);
  Image2D data = GL::defaultFramebuffer.read(
      Range2Di::fromSize(fbPosition, Vector2i(1)).padded(Vector2i(2)),
      {GL::PixelFormat::DepthComponent, GL::PixelType::Float});

  return Math::min<Float>(Containers::arrayCast<const Float>(data.data()));
}

Vector3 PointCaster::unproject(const Vector2i &window_position,
			       float depth) const {
  /* We have to take window size, not framebuffer size, since the position is
     in window coordinates and the two can be different on HiDPI systems */
  const Vector2i viewSize = windowSize();
  const Vector2i viewPosition =
      Vector2i(window_position.x(), viewSize.y() - window_position.y() - 1);
  const Vector3 in(2.0f * Vector2(viewPosition) / Vector2(viewSize) -
		       Vector2(1.0f),
		   depth * 2.0f - 1.0f);

  return camera->projectionMatrix().inverted().transformPoint(in);
}

void PointCaster::viewportEvent(ViewportEvent &event) {
  // resize main framebuffer
  GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

  // relayout imgui
  imgui_context.relayout(Vector2{event.windowSize()} / event.dpiScaling(),
			  event.windowSize(), event.framebufferSize());

  // recompute the camera's projection matrix
  camera->setViewport(event.framebufferSize());
}

void PointCaster::keyPressEvent(KeyEvent &event) {
  switch (event.key()) {
  case KeyEvent::Key::Q:
    quit();
    break;
  case KeyEvent::Key::S:
    show_sensors_window = !show_sensors_window;
    break;
  case KeyEvent::Key::C:
    show_controllers_window = !show_controllers_window;
    if (!show_controllers_window) gui::midi_learn_mode = false;
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

  // Update camera
  prev_mouse_position = event.position();
  const Float currentDepth = depthAt(event.position());
  const Float depth = currentDepth == 1.0f ? last_depth : currentDepth;
  translation_point = unproject(event.position(), depth);

  /* Update the rotation point only if we're not zooming against infinite
     depth or if the original rotation point is not yet initialized */
  if (currentDepth != 1.0f || rotation_point.isZero()) {
    rotation_point = translation_point;
    last_depth = depth;
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

  const Vector2 delta = 3.0f *
			Vector2(event.position() - prev_mouse_position) /
			Vector2(framebufferSize());
  prev_mouse_position = event.position();

  if (!event.buttons())
    return;

  /* Translate */
  if (event.modifiers() & MouseMoveEvent::Modifier::Shift) {
    const Vector3 p = unproject(event.position(), last_depth);
    object_camera->translateLocal(translation_point - p); /* is Z always 0? */
    translation_point = p;

    /* Rotate around rotation point */
  } else {
    object_camera->transformLocal(Matrix4::translation(rotation_point) *
				   Matrix4::rotationX(-0.51_radf * delta.y()) *
				   Matrix4::rotationY(-0.51_radf * delta.x()) *
				   Matrix4::translation(-rotation_point));
  }

  event.setAccepted();
}

void PointCaster::mouseScrollEvent(MouseScrollEvent &event) {
  const Float delta = event.offset().y();
  if (Math::abs(delta) < 1.0e-2f)
    return;

  if (imgui_context.handleMouseScrollEvent(event)) {
    /* Prevent scrolling the page */
    event.setAccepted();
    return;
  }

  const Float current_depth = depthAt(event.position());
  const Float depth = current_depth == 1.0f ? last_depth : current_depth;
  const Vector3 p = unproject(event.position(), depth);
  /* Update the rotation point only if we're not zooming against infinite
     depth or if the original rotation point is not yet initialized */
  if (current_depth != 1.0f || rotation_point.isZero()) {
    rotation_point = p;
    last_depth = depth;
  }

  /* Move towards/backwards the rotation point in cam coords */
  object_camera->translateLocal(rotation_point * delta * 0.1f);
}


} // namespace bob

MAGNUM_APPLICATION_MAIN(bob::PointCaster);
