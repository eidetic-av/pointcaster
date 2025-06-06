#include "camera_controller.h"
#include "../parameters.h"
#include "../uuid.h"
#include "../gui/widgets.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/ArrayView.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/BufferImage.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/ImageFormat.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Trade/ImageData.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace pc::camera {

using namespace Magnum;
using Magnum::Platform::Sdl2Application;
using Magnum::Image2D;
using Magnum::Matrix4;
using Magnum::Quaternion;
using Magnum::Vector3;
using Magnum::Math::Deg;

using namespace pc::parameters;

std::atomic<std::size_t> CameraController::count = 0;

CameraController::CameraController(Magnum::Platform::Application *app,
                                   Scene3D *scene,
                                   std::string_view host_session_id,
                                   CameraConfiguration config)
    : session_id(host_session_id), _app(app), _config(config),
      _frame_analyser(this) {

  _anchor = std::make_unique<Object3D>(scene);
  _orbit_parent_left_right = std::make_unique<Object3D>(_anchor.get());
  _orbit_parent_up_down = std::make_unique<Object3D>(_orbit_parent_left_right.get());
  _camera_parent = std::make_unique<Object3D>(_orbit_parent_up_down.get());

  _camera = std::make_unique<Camera3D>(*_camera_parent);
  _camera->setViewport(GL::defaultFramebuffer.viewport().size());

  if (_config.id.empty()) _config.id = pc::uuid::word();

  // apply any loaded configuration
  set_distance(_config.transform.distance);
  set_orbit(_config.transform.orbit);
  set_roll(_config.transform.roll);
  set_translation(_config.transform.translation);

  reset_projection_matrix();

  auto &resolution = _config.rendering.resolution;
  if (resolution[0] == 0 || resolution[1] == 0)
    resolution = defaults::rendering_resolution;

  pc::logger->info("{}x{}", resolution[0], resolution[1]);

  setup_frame({resolution[0], resolution[1]});

  declare_parameters(session_id, _config.id, _config);

  pc::logger->info("Initialised new camera controller ({})", _config.id);
}

CameraController::~CameraController() { CameraController::count--; }

void CameraController::setup_frame(Vector2i frame_size) {

  Vector2i scaled_size{
      static_cast<int>(frame_size.x() / _app->dpiScaling().x()),
      static_cast<int>(frame_size.y() / _app->dpiScaling().y())};

  auto aspect_ratio = frame_size.x() / static_cast<float>(frame_size.y());
  if (_config.rendering.scale_mode == (int)ScaleMode::Span &&
      viewport_size.has_value()) {
    // automatically set frame height based on size of viewport
    // TODO this is only working when aspect ratio > 1.0
    aspect_ratio = viewport_size->x() / viewport_size->y();
    scaled_size.y() = scaled_size.x() / aspect_ratio;
    _config.rendering.resolution[1] =
        _config.rendering.resolution[0] / aspect_ratio;
  }

  // update any transforms that have changed
  set_distance(_config.transform.distance);
  set_orbit(_config.transform.orbit);
  set_roll(_config.transform.roll);
  set_translation(_config.transform.translation);

  // if no change in frame size, bind the framebuffer and finish here
  if (scaled_size == _frame_size) {
    bind_framebuffer();
    return;
  }

  // otherwise recreate containers with new frame size

  _frame_size = scaled_size;

  _frame_analyser.set_frame_size(_frame_size);

  reset_projection_matrix();

  std::lock_guard lock_color(_color_frame_mutex);

  _color = std::make_unique<GL::Texture2D>();
  _color->setStorage(1, GL::TextureFormat::RGBA8, _frame_size);

  _depth_stencil = std::make_unique<GL::Renderbuffer>();
  _depth_stencil->setStorage(GL::RenderbufferFormat::Depth24Stencil8,
                             _frame_size);

  _framebuffer = std::make_unique<GL::Framebuffer>(Range2Di{{}, _frame_size});

  _framebuffer->attachTexture(GL::Framebuffer::ColorAttachment{0}, *_color, 0);

  _framebuffer->attachRenderbuffer(
      GL::Framebuffer::BufferAttachment::DepthStencil, *_depth_stencil);

  bind_framebuffer();
}

void CameraController::bind_framebuffer() {
  _framebuffer->clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth)
      .bind();
}

void CameraController::reset_projection_matrix() {
  auto resolution = _config.rendering.resolution;
  auto aspect_ratio = static_cast<float>(resolution[0]) / resolution[1];
  auto clipping = _config.rendering.clipping;

  if (_config.rendering.orthographic) {
    auto &ortho_size = _config.rendering.orthographic_size;
    _camera->setProjectionMatrix(Matrix4::orthographicProjection(
	{ortho_size.x * aspect_ratio, ortho_size.y}, clipping.min,
	clipping.max));
  } else {
    _camera->setProjectionMatrix(Matrix4::perspectiveProjection(
	Deg_f(_config.fov), aspect_ratio, clipping.min, clipping.max));
  }
}

GL::Texture2D &CameraController::color_frame() {
  std::unique_lock lock(_color_frame_mutex);
  return *_color;
}

GL::Texture2D &CameraController::analysis_frame() {
  return _frame_analyser.analysis_frame();
}

void CameraController::dispatch_analysis() {
  std::unique_lock lock(_color_frame_mutex);
  _frame_analyser.dispatch_analysis(*_color.get(), _config.analysis);
}

int CameraController::analysis_time() {
  return _frame_analyser.analysis_time();
}

void CameraController::set_distance(const float metres) {
  auto camera_transform_matrix = _camera_parent->transformationMatrix();
  camera_transform_matrix.translation().z() = metres;
  _camera_parent->setTransformation(camera_transform_matrix);
  _config.transform.distance = metres;
}

void CameraController::add_distance(const float metres) {
  auto camera_transform_matrix = _camera_parent->transformationMatrix();
  camera_transform_matrix.translation().z() += metres;
  _camera_parent->setTransformation(camera_transform_matrix);
  _config.transform.distance = camera_transform_matrix.translation().z();
}

void CameraController::set_orbit(const Float2 degrees) {
  _orbit_parent_left_right->setTransformation(
      Matrix4::rotationY(Deg{degrees.x}));
  _orbit_parent_up_down->setTransformation(Matrix4::rotationX(Deg{-degrees.y}));
  _config.transform.orbit = {degrees.x, degrees.y};
}

void CameraController::add_orbit(const Float2 degrees) {
  _orbit_parent_left_right->setTransformation(
      _orbit_parent_left_right->transformationMatrix() *
      Matrix4::rotationY(Deg{degrees.x}));

  _orbit_parent_up_down->setTransformation(
      _orbit_parent_up_down->transformationMatrix() *
      Matrix4::rotationX(Deg{-degrees.y}));

  _config.transform.orbit.x += degrees.x;
  _config.transform.orbit.y += degrees.y;
}

void CameraController::set_roll(const float degrees) {
  auto current_translation =
      _camera_parent->transformationMatrix().translation();
  auto roll = Matrix4::rotationZ(Deg{degrees}).rotation();
  _camera_parent->setTransformation(Matrix4::from(roll, current_translation));
  _config.transform.roll = degrees;
}

void CameraController::add_roll(const float degrees) {
  auto current_transform = _camera_parent->transformationMatrix();
  auto added_roll = Matrix4::rotationZ(Deg{degrees});
  auto new_transform = current_transform * added_roll;
  _camera_parent->setTransformation(new_transform);
  _config.transform.roll += degrees;
}

void CameraController::set_translation(const Float3 metres) {
  _anchor->setTransformation(
      Matrix4::translation({metres.x, metres.y, metres.z}));
  _config.transform.translation = metres;
}

void CameraController::add_translation(const Float3 metres) {
  auto transform = _anchor->transformationMatrix();
  transform.translation() += {metres.x, metres.y, metres.z};
  _anchor->setTransformation(transform);

  _config.transform.translation.x += metres.x;
  _config.transform.translation.y += metres.y;
  _config.transform.translation.z += metres.z;
}

void CameraController::dolly(Sdl2Application::MouseScrollEvent &event) {
  const auto delta =
      -event.offset().y() / static_cast<float>(_config.scroll_precision);
  if (!_config.rendering.orthographic) add_distance(delta / 10);
  else {
    auto &ortho_size = _config.rendering.orthographic_size;
    ortho_size = {ortho_size.x + delta, ortho_size.y + delta};
    if (ortho_size.x < 0.05f) ortho_size.x = 0.05f;
    if (ortho_size.y < 0.05f) ortho_size.y = 0.05f;
    reset_projection_matrix();
  }
}

void CameraController::mouse_orbit(Sdl2Application::MouseMoveEvent &event) {
  auto delta = Vector2{event.relativePosition()} * _rotate_speed;
  if (event.modifiers() == Sdl2Application::InputEvent::Modifier::Ctrl) {
    add_roll(-delta.y());
  } else {
    add_orbit({-delta.x(), delta.y()});
  }
}

void CameraController::mouse_translate(Sdl2Application::MouseMoveEvent &event,
                                       bool lock_y_axis) {
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const Vector3 p = unproject(event.position(), centre_depth);

  Vector3 delta = {(float)event.relativePosition().x() * _move_speed.x(),
		   (float)event.relativePosition().y() * _move_speed.y(), 0};

  if (lock_y_axis) {
    delta = _orbit_parent_left_right->transformationMatrix().rotation() *
	    _orbit_parent_up_down->transformationMatrix().rotation() * delta;
    add_translation({delta.x(), 0, delta.z()});
  } else {

    delta = _orbit_parent_left_right->transformationMatrix().rotation() *
	    _orbit_parent_up_down->transformationMatrix().rotation() * delta;
    add_translation({delta.x(), delta.y(), delta.z()});
  }
}

Magnum::Vector3
CameraController::unproject(const Magnum::Vector2i &window_position,
			    Magnum::Float depth) const {
  const Vector2i view_position{window_position.x(),
			       _frame_size.y() - window_position.y() - 1};
  const Vector3 in{2 * Vector2{view_position} / Vector2{_frame_size} -
		       Vector2{1.0f},
		   depth * 2.0f - 1.0f};
  return _camera->projectionMatrix().inverted().transformPoint(in);
}

Magnum::Float CameraController::depth_at(const Vector2i &window_position) {
  const Vector2i position = window_position * Vector2{_app->framebufferSize()} /
                            Vector2{_app->windowSize()};
  const Vector2i fbPosition{position.x(),
                            GL::defaultFramebuffer.viewport().sizeY() -
                                position.y() - 1};

  GL::defaultFramebuffer.mapForRead(
      GL::DefaultFramebuffer::ReadAttachment::Front);
  auto data = GL::defaultFramebuffer.read(
      Range2Di::fromSize(fbPosition, Vector2i{1}).padded(Vector2i{2}),
      {GL::PixelFormat::DepthComponent, GL::PixelType::Float});

  return Math::min<Magnum::Float>(data.pixels<Magnum::Float>().asContiguous());
}

void CameraController::draw_imgui_controls() {
  ImGui::SetNextWindowPos({200.0f, 200.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({250.0f, 400.0f}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Camera");
  // static std::string parameter_key = std::format("{}/camera", session_id);
  if (pc::gui::draw_parameters(_config.id)) { reset_projection_matrix(); }
  ImGui::End();
}

std::vector<gui::OverlayText> CameraController::labels() {
  return _frame_analyser.analysis_labels();
}

} // namespace pc::camera
