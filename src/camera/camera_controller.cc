#include "camera_controller.h"
#include "../analysis/analyser_2d_config.h"
#include "../gui_helpers.h"
#include "../logger.h"
#include "../math.h"
#include "../uuid.h"
#include "camera_config.h"
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
#include <algorithm>
#include <numbers>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#if WITH_MQTT
#include "../mqtt/mqtt_client.h"
#endif

namespace pc::camera {

using namespace Magnum;
using Magnum::Image2D;
using Magnum::Matrix4;
using Magnum::Quaternion;
using Magnum::Vector3;
using Magnum::Math::Deg;

std::atomic<uint> CameraController::count = 0;

CameraController::CameraController(Magnum::Platform::Application *app,
                                   Scene3D *scene)
    : CameraController(app, scene, CameraConfiguration{}){};

CameraController::CameraController(Magnum::Platform::Application *app,
				   Scene3D *scene, CameraConfiguration config)
    : _app(app), _config(config) {

  // rotations are manipulated by individual parent objects...
  // this makes rotations easier to reason about and serialize,
  // as we don't need quaternion multiplications or conversions to and from
  // matrices to serialize as three independent euler values

  _yaw_parent = std::make_unique<Object3D>(scene);
  _pitch_parent = std::make_unique<Object3D>(_yaw_parent.get());
  _camera_parent = std::make_unique<Object3D>(_pitch_parent.get());
  _roll_parent = std::make_unique<Object3D>(_camera_parent.get());
  _camera = std::make_unique<Camera3D>(*_roll_parent);

  _camera_parent->setTransformation(Matrix4::lookAt(
      {defaults::magnum::translation.z(), defaults::magnum::translation.y(),
       defaults::magnum::translation.x()},
      {}, Vector3::yAxis()));

  // deserialize our camera configuration into Magnum types
  auto rotation = Euler{Deg_f(_config.rotation[0]), Deg_f(_config.rotation[1]),
                        Deg_f(_config.rotation[2])};
  auto translation = Position{_config.translation[0], _config.translation[1],
                              _config.translation[2]};

  set_rotation(rotation, true);
  set_translation(translation);

  if (_config.id.empty()) {
    _config.id = pc::uuid::word();
  }
  _config.name = "camera_" + std::to_string(++CameraController::count);

  auto resolution = _config.rendering.resolution;
  _camera->setProjectionMatrix(Matrix4::perspectiveProjection(
      Deg_f(_config.fov), static_cast<float>(resolution[0]) / resolution[1],
      0.001f, 200.0f));

  if (resolution[0] == 0 || resolution[1] == 0)
    _config.rendering.resolution = pc::camera::defaults::rendering_resolution;

  pc::logger->info("{}x{}", resolution[0], resolution[1]);

  setup_framebuffer({resolution[0], resolution[1]});

  pc::logger->info("Initialised Camera Controller {} with id {}", _config.name,
                   _config.id);
}

CameraController::~CameraController() {
  CameraController::count--;
}

void CameraController::setup_framebuffer(Vector2i frame_size) {

  Vector2i scaled_size{
      static_cast<int>(frame_size.x() / _app->dpiScaling().x()),
      static_cast<int>(frame_size.y() / _app->dpiScaling().y())};

  auto aspect_ratio = frame_size.x() / static_cast<float>(frame_size.y());
  if (_config.rendering.scale_mode == ScaleMode::Span &&
      viewport_size.has_value()) {
    // automatically set frame height based on size of viewport
    aspect_ratio = viewport_size->x() / viewport_size->y();
    scaled_size.y() = scaled_size.x() / aspect_ratio;
    _config.rendering.resolution[1] =
        _config.rendering.resolution[0] / aspect_ratio;
  }

  if (scaled_size == _frame_size) {
    bind_framebuffer();
    return;
  }

  _frame_size = scaled_size;

  _frame_analyser.set_frame_size(_frame_size);

  // TODO replace the aspect ratio fix here when we have
  // a more solid handle over fov control
  _camera->setProjectionMatrix(Matrix4::perspectiveProjection(
      Deg_f(_config.fov), aspect_ratio, 0.001f, 200.0f));

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

GL::Texture2D &CameraController::color_frame() {
  std::unique_lock lock(_color_frame_mutex);
  return *_color;
}

GL::Texture2D &CameraController::analysis_frame() {
  return _frame_analyser.analysis_frame();
}

void CameraController::dispatch_analysis() {
  std::unique_lock lock(_color_frame_mutex);
  _frame_analyser.dispatch_analysis(*_color.get(), _config.frame_analysis);
}

int CameraController::analysis_time() {
  return _frame_analyser.analysis_time();
}

// TODO
bool _is_dz_started = false;
float _init_height_at_distance = 0.0f;

Matrix4 CameraController::make_projection_matrix() {
  auto frustum_height_at_distance = [](float distance, float fov) {
    return 2.0f * distance *
           std::tan(fov * 0.5f * Math::Constants<float>::pi() / 180.0f);
  };

  auto fov_for_height_and_distance = [](float height, float distance) {
    return 2.0f * std::atan(height * 0.5f / distance) * 180.0f /
           Math::Constants<float>::pi();
  };

  const auto focal_point =
      unproject(_frame_size / 2, depth_at(_frame_size / 2));
  auto camera_location = _camera_parent->transformation().translation();
  pc::logger->info("camera_location: {}, {}, {}", camera_location.x(),
                   camera_location.y(), camera_location.z());
  const auto target_distance = (camera_location - focal_point).length();

  if (!_is_dz_started) {
    _init_height_at_distance =
        frustum_height_at_distance(target_distance, _perspective_value);
    _is_dz_started = true;
  }

  auto fov = pc::math::remap(0.0f, 1.0f, 0.01f, 90.0f - 0.01f,
                             _perspective_value, true);

  auto height = frustum_height_at_distance(target_distance, fov);
  auto new_fov = fov_for_height_and_distance(height, target_distance);

  // Compute new distance to maintain the initial frustum height.
  // auto new_distance = _init_height_at_distance / (2.0f *
  // Math::tan(Deg(new_fov * 0.5f) * Math::Constants<float>::pi() / 180.0f));
  // Compute the new camera position to move towards or away from the subject as
  // FOV changes.
  auto direction = (camera_location - focal_point).normalized();
  // camera_location = focal_point + direction * -new_distance;

  // Update the camera's position.
  // auto transform = Matrix4::from(_camera_parent->transformation().rotation(),
  // 				 camera_location);
  // _camera_parent->setTransformation(transform);

  // return Matrix4::perspectiveProjection(Deg(new_fov), 4.0f / 3.0f, 0.001f,
  return Matrix4::perspectiveProjection(
      Deg(new_fov), _frame_size.x() / _frame_size.y(), 0.001f, 200.0f);
}

void CameraController::set_rotation(
    const Magnum::Math::Vector3<Magnum::Math::Rad<float>> &rotation,
    bool force) {

  std::array<float, 3> input_deg = {float(Deg_f(rotation.x())),
                                    float(Deg_f(rotation.y())),
                                    float(Deg_f(rotation.z()))};
  if (!force && _config.rotation[0] == input_deg[0] &&
      _config.rotation[1] == input_deg[1] &&
      _config.rotation[2] == input_deg[2]) {
    return;
  }

  _config.rotation = {float(Deg_f(rotation.x())), float(Deg_f(rotation.y())),
                      float(Deg_f(rotation.z()))};

  _yaw_parent->resetTransformation();
  auto y_rotation = rotation.y() - defaults::magnum::rotation.y();
  _yaw_parent->rotate(y_rotation, Vector3::yAxis());

  _pitch_parent->resetTransformation();
  auto x_rotation = rotation.x() - defaults::magnum::rotation.x();
  _pitch_parent->rotate(x_rotation, Vector3::zAxis());

  _roll_parent->resetTransformation();
  auto z_rotation = rotation.z() - defaults::magnum::rotation.z();
  _roll_parent->rotate(z_rotation, Vector3::zAxis());
}

void CameraController::set_translation(
    const Magnum::Math::Vector3<float> &translation) {
  _config.translation = {translation.x(), translation.y(), translation.z()};
  auto transform = _camera_parent->transformationMatrix();
  transform.translation() = {translation.z(), translation.y(), translation.x()};
  _camera_parent->setTransformation(transform);
}

void CameraController::dolly(
    Magnum::Platform::Sdl2Application::MouseScrollEvent &event) {
  const auto delta = event.offset().y();
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const auto focal_point = unproject(frame_centre, centre_depth);
  constexpr auto speed = 0.01f;
  _camera_parent->translateLocal(focal_point * delta * speed);
  auto translation = _camera_parent->transformationMatrix().translation();
  _config.translation = {translation.z(), translation.y(), translation.x()};
}

void CameraController::mouse_rotate(
    Magnum::Platform::Sdl2Application::MouseMoveEvent &event) {
  auto delta = Vector2{event.relativePosition()} * _rotate_speed;
  Euler rotation_amount;
  if (event.modifiers() ==
      Magnum::Platform::Sdl2Application::InputEvent::Modifier::Ctrl) {
    rotation_amount.z() = Rad(delta.y());
  } else {
    rotation_amount.x() = Rad(-delta.y());
    rotation_amount.y() = Rad(delta.x());
  }
  auto rotation = Euler{Deg_f(_config.rotation[0]), Deg_f(_config.rotation[1]),
                        Deg_f(_config.rotation[2])};
  set_rotation(rotation + rotation_amount);
}

void CameraController::mouse_translate(
    Magnum::Platform::Sdl2Application::MouseMoveEvent &event) {
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const Vector3 p = unproject(event.position(), centre_depth);
  const auto delta =
      Vector3{(float)event.relativePosition().x() * _move_speed.x(),
              (float)event.relativePosition().y() * _move_speed.y(), 0};
  auto translation = Position{_config.translation[0], _config.translation[1],
                              _config.translation[2]};
  set_translation(translation + delta);
}

CameraController &
CameraController::set_perspective(const Magnum::Float &perspective_value) {
  _perspective_value = Math::max(Math::min(perspective_value, 1.0f),
                                 std::numeric_limits<float>::min());
  _camera->setProjectionMatrix(make_projection_matrix());
  return *this;
}

CameraController &CameraController::zoom_perspective(
    Magnum::Platform::Sdl2Application::MouseScrollEvent &event) {
  auto delta = event.offset().y();
  set_perspective(_perspective_value - delta / 10);
  return *this;
}

Magnum::Vector3
CameraController::unproject(const Magnum::Vector2i &window_position,
                            Magnum::Float depth) const {
  const Vector2i view_position{window_position.x(),
                               _frame_size.y() - window_position.y() - 1};
  const Vector3 in{2 * Vector2{view_position} / Vector2{_frame_size} -
                       Vector2{1.0f},
                   depth * 2.0f - 1.0f};
  /*
  Use the following to get global coordinates instead of camera-relative:

  (_camera_parent->absoluteTransformationMatrix()*_camera->projectionMatrix().inverted()).transformPoint(in)
  */
  return _camera->projectionMatrix().inverted().transformPoint(in);
}

Float CameraController::depth_at(const Vector2i &window_position) {

  /* First scale the position from being relative to window size to being
     relative to framebuffer size as those two can be different on HiDPI
     systems */

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

  /* TODO: change to just Math::min<Float>(data.pixels<Float>() when the
     batch functions in Math can handle 2D views */
  return Math::min<Float>(data.pixels<Float>().asContiguous());
}

void CameraController::draw_imgui_controls() {

  using pc::gui::draw_slider;
  using pc::gui::vector_table;

  ImGui::SetNextItemOpen(_config.transform_open);
  if (ImGui::CollapsingHeader("Transform", _config.transform_open)) {
    _config.transform_open = true;
    auto &translate = _config.translation;
    if (vector_table("Translation", translate, -10.f, 10.f, 0.0f)) {
      set_translation(Position{translate[0], translate[1], translate[2]});
    }
    auto &rotate = _config.rotation;
    if (vector_table("Rotation", rotate, -360.0f, 360.0f, 0.0f)) {
      set_rotation(Euler{Deg_f(rotate[0]), Deg_f(rotate[1]), Deg_f(rotate[2])});
    }
  } else {
    _config.transform_open = false;
  }

  ImGui::SetNextItemOpen(_config.rendering_open);
  if (ImGui::CollapsingHeader("Rendering")) {
    _config.rendering_open = true;
    auto &rendering = _config.rendering;

    auto current_scale_mode = rendering.scale_mode;
    if (current_scale_mode == ScaleMode::Letterbox) {
      vector_table("Resolution", rendering.resolution, 2, 7680,
                   {pc::camera::defaults::rendering_resolution[0],
                    pc::camera::defaults::rendering_resolution[1]});
    } else if (current_scale_mode == ScaleMode::Span) {
      // disable setting y resolution manually in span mode,
      // it's inferred from the x resolution and window size
      vector_table("Resolution", rendering.resolution, 2, 7680,
                   {pc::camera::defaults::rendering_resolution[0],
                    pc::camera::defaults::rendering_resolution[1]},
                   {false, true});
    }

    auto scale_mode_i = static_cast<int>(current_scale_mode);
    const char *options[] = {"Span", "Letterbox"};
    ImGui::Combo("Scale mode", &scale_mode_i, options,
                 static_cast<int>(ScaleMode::Count));
    rendering.scale_mode = static_cast<ScaleMode>(scale_mode_i);
    if (rendering.scale_mode != current_scale_mode) {
      const auto aspect_ratio_policy =
          rendering.scale_mode == ScaleMode::Letterbox
              ? SceneGraph::AspectRatioPolicy::NotPreserved
              : SceneGraph::AspectRatioPolicy::Extend;
      _camera->setAspectRatioPolicy(aspect_ratio_policy);
    }

    ImGui::Spacing();

    ImGui::Checkbox("ground grid", &rendering.ground_grid);

    draw_slider<float>("point size", &rendering.point_size, 0.00001f, 0.08f,
                       0.0015f);
  } else {
    _config.rendering_open = false;
  }

  ImGui::SetNextItemOpen(_config.analysis_open);
  if (ImGui::CollapsingHeader("Analysis", _config.analysis_open)) {
    _config.analysis_open = true;
    auto &frame_analysis = _config.frame_analysis;
    ImGui::Checkbox("Enabled", &frame_analysis.enabled);
    if (frame_analysis.enabled) {

      vector_table("Resolution", frame_analysis.resolution, 2, 3840,
                   {pc::camera::defaults::analysis_resolution[0],
                    pc::camera::defaults::analysis_resolution[1]},
                   {}, {"width", "height"});
      vector_table("Binary threshold", frame_analysis.binary_threshold, 1, 255,
                   {50, 255}, {}, {"min", "max"});

      draw_slider<int>("Blur size", &frame_analysis.blur_size, 0, 40, 3);

      auto &canny = frame_analysis.canny;
      ImGui::Checkbox("Canny edge detection", &canny.enabled);
      if (canny.enabled) {
        draw_slider<int>("canny min", &canny.min_threshold, 0, 255, 100);
        draw_slider<int>("canny max", &canny.max_threshold, 0, 255, 255);
        int aperture_in = (canny.aperture_size - 1) / 2;
        draw_slider<int>("canny aperture", &aperture_in, 1, 3, 1);
        canny.aperture_size = aperture_in * 2 + 1;
      }

      if (gui::begin_tree_node("Contours", frame_analysis.contours_open)) {
        auto &contours = frame_analysis.contours;

        ImGui::Checkbox("Draw on viewport", &contours.draw);

        ImGui::Checkbox("Simplify", &contours.simplify);
        if (contours.simplify) {
          draw_slider<float>("arc scale", &contours.simplify_arc_scale,
                             0.000001f, 0.15f, 0.01f);
          draw_slider<float>("min area", &contours.simplify_min_area, 0.0001f,
                             2.0f, 0.0001f);
        }

        ImGui::Checkbox("Publish", &contours.publish);
        ImGui::Spacing();

        ImGui::Checkbox("Triangulate", &contours.triangulate.enabled);
        if (contours.triangulate.enabled) {
          ImGui::Checkbox("Draw triangles", &contours.triangulate.draw);
          ImGui::Checkbox("Publish triangles", &contours.triangulate.publish);
          draw_slider<float>("Min tri area", &contours.triangulate.minimum_area,
                             0.0f, 0.02f, 0.0f);
        }
        ImGui::TreePop();
      }

      if (gui::begin_tree_node("Optical Flow",
                               frame_analysis.optical_flow_open)) {
        auto &optical_flow = frame_analysis.optical_flow;
        ImGui::Checkbox("Enabled", &optical_flow.enabled);
        ImGui::Checkbox("Draw", &optical_flow.draw);
        ImGui::Checkbox("Publish", &optical_flow.publish);

        draw_slider<int>("Feature points", &optical_flow.feature_point_count,
                         25, 1000, 250);
        draw_slider<float>("Points distance",
                           &optical_flow.feature_point_distance, 0.001f, 30.0f,
                           10.0f);
        draw_slider<float>("Magnitude", &optical_flow.magnitude_scale, 0.1f,
                           5.0f, 1.0f);
        draw_slider<float>("Magnitude exponent",
                           &optical_flow.magnitude_exponent, 0.001f, 2.5f,
                           1.0f);
        draw_slider<float>("Minimum distance", &optical_flow.minimum_distance,
                           0.0f, 0.2f, 0.0f);
        draw_slider<float>("Maximum distance", &optical_flow.maximum_distance,
                           0.0f, 0.8f, 0.8f);

        ImGui::TreePop();
      }
    }
  } else {
    _config.analysis_open = false;
  }
}

} // namespace pc::camera
