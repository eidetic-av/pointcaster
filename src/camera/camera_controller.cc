#include "camera_controller.h"
#include "../gui_helpers.h"
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
#include <spdlog/spdlog.h>
#include <vector>

namespace pc::camera {

using namespace Magnum;
using Magnum::Matrix4;
using Magnum::Quaternion;
using Magnum::Vector3;
using Magnum::Math::Deg;
using Magnum::Image2D;

std::atomic<uint> CameraController::count = 0;

CameraController::CameraController(Magnum::Platform::Application *app,
                                   Scene3D *scene)
    : CameraController(app, scene, CameraConfiguration{}){};

CameraController::CameraController(Magnum::Platform::Application *app,
				   Scene3D *scene, CameraConfiguration config)
    : _app(app), _config(config), _analysis_thread([this](auto stop_token) {
	this->frame_analysis(stop_token);
      }) {

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

  _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend);

  // deserialize our camera configuration into Magnum types
  auto rotation = Euler{Deg_f(_config.rotation[0]), Deg_f(_config.rotation[1]),
			Deg_f(_config.rotation[2])};
  auto translation = Position{_config.translation[0], _config.translation[1],
		      _config.translation[2]};
  auto fov = Deg_f(_config.fov);

  setRotation(rotation, true);
  setTranslation(translation, true);

  _camera->setProjectionMatrix(
      Matrix4::perspectiveProjection(fov, 4.0f / 3.0f, 0.001f, 200.0f));

  if (_config.id.empty()) {
    _config.id = pc::uuid::word();
  }
  _config.name = "camera_" + std::to_string(++CameraController::count);
  if (_config.rendering.resolution[0] == 0) {
    const auto res = _app->framebufferSize();
    _config.rendering.resolution = {res.x(), res.y()};
  }

  spdlog::info("Initialised Camera Controller {} with id {}", _config.name,
	  _config.id);
}

CameraController::~CameraController() { CameraController::count--; }

void CameraController::setupFramebuffer(Vector2i frame_size) {
  if (frame_size == _frame_size) return;


  std::lock(_dispatch_analysis_mutex, _color_frame_mutex,
	    _analysis_frame_buffer_data_mutex);
  std::lock_guard lock_dispatch(_dispatch_analysis_mutex, std::adopt_lock);
  std::lock_guard lock_color(_color_frame_mutex, std::adopt_lock);
  std::lock_guard lock(_analysis_frame_buffer_data_mutex, std::adopt_lock);

  _frame_size = frame_size;

  frame_size /= _app->dpiScaling();

  _color = std::make_unique<GL::Texture2D>();
  _color->setStorage(1, GL::TextureFormat::RGBA8, _frame_size);

  _analysis_frame = std::make_unique<GL::Texture2D>();
  _analysis_frame->setStorage(1, GL::TextureFormat::RGBA8, _frame_size);

  _depth_stencil = std::make_unique<GL::Renderbuffer>();
  _depth_stencil->setStorage(GL::RenderbufferFormat::Depth24Stencil8,
                             _frame_size);

  _framebuffer = std::make_unique<GL::Framebuffer>(Range2Di{{}, _frame_size});

  _framebuffer->attachTexture(GL::Framebuffer::ColorAttachment{0}, *_color, 0);

  _framebuffer->attachRenderbuffer(
      GL::Framebuffer::BufferAttachment::DepthStencil, *_depth_stencil);
}

void CameraController::bindFramebuffer() {
  _framebuffer->clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth)
      .bind();
}

GL::Texture2D &CameraController::color_frame() {
  std::unique_lock lock(_color_frame_mutex);
  return *_color;
}

GL::Texture2D &CameraController::analysis_frame() {
  if (_analysis_frame_buffer_updated) {
    std::lock_guard lock(_analysis_frame_buffer_data_mutex);
    // move updated buffer data into our analysis frame Texture2D...
    // we first need to create an OpenGL Buffer for it
    GL::BufferImage2D buffer{
	Magnum::GL::PixelFormat::RGBA, Magnum::GL::PixelType::UnsignedByte,
	_frame_size, _analysis_frame_buffer_data, GL::BufferUsage::StaticDraw};
    // then we can set the data
    _analysis_frame->setSubImage(0, {}, buffer);
  }
  return *_analysis_frame;
}

void CameraController::dispatch_analysis() {
  if (!_config.frame_analysis.enabled) return;
  std::lock(_dispatch_analysis_mutex, _color_frame_mutex);
  std::lock_guard lock_dispatch(_dispatch_analysis_mutex, std::adopt_lock);
  std::lock_guard lock_color(_color_frame_mutex, std::adopt_lock);
  // move the image onto the CPU for use in our analysis thread
  _analysis_image = _color->image(0, Image2D{PixelFormat::RGBA8Unorm});
  _analysis_config = _config.frame_analysis;
  _analysis_condition_variable.notify_one();
}

void CameraController::frame_analysis(std::stop_token stop_token) {
  while (!stop_token.stop_requested()) {

    std::optional<Magnum::Image2D> image_opt;
    std::optional<FrameAnalysisConfiguration> config_opt;

    {
      std::unique_lock dispatch_lock(_dispatch_analysis_mutex);

      _analysis_condition_variable.wait(dispatch_lock, [&] {
        auto values_filled =
            (_analysis_image.has_value() && _analysis_config.has_value());
        return stop_token.stop_requested() || values_filled;
      });

      if (stop_token.stop_requested()) break;

      // start move the data onto this thread
      image_opt = std::move(_analysis_image);
      _analysis_image.reset();

      config_opt = std::move(_analysis_config);
      _analysis_config.reset();
    }

    // now we are free to process our image without holding the main thread

    auto& image = *image_opt;
    auto input_frame_size = image.size();
    auto& analysis_config = *config_opt;

    // start by wrapping the image data in an opencv matrix type
    cv::Mat input_mat(input_frame_size.y(), input_frame_size.x(), CV_8UC4,
		image.data());

    // and scale to analysis size if changed
    cv::Mat scaled_mat;
    if (input_frame_size.x() != analysis_config.resolution[0] ||
        input_frame_size.y() != analysis_config.resolution[1]) {
      auto& resolution = analysis_config.resolution;
      if (resolution[0] == 0) resolution[0] = input_frame_size.x();
      if (resolution[1] == 0) resolution[1] = input_frame_size.y();
      cv::resize(input_mat, scaled_mat, {resolution[0], resolution[1]}, 0, 0,
		 cv::INTER_LINEAR);
    } else {
      scaled_mat = input_mat;
    }

    auto& contours = analysis_config.contours;

    // convert the image to grayscale
    cv::Mat grey;
    cv::cvtColor(scaled_mat, grey, cv::COLOR_RGBA2GRAY);

    // blur the image
    if (contours.blur_size > 0)
      cv::blur(grey, grey, cv::Size(contours.blur_size, contours.blur_size));

    // canny edge detection
    cv::Mat canny;
    cv::Canny(grey, canny, contours.canny_min_threshold,
	      contours.canny_max_threshold, contours.canny_aperture_size);

    // find the countours
    std::vector<std::vector<cv::Point>> contour_list;
    cv::findContours(canny, contour_list, cv::RETR_LIST,
		     cv::CHAIN_APPROX_SIMPLE);

    // normalise the contours
    std::vector<std::vector<cv::Point2f>> contour_list_norm;

    std::vector<cv::Point2f> norm_contour;
    for (auto &contour : contour_list) {
      norm_contour.clear();
      for (auto &point : contour) {
	norm_contour.push_back(
	    {point.x / static_cast<float>(analysis_config.resolution[0]),
	     point.y / static_cast<float>(analysis_config.resolution[1])});
      }
      contour_list_norm.push_back(norm_contour);
    }

    // poly simplification
    if (contours.simplify) {
      for (auto it = contour_list_norm.begin();
	   it != contour_list_norm.end();) {
	auto &contour = *it;

	// remove contours that are too small
        const double area = cv::contourArea(contour);
        if (area < contours.simplify_min_area / 1000) {
          it = contour_list_norm.erase(it);
          continue;
        }

        // simplify the remaining shapes
        const double epsilon =
            contours.simplify_arc_scale * cv::arcLength(contour, true);
        cv::approxPolyDP(contour, contour, epsilon, true);

        ++it;
      }
    }

      // TODO the contours can be published on our MQTT network here

      // if we don't need to draw the result onto our camera view,
      // we can finish here with just the contour arrays
      if (!analysis_config.draw_on_viewport)
        continue;

      // create a new RGBA image initialised to fully transparent
      cv::Mat output_mat(input_frame_size.y(), input_frame_size.x(), CV_8UC4,
                         cv::Scalar(0, 0, 0, 0));

      // scale the contours to the size of our output image
      std::vector<std::vector<cv::Point>> contour_list_scaled;

      std::vector<cv::Point> scaled_contour;
      for (auto &contour : contour_list_norm) {
        scaled_contour.clear();
        for (auto &point : contour) {
          scaled_contour.push_back(
              {static_cast<int>(point.x * input_frame_size.x()),
               static_cast<int>(point.y * input_frame_size.y())});
          // spdlog::info("{}. {}", scaled_contour.back().x,
          // scaled_contour.back().y);
        }
        contour_list_scaled.push_back(scaled_contour);
      }

      // draw the contours onto our transparent image
      const cv::Scalar line_colour{255, 0, 0, 255};
      constexpr auto line_thickness = 4;
      cv::drawContours(output_mat, contour_list_scaled, -1, line_colour,
                       line_thickness, cv::LINE_4);

      // copy the resulting cv::Mat data into our buffer data container

      auto element_count = output_mat.total() * output_mat.channels();
      std::unique_lock lock(_analysis_frame_buffer_data_mutex);
      _analysis_frame_buffer_data =
          Containers::Array<uint8_t>(NoInit, element_count);

      std::copy(output_mat.datastart, output_mat.dataend,
                _analysis_frame_buffer_data.data());

      _analysis_frame_buffer_updated = true;
    }
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
  spdlog::info("camera_location: {}, {}, {}", camera_location.x(),
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

void CameraController::setRotation(
				   const Magnum::Math::Vector3<Magnum::Math::Rad<float>> &rotation, bool force) {

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

void CameraController::setTranslation(
				      const Magnum::Math::Vector3<float> &translation, bool force) {
  if (!force && _config.translation[0] == translation.x() &&
      _config.translation[1] == translation.y() &&
      _config.translation[2] == translation.z()) {
    return;
  }
  _config.translation = {translation.x(), translation.y(), translation.z()};
  auto transform = _camera_parent->transformationMatrix();
  transform.translation() = {translation.z(), translation.y(), translation.x()};
  _camera_parent->setTransformation(transform);
}

void CameraController::dolly(Magnum::Platform::Sdl2Application::MouseScrollEvent &event) {
  const auto delta = event.offset().y();
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const auto focal_point = unproject(frame_centre, centre_depth);
  constexpr auto speed = 0.01f;
  _camera_parent->translateLocal(focal_point * delta * speed);
  auto translation = _camera_parent->transformationMatrix().translation();
  _config.translation = {translation.z(), translation.y(), translation.x()};
}

void CameraController::mouseRotate(
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
  setRotation(rotation + rotation_amount);
}

void CameraController::mouseTranslate(
    Magnum::Platform::Sdl2Application::MouseMoveEvent &event) {
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const Vector3 p = unproject(event.position(), centre_depth);
  const auto delta =
      Vector3{(float)event.relativePosition().x() * _move_speed.x(),
	      (float)event.relativePosition().y() * _move_speed.y(), 0};
  auto translation = Position{_config.translation[0], _config.translation[1],
		      _config.translation[2]};
  setTranslation(translation + delta);
}

CameraController &
CameraController::setPerspective(const Magnum::Float &perspective_value) {
  _perspective_value = Math::max(Math::min(perspective_value, 1.0f),
                                 std::numeric_limits<float>::min());
  _camera->setProjectionMatrix(make_projection_matrix());
  return *this;
}

CameraController &CameraController::zoomPerspective(
    Magnum::Platform::Sdl2Application::MouseScrollEvent &event) {
  auto delta = event.offset().y();
  setPerspective(_perspective_value - delta / 10);
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

Float CameraController::depth_at(const Vector2i& window_position) {

  /* First scale the position from being relative to window size to being
     relative to framebuffer size as those two can be different on HiDPI
     systems */

  const Vector2i position = window_position * Vector2{ _app->framebufferSize() } /
     Vector2{_app->windowSize()};
  const Vector2i fbPosition{ position.x(),
                           GL::defaultFramebuffer.viewport().sizeY() -
                               position.y() - 1 };

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
  
  if (ImGui::TreeNode("Transform")) {

    ImGui::TextDisabled("Translation");
    auto translate = _config.translation;
    draw_slider<float>("x", &translate[0], -10, 10);
    draw_slider<float>("y", &translate[1], -10, 10);
    draw_slider<float>("z", &translate[2], -10, 10);
    setTranslation(Position{translate[0], translate[1], translate[2]});

    ImGui::TextDisabled("Rotation");
    auto rotate = _config.rotation;
    draw_slider<float>("x", &rotate[0], -360, 360);
    draw_slider<float>("y", &rotate[1], -360, 360);
    draw_slider<float>("z", &rotate[2], -360, 360);
    setRotation(Euler{Deg_f(rotate[0]), Deg_f(rotate[1]), Deg_f(rotate[2])});

    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Rendering")) {
    auto& rendering = _config.rendering;
    ImGui::Checkbox("ground grid", &rendering.ground_grid);
    ImGui::TextDisabled("Resolution");
    ImGui::InputInt("width", &rendering.resolution[0]);
    ImGui::SameLine();
    ImGui::InputInt("height", &rendering.resolution[1]);
    ImGui::Spacing();
    draw_slider<float>("point size", &rendering.point_size, 0.00001f, 0.08f, 0.0015f);
    ImGui::Spacing();
    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Frame Analysis")) {
    auto& frame_analysis = _config.frame_analysis;
    ImGui::Checkbox("Enabled", &frame_analysis.enabled);
    if (frame_analysis.enabled) {
      ImGui::Checkbox("Draw on viewport", &frame_analysis.draw_on_viewport);

      ImGui::TextDisabled("Resolution");
      ImGui::InputInt("width", &frame_analysis.resolution[0]);
      ImGui::SameLine();
      ImGui::InputInt("height", &frame_analysis.resolution[1]);
      ImGui::Spacing();

      if (ImGui::TreeNode("Contour Detection")) {
	auto& contours = frame_analysis.contours;

	draw_slider<int>("blur size", &contours.blur_size, 0, 40, 3);
	draw_slider<int>("canny min", &contours.canny_min_threshold, 0, 255, 100);
	draw_slider<int>("canny max", &contours.canny_max_threshold, 0, 255, 255);
	int aperture_in = (contours.canny_aperture_size - 1) / 2;
	draw_slider<int>("canny aperture", &aperture_in, 1, 3, 1);
	contours.canny_aperture_size = aperture_in * 2 + 1;

	ImGui::Checkbox("Simplify", &contours.simplify);
        if (contours.simplify) {
	  draw_slider<float>("arc scale", &contours.simplify_arc_scale, 0.000001f, 0.015f, 0.001f);
	  draw_slider<float>("min area", &contours.simplify_min_area, 0.0001f, 2.0f, 0.0001f);
        }

        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }
}

} // namespace pc::camera
