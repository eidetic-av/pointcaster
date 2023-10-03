#include "camera_controller.h"
#include "../analysis/analyser_2d_config.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../math.h"
#include "../parameters.h"
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
using Magnum::Platform::Sdl2Application;
using Magnum::Image2D;
using Magnum::Matrix4;
using Magnum::Quaternion;
using Magnum::Vector3;
using Magnum::Math::Deg;

std::atomic<std::size_t> CameraController::count = 0;

CameraController::CameraController(Magnum::Platform::Application *app,
                                   Scene3D *scene)
    : CameraController(app, scene, CameraConfiguration{}){};

CameraController::CameraController(Magnum::Platform::Application *app,
				   Scene3D *scene, CameraConfiguration config)
    : _app(app), _config(config), _frame_analyser(this) {

  _anchor = std::make_unique<Object3D>(scene);
  _orbit_parent_left_right = std::make_unique<Object3D>(_anchor.get());
  _orbit_parent_up_down = std::make_unique<Object3D>(_orbit_parent_left_right.get());
  _camera_parent = std::make_unique<Object3D>(_orbit_parent_up_down.get());
  _camera = std::make_unique<Camera3D>(*_camera_parent);

  // apply any loaded configuration
  set_distance(_config.transform.distance);
  set_orbit(_config.transform.orbit);
  set_roll(_config.transform.roll);
  set_translation(_config.transform.translation);

  if (_config.id.empty()) {
    _config.id = pc::uuid::word();
  }
  _config.name = "camera_" + std::to_string(++CameraController::count);

  reset_projection_matrix();

  auto &resolution = _config.rendering.resolution;
  if (resolution[0] == 0 || resolution[1] == 0)
    resolution = defaults::rendering_resolution;

  pc::logger->info("{}x{}", resolution[0], resolution[1]);

  setup_frame({resolution[0], resolution[1]});

  pc::logger->info("Initialised Camera Controller {} with id {}", _config.name,
                   _config.id);

  declare_parameters(name(), _config);
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

Float CameraController::depth_at(const Vector2i &window_position) {
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

  return Math::min<Float>(data.pixels<Float>().asContiguous());
}

void CameraController::draw_imgui_controls() {

  using pc::gui::slider;
  using pc::gui::vector_table;

  ImGui::SetNextItemOpen(_config.transform.unfolded);
  if (ImGui::CollapsingHeader("Transform", _config.transform.unfolded)) {
    _config.transform.unfolded = true;

    ImGui::Checkbox("transform.show_anchor", &_config.transform.show_anchor);

    vector_table(name(), "transform.translation", _config.transform.translation,
		 -10.f, 10.f, 0.0f);

    vector_table(name(), "transform.orbit", _config.transform.orbit, -360.f,
                 360.f, 0.0f);

    slider(name(), "transform.roll", _config.transform.roll, -360.f, 360.f,
           0.0f);

    slider(name(), "scroll_precision", _config.scroll_precision, 1, 30, 1);

  } else {
    _config.transform.unfolded = false;
  }

  ImGui::SetNextItemOpen(_config.rendering.unfolded);
  if (ImGui::CollapsingHeader("Rendering")) {
    _config.rendering.unfolded = true;
    auto &rendering = _config.rendering;

    auto current_scale_mode = rendering.scale_mode;
    if (current_scale_mode == (int)ScaleMode::Letterbox) {

      vector_table(name(), "rendering.resolution", rendering.resolution, 2,
                   7680, pc::camera::defaults::rendering_resolution);

    } else if (current_scale_mode == (int)ScaleMode::Span) {
      // disable setting y resolution manually in span mode,
      // it's inferred from the x resolution and window size
      vector_table(name(), "rendering.resolution", rendering.resolution, 2,
                   7680, pc::camera::defaults::rendering_resolution,
                   {false, true});
    }

    auto scale_mode_i = static_cast<int>(current_scale_mode);
    const char *options[] = {"Span", "Letterbox"};
    ImGui::Combo("Scale mode", &scale_mode_i, options,
                 static_cast<int>(ScaleMode::Count));
    rendering.scale_mode = scale_mode_i;
    if (rendering.scale_mode != current_scale_mode) {
      const auto aspect_ratio_policy =
	rendering.scale_mode == (int)ScaleMode::Letterbox
              ? SceneGraph::AspectRatioPolicy::NotPreserved
              : SceneGraph::AspectRatioPolicy::Extend;
      _camera->setAspectRatioPolicy(aspect_ratio_policy);
    }
    ImGui::Spacing();

    auto was_ortho = rendering.orthographic;
    if (ImGui::Checkbox("rendering.orthographic", &rendering.orthographic)) {
      MinMax<float> from;
      MinMax<float> to;
      if (was_ortho && !rendering.orthographic) {
	from = defaults::orthographic_clipping;
	to = defaults::perspective_clipping;
      } else if (!was_ortho && rendering.orthographic) {
	from = defaults::perspective_clipping;
	to = defaults::orthographic_clipping;
      }
      rendering.clipping.min = math::remap(from.min, from.max, to.min, to.max,
					   rendering.clipping.min);
      rendering.clipping.max = math::remap(from.min, from.max, to.min, to.max,
					   rendering.clipping.max);
      reset_projection_matrix();
    }

    ImGui::Spacing();

    auto clipping_minmax = rendering.orthographic
                               ? defaults::orthographic_clipping
                               : defaults::perspective_clipping;

    if (slider(name(), "rendering.clipping.min", rendering.clipping.min,
	       clipping_minmax.min, clipping_minmax.max, clipping_minmax.min)) {
      reset_projection_matrix();
    };
    if (slider(name(), "rendering.clipping.max", rendering.clipping.max,
	       clipping_minmax.min, clipping_minmax.max, clipping_minmax.max)) {
      reset_projection_matrix();
    };

    ImGui::Spacing();

    ImGui::Checkbox("rendering.ground_grid", &rendering.ground_grid);
    ImGui::Checkbox("rendering.skeletons", &rendering.skeletons);

    // ** >>>> TODO HERet
    slider(name(), "rendering.point_size", rendering.point_size, 0.0f, 0.01f,
	   0.0015f);

  } else {
    _config.rendering.unfolded = false;
  }

  ImGui::SetNextItemOpen(_config.analysis.unfolded);
  if (ImGui::CollapsingHeader("Analysis", _config.analysis.unfolded)) {
    _config.analysis.unfolded = true;

    auto &analysis = _config.analysis;
    ImGui::Checkbox("Enabled", &analysis.enabled);
    if (analysis.enabled) {
      ImGui::SameLine();
      ImGui::Checkbox("analysis.use_cuda", &analysis.use_cuda);

      vector_table(name(), "analysis.resolution", analysis.resolution, 2, 3840,
                   pc::camera::defaults::analysis_resolution, {},
                   {"width", "height"});

      vector_table(name(), "analysis.binary_threshold",
                   analysis.binary_threshold, 1, 255, Int2{50, 255}, {},
                   {"min", "max"});

      slider(name(), "analysis.blur_size", analysis.blur_size, 0, 40, 1);

      auto &canny = analysis.canny;
      ImGui::Checkbox("Canny edge detection", &canny.enabled);
      if (canny.enabled) {
	slider(name(), "analysis.canny.min_threshold", canny.min_threshold, 0,
	       255, 100);
	slider(name(), "analysis.canny.max_threshold", canny.max_threshold, 0,
	       255, 255);
	int aperture_in = (canny.aperture_size - 1) / 2;
	slider(name(), "analysis.canny.aperture_size", aperture_in, 1, 3, 1);
	canny.aperture_size = aperture_in * 2 + 1;
      }

      if (gui::begin_tree_node("Contours", analysis.contours.unfolded)) {
        auto &contours = analysis.contours;

        ImGui::Checkbox("analysis.contours.draw", &contours.draw);

        ImGui::Checkbox("analysis.contours.label", &contours.label);
        ImGui::Spacing();

        ImGui::Checkbox("analysis.contours.simplify", &contours.simplify);
        if (contours.simplify) {
	  slider(name(), "analysis.contours.simplify_arc_scale",
		 contours.simplify_arc_scale, 0.000001f, 0.15f, 0.01f);
	  slider(name(), "analysis.contours.simplify_min_area",
		 contours.simplify_min_area, 0.0001f, 2.0f, 0.0001f);
        }
        ImGui::Spacing();

        ImGui::Checkbox("analysis.contours.publish", &contours.publish);
        ImGui::Checkbox("analysis.contours.publish_centroids", &contours.publish_centroids);
        ImGui::Spacing();

        ImGui::Checkbox("Triangulate", &contours.triangulate.enabled);
        if (contours.triangulate.enabled) {
          ImGui::Checkbox("analysis.contours.triangulate.draw",
                          &contours.triangulate.draw);
          ImGui::Checkbox("analysis.contours.triangulate.publish",
                          &contours.triangulate.publish);
	  slider(name(), "analysis.contours.triangulate.minimum_area",
		 contours.triangulate.minimum_area, 0.0f, 0.02f, 0.0f);
        }
        ImGui::TreePop();
      }

      if (gui::begin_tree_node("Optical Flow",
                               analysis.optical_flow.unfolded)) {
        auto &optical_flow = analysis.optical_flow;
        ImGui::Checkbox("analysis.optical_flow.enabled", &optical_flow.enabled);
        ImGui::Checkbox("analysis.optical_flow.draw", &optical_flow.draw);
        ImGui::Checkbox("analysis.optical_flow.publish", &optical_flow.publish);

	slider(name(), "analysis.optical_flow.feature_point_count",
	       optical_flow.feature_point_count, 25, 1000, 250);
	slider(name(), "analysis.optical_flow.feature_point_distance",
	       optical_flow.feature_point_distance, 0.001f, 30.0f, 10.0f);
        if (analysis.use_cuda) {
          float quality_level =
              optical_flow.cuda_feature_detector_quality_cutoff * 10.0f;
	  slider(name(), "analysis.optical_flow.quality_level", quality_level,
		 0.01f, 1.0f, 0.1f);
          optical_flow.cuda_feature_detector_quality_cutoff =
              quality_level / 10.0f;
        }
	slider(name(), "analysis.optical_flow.magnitude_scale",
	       optical_flow.magnitude_scale, 0.1f, 5.0f, 1.0f);
	slider(name(), "analysis.optical_flow.magnitude_exponent",
	       optical_flow.magnitude_exponent, 0.001f, 2.5f, 1.0f);
	slider(name(), "analysis.optical_flow.minimum_distance",
	       optical_flow.minimum_distance, 0.0f, 0.2f, 0.0f);
	slider(name(), "analysis.optical_flow.maximum_distance",
	       optical_flow.maximum_distance, 0.0f, 0.8f, 0.8f);

        ImGui::TreePop();
      }

      if (gui::begin_tree_node("Output", analysis.output.unfolded)) {
        vector_table(name(), "analysis.output.scale", analysis.output.scale,
                     0.0f, 2.0f, 1.0f);
        vector_table(name(), "analysis.output.offset", analysis.output.offset,
                     -1.0f, 1.0f, 0.0f);
        ImGui::TreePop();
      }
    }
  } else {
    _config.analysis.unfolded = false;
  }
}

std::vector<gui::OverlayText> CameraController::labels() {
  return _frame_analyser.analysis_labels();
}

} // namespace pc::camera
