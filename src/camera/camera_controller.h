#pragma once

#include "../analysis/analyser_2d.h"
#include "camera_config.gen.h"
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string_view>

namespace pc::camera {

using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;
using Scene3D =
    Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;

using Magnum::SceneGraph::Camera3D;

class CameraController : public analysis::Analyser2DHost {
public:
  static std::atomic<std::size_t> count;
  const std::string session_id;

  std::optional<Magnum::Vector2> viewport_size;

  explicit CameraController(Magnum::Platform::Application *app, Scene3D *scene,
                            std::string_view host_session_id,
                            CameraConfiguration config = {});

  ~CameraController();

  std::string_view host_id() const override { return session_id; }

  CameraConfiguration &config() { return _config; };
  const CameraConfiguration &config() const { return _config; }

  Camera3D &camera() const { return *_camera; }

  void setup_frame(Magnum::Vector2i frame_size);

  Magnum::GL::Texture2D &color_frame();
  Magnum::GL::Texture2D &analysis_frame();

  void dispatch_analysis();
  int analysis_time();

  void set_distance(const float metres);
  void add_distance(const float metres);

  void set_orbit(const Float2 degrees);
  void add_orbit(const Float2 degrees);

  void set_roll(const float degrees);
  void add_roll(const float degrees);

  void set_translation(const Float3 metres);
  void add_translation(const Float3 metres);

  void dolly(Magnum::Platform::Sdl2Application::MouseScrollEvent &event);
  void mouse_orbit(Magnum::Platform::Sdl2Application::MouseMoveEvent &event);
  void mouse_translate(Magnum::Platform::Sdl2Application::MouseMoveEvent &event,
                       bool lock_y_axis = false);

  void draw_imgui_controls();

  std::vector<gui::OverlayText> labels();

private:
  CameraConfiguration _config;

  Magnum::Platform::Application *_app;
  pc::analysis::Analyser2D _frame_analyser;

  std::unique_ptr<Object3D> _anchor;
  std::unique_ptr<Object3D> _orbit_parent_left_right;
  std::unique_ptr<Object3D> _orbit_parent_up_down;
  std::unique_ptr<Object3D> _camera_parent;
  std::unique_ptr<Camera3D> _camera;

  Magnum::Vector2i _frame_size;
  std::unique_ptr<Magnum::GL::Texture2D> _color;
  std::unique_ptr<Magnum::GL::Renderbuffer> _depth_stencil;
  std::unique_ptr<Magnum::GL::Framebuffer> _framebuffer;

  Magnum::Vector2 _rotate_speed{0.035f, 0.035f};
  Magnum::Vector2 _move_speed{-0.0035f, 0.0035f};

  std::mutex _color_frame_mutex;

  void bind_framebuffer();

  void reset_projection_matrix();

  Magnum::Matrix4 make_projection_matrix();

  Magnum::Vector3 unproject(const Magnum::Vector2i &window_position,
                            Magnum::Float depth) const;
  Magnum::Float depth_at(const Magnum::Vector2i &window_position);
};

inline std::optional<std::reference_wrapper<CameraController>>
    interacting_camera_controller;

} // namespace pc::camera
