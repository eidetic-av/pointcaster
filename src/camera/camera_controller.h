#pragma once

#include "camera_config.h"
#include <Magnum/Image.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <opencv2/opencv.hpp>

namespace pc::camera {

typedef Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>
    Object3D;
using Scene3D = Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;
using Camera3D = Magnum::SceneGraph::Camera3D;

using uint = unsigned int;

class CameraController {
public:
  static std::atomic<uint> count;

  std::optional<Magnum::Vector2> viewport_size;

  CameraController(Magnum::Platform::Application *app, Scene3D *scene);
  CameraController(Magnum::Platform::Application *app, Scene3D *scene, CameraConfiguration config);

  ~CameraController();

  CameraConfiguration &config() { return _config; };
  const std::string name() const { return _config.name; };
  Camera3D &camera() const { return *_camera; }

  void set_rotation(const Magnum::Math::Vector3<Magnum::Math::Rad<float>>& rotation, bool force = false);
  void set_translation(const Magnum::Math::Vector3<float>& translation);
  void dolly(Magnum::Platform::Sdl2Application::MouseScrollEvent& event);

  void mouse_rotate(Magnum::Platform::Sdl2Application::MouseMoveEvent &event);
  void mouse_translate(Magnum::Platform::Sdl2Application::MouseMoveEvent &event);

  // 0 is regular perspective, 1 is orthographic
  CameraController &set_perspective(const Magnum::Float &value);
  CameraController &zoom_perspective(Magnum::Platform::Sdl2Application::MouseScrollEvent &event);

  void setup_framebuffer(Magnum::Vector2i frame_size);

  Magnum::GL::Texture2D& color_frame();
  Magnum::GL::Texture2D& analysis_frame();

  void dispatch_analysis();
  int analysis_time();

  void draw_imgui_controls();

private:
  Magnum::Platform::Application* _app;
  CameraConfiguration _config;

  std::unique_ptr<Object3D> _yaw_parent;
  std::unique_ptr<Object3D> _pitch_parent;
  std::unique_ptr<Object3D> _camera_parent;
  std::unique_ptr<Object3D> _roll_parent;
  std::unique_ptr<Camera3D> _camera;

  Magnum::Vector2i _frame_size;
  std::unique_ptr<Magnum::GL::Texture2D> _color;
  std::unique_ptr<Magnum::GL::Renderbuffer> _depth_stencil;
  std::unique_ptr<Magnum::GL::Framebuffer> _framebuffer;

  Magnum::Vector2 _rotate_speed{-0.0035f, 0.0035f};
  Magnum::Vector2 _move_speed{-0.0035f, 0.0035f};

  Magnum::Deg _fov{45};
  Magnum::Float _perspective_value{0.5};
  Magnum::Vector3 _translation{};

  std::mutex _color_frame_mutex;
  std::mutex _analysis_frame_mutex;
  std::mutex _dispatch_analysis_mutex;
  std::mutex _analysis_frame_buffer_data_mutex;

  std::unique_ptr<Magnum::GL::Texture2D> _analysis_frame;
  std::optional<Magnum::Image2D> _analysis_image;
  std::optional<FrameAnalysisConfiguration> _analysis_config;
  std::jthread _analysis_thread;
  Corrade::Containers::Array<uint8_t> _analysis_frame_buffer_data;
  std::atomic_bool _analysis_frame_buffer_updated;
  std::condition_variable _analysis_condition_variable;
  std::optional<cv::Mat> _previous_analysis_image;
  std::atomic<std::chrono::milliseconds> _analysis_time;

  void frame_analysis(std::stop_token stop_token);

  void bind_framebuffer();

  Magnum::Matrix4 make_projection_matrix();

  Magnum::Vector3 unproject(const Magnum::Vector2i &window_position,
                            Magnum::Float depth) const;
  Magnum::Float depth_at(const Magnum::Vector2i &window_position);
};

} // namespace pc::camera
