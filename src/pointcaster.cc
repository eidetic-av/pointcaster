#ifdef _WIN32
#include <WinSock2.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include <spdlog/spdlog.h>

#include <random>

#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>

#include <Corrade/Containers/Pointer.h>
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

#include <Magnum/ImGuiIntegration/Context.hpp>

#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>

#include "devices/k4a/k4a_device.h"
#include "devices/k4a/k4a_driver.h"
#include "point_cloud.h"
#include "wireframe_objects.h"

namespace bob {

using namespace bob;
using namespace bob::sensors;
using namespace Magnum;
using namespace Math::Literals;

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

K4ADevice kinect;

class PointCaster : public Platform::Application {
public:
  explicit PointCaster(const Arguments &args);

protected:
  ImGuiIntegration::Context _imgui_context{NoCreate};

  Containers::Pointer<Scene3D> _scene;
  Containers::Pointer<SceneGraph::DrawableGroup3D> _drawable_group;

  // camera helpers
  Vector3 _default_cam_position{0.0f, 1.5f, 8.0f};
  Vector3 _default_cam_target{0.0f, 1.0f, 0.0f};
  Vector2i _prev_mouse_position;
  Vector3 _rotation_point, _translation_point;
  Float _last_depth;
  Containers::Pointer<Object3D> _object_camera;
  Containers::Pointer<SceneGraph::Camera3D> _camera;

  // Our particle system
  Containers::Pointer<PointCloudRenderer> _point_cloud_renderer;

  // Ground grid
  Containers::Pointer<WireframeGrid> _grid;

  bool _mouse_pressed = false;

  // helper functions for camera movement
  Float depthAt(const Vector2i &window_position);
  Vector3 unproject(const Vector2i &window_position, Float depth) const;

  void drawGui();

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

  // Set up the window
  const Vector2 dpi_scaling = this->dpiScaling({});
  Configuration conf;
  conf.setTitle("pointcaster");
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
  Containers::ArrayView<const char> font = rs.getRaw("SpaceGrotesk");
  ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
      const_cast<char *>(font.data()), Int(font.size()),
      14.0f * framebufferSize().x() / size.x(), &font_config);

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

  // Set up scene and camera
  _scene.reset(new Scene3D{});
  _drawable_group.reset(new SceneGraph::DrawableGroup3D{});

  _object_camera.reset(new Object3D{_scene.get()});
  _object_camera->setTransformation(
      Matrix4::lookAt(Vector3(0, 1.5, 8), Vector3(0, 1, 0), Vector3(0, 1, 0)));

  const auto viewport_size = GL::defaultFramebuffer.viewport().size();
  _camera.reset(new SceneGraph::Camera3D(*_object_camera));
  _camera
      ->setProjectionMatrix(Matrix4::perspectiveProjection(
	  45.0_degf, Vector2{viewport_size}.aspectRatio(), 0.01f, 1000.0f))
      .setViewport(viewport_size);

  // set default camera parameters
  _default_cam_position = Vector3(1.5f, 3.3f, 6.0f);
  _default_cam_target = Vector3(1.5f, 1.3f, 0.0f);
  _object_camera->setTransformation(Matrix4::lookAt(
      _default_cam_position, _default_cam_target, Vector3(0, 1, 0)));

  // initialise depth to the value at scene center
  _last_depth = ((_camera->projectionMatrix() * _camera->cameraMatrix())
		     .transformPoint({}) .z() + 1.0f) * 0.5f;

  // Set up ground grid
  _grid.reset(new WireframeGrid(_scene.get(), _drawable_group.get()));
  _grid->transform(Matrix4::scaling(Vector3(0.5f)) *
		   Matrix4::translation(Vector3(3, 0, -5)));

  // Enable depth test, render particles as sprites
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::ProgramPointSize);

  // set background color
  GL::Renderer::setClearColor(0x0d1117_rgbf);

  // Start the timer, loop at 144 Hz max
  setSwapInterval(1);
  setMinimalLoopPeriod(7);
}

void PointCaster::drawEvent() {
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
			       GL::FramebufferClear::Depth);
  _imgui_context.newFrame();

  // Enable text input, if needed/
  if (ImGui::GetIO().WantTextInput && !isTextInputActive())
    startTextInput();
  else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
    stopTextInput();

  // Fill point cloud
  auto points = kinect.getPointCloud();
  _point_cloud_renderer.reset(new PointCloudRenderer(0.005f));
  _point_cloud_renderer->_points = points;
  _point_cloud_renderer->setDirty();

  // Draw objects
  _point_cloud_renderer->draw(_camera, framebufferSize());
  _camera->draw(*_drawable_group);

  drawGui();

  _imgui_context.updateApplicationCursor(*this);

  // Render ImGui window
  GL::Renderer::enable(GL::Renderer::Feature::Blending);
  GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);

  _imgui_context.drawFrame();

  GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::disable(GL::Renderer::Feature::Blending);

  // The context is double-buffered, swap buffers
  swapBuffers();

  // Run the next frame immediately
  redraw();
}

void PointCaster::drawGui() {
  ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.5f);
  ImGui::Begin("options", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.6f);
  ImGui::PopItemWidth();
  ImGui::End();
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

  return _camera->projectionMatrix().inverted().transformPoint(in);
}

void PointCaster::viewportEvent(ViewportEvent &event) {
  // resize main framebuffer
  GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

  // relayout imgui
  _imgui_context.relayout(Vector2{event.windowSize()} / event.dpiScaling(),
			  event.windowSize(), event.framebufferSize());

  // recompute the camera's projection matrix
  _camera->setViewport(event.framebufferSize());
}

void PointCaster::keyPressEvent(KeyEvent &event) {
  switch (event.key()) {
  case KeyEvent::Key::Q:
    exit(0);
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

  // Update camera
  _prev_mouse_position = event.position();
  const Float currentDepth = depthAt(event.position());
  const Float depth = currentDepth == 1.0f ? _last_depth : currentDepth;
  _translation_point = unproject(event.position(), depth);

  /* Update the rotation point only if we're not zooming against infinite
     depth or if the original rotation point is not yet initialized */
  if (currentDepth != 1.0f || _rotation_point.isZero()) {
    _rotation_point = _translation_point;
    _last_depth = depth;
  }

  _mouse_pressed = true;
}

void PointCaster::mouseReleaseEvent(MouseEvent &event) {
  _mouse_pressed = false;
  if (_imgui_context.handleMouseReleaseEvent(event)) {
    event.setAccepted(true);
  }
}

void PointCaster::mouseMoveEvent(MouseMoveEvent &event) {
  if (_imgui_context.handleMouseMoveEvent(event)) {
    event.setAccepted(true);
    return;
  }

  const Vector2 delta = 3.0f *
			Vector2(event.position() - _prev_mouse_position) /
			Vector2(framebufferSize());
  _prev_mouse_position = event.position();

  if (!event.buttons())
    return;

  /* Translate */
  if (event.modifiers() & MouseMoveEvent::Modifier::Shift) {
    const Vector3 p = unproject(event.position(), _last_depth);
    _object_camera->translateLocal(_translation_point - p); /* is Z always 0? */
    _translation_point = p;

    /* Rotate around rotation point */
  } else {
    _object_camera->transformLocal(Matrix4::translation(_rotation_point) *
				   Matrix4::rotationX(-0.51_radf * delta.y()) *
				   Matrix4::rotationY(-0.51_radf * delta.x()) *
				   Matrix4::translation(-_rotation_point));
  }

  event.setAccepted();
}

void PointCaster::mouseScrollEvent(MouseScrollEvent &event) {
  const Float delta = event.offset().y();
  if (Math::abs(delta) < 1.0e-2f)
    return;

  if (_imgui_context.handleMouseScrollEvent(event)) {
    /* Prevent scrolling the page */
    event.setAccepted();
    return;
  }

  const Float current_depth = depthAt(event.position());
  const Float depth = current_depth == 1.0f ? _last_depth : current_depth;
  const Vector3 p = unproject(event.position(), depth);
  /* Update the rotation point only if we're not zooming against infinite
     depth or if the original rotation point is not yet initialized */
  if (current_depth != 1.0f || _rotation_point.isZero()) {
    _rotation_point = p;
    _last_depth = depth;
  }

  /* Move towards/backwards the rotation point in cam coords */
  _object_camera->translateLocal(_rotation_point * delta * 0.1f);
}

} // namespace bob

MAGNUM_APPLICATION_MAIN(bob::PointCaster);
