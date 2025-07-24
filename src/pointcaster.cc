#include "pointcaster.h"

#include "camera/camera_config.gen.h"
#include "camera/camera_controller.h"
#include "devices/device.h"
#include "devices/k4a/k4a_config.gen.h"
#include "devices/k4a/k4a_device.h"
#include "devices/orbbec/orbbec_device_config.gen.h"
#include "devices/sequence/ply_sequence_player_config.gen.h"
#include "fonts/IconsFontAwesome6.h"
#include "gui/catpuccin.h"
#include "gui/main_viewport.h"
#include "gui/windows.h"
#include "imgui_internal.h"
#include "operators/session_operator_host.h"
#include "parameters.h"
#include "pch.h"
#include "point_cloud_renderer/point_cloud_renderer_config.gen.h"
#include "session.gen.h"
#include "workspace.gen.h"
#include <ImGuizmo.h>
#include <algorithm>
#include <imgui.h>
#include <imgui_stdlib.h>
#include <serdepp/serializer.hpp>
#include <variant>

#ifdef WIN32
#include <SDL2/SDL_Clipboard.h>
#else
#include <SDL2/SDL.h>
#endif

#ifdef WITH_OSC
#include "osc/osc_client.h"
#include "osc/osc_server.h"
#endif

// Temporary headers to be removed
#include <k4a/k4a.h>

namespace pc {

PointCaster::PointCaster(const Arguments &args)
    : Platform::Application(args, NoCreate) {

  pc::logger->info("This is pointcaster");

  MainThreadDispatcher::init();

  // Get OS resolution
  SDL_DisplayMode dm;
  if (SDL_GetDesktopDisplayMode(0, &dm) != 0)
    pc::logger->warn("Failed to get display resolution using SDL: {}",
                     SDL_GetError());
  else {
    _display_resolution = {dm.w, dm.h};
    pc::logger->info("SDL2 returned display resolution: {}x{}", dm.w, dm.h);
  }

  // Set up the window
  Sdl2Application::Configuration conf;
  conf.setTitle("Pointcaster");

  // TODO figure out how to persist window size accross launches
  if (_display_resolution.has_value()) {
    auto &resolution = _display_resolution.value();
    constexpr auto start_res_scale = 2.0f / 3.0f;
    constexpr auto start_ratio = 2.0f / 3.0f;
    auto start_width = int(resolution.x() / 1.5f * start_res_scale);
    auto start_height = int(start_width * start_ratio);
    conf.setSize({start_width, start_height}, {1.5f, 1.5f});
  }

  conf.setWindowFlags(Sdl2Application::Configuration::WindowFlag::Resizable);

  // Try 8x MSAA, fall back to zero if not possible.
  // Enable only 2x MSAA if we have enough DPI.
  GLConfiguration gl_conf;
  gl_conf.setSampleCount(8);
  if (!tryCreate(conf, gl_conf)) create(conf, gl_conf.setSampleCount(0));

  // Set up ImGui
  ImGui::CreateContext();
  pc::gui::init_parameter_styles();

  // Don't save imgui layout to a file, handle it manually
  ImGui::GetIO().IniFilename = nullptr;

  const auto size = Vector2(windowSize()) / dpiScaling();

  // load fonts from resources
  Utility::Resource rs("data");

  auto font = rs.getRaw("AtkinsonHyperlegibleRegular");
  // we need to malloc font_data because once we pass it to imgui, imgui expects
  // to be responsible for freeing it
  char *font_data = static_cast<char *>(malloc(font.size()));
  memcpy(font_data, font.data(), font.size());

  ImFontConfig font_config;
  font_config.FontDataOwnedByAtlas = true;
  constexpr auto font_size = 15.0f;
  _font = ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
      font_data, font.size(), font_size * framebufferSize().x() / size.x(),
      &font_config);

  auto mono_font = rs.getRaw("IosevkaArtisan");
  char *mono_font_data = static_cast<char *>(malloc(mono_font.size()));
  memcpy(mono_font_data, mono_font.data(), mono_font.size());

  ImFontConfig mono_font_config;
  mono_font_config.FontDataOwnedByAtlas = true;
  const auto mono_font_size = 14.5f;
  _mono_font = ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
      mono_font_data, mono_font.size(),
      mono_font_size * framebufferSize().x() / size.x(), &mono_font_config);

  static const ImWchar icons_ranges[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
  const auto icon_font_size = 15.0f;
  const auto icon_font_size_pixels =
      icon_font_size * framebufferSize().x() / size.x();

  auto font_icons = rs.getRaw("FontAwesomeSolid");
  char *font_icons_data = static_cast<char *>(malloc(font_icons.size()));
  memcpy(font_icons_data, font_icons.data(), font_icons.size());

  ImFontConfig icons_config;
  icons_config.MergeMode = true;
  icons_config.PixelSnapH = true;
  icons_config.FontDataOwnedByAtlas = true;
  icons_config.GlyphMinAdvanceX = icon_font_size;

  constexpr auto empty_deleter = [](auto *ptr) {};

  _icon_font = std::shared_ptr<ImFont>(
      ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
          font_icons_data, font_icons.size(), icon_font_size_pixels,
          &icons_config, icons_ranges),
      empty_deleter);
  pc::gui::icon_font = _icon_font;

  const auto small_icon_font_size = 6.0f;
  const auto small_icon_font_size_pixels = small_icon_font_size *
                                           framebufferSize().x() /
                                           static_cast<float>(size.x());

  char *small_icon_font_data = static_cast<char *>(malloc(font_icons.size()));
  memcpy(small_icon_font_data, font_icons.data(), font_icons.size());

  ImFontConfig icons_config_small;
  icons_config.MergeMode = false;
  icons_config.PixelSnapH = true;
  icons_config.FontDataOwnedByAtlas = true;
  icons_config_small.GlyphMinAdvanceX = small_icon_font_size;
  _icon_font_small = std::shared_ptr<ImFont>(
      ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
          small_icon_font_data, font_icons.size(), small_icon_font_size_pixels,
          &icons_config_small, icons_ranges),
      empty_deleter);
  pc::gui::icon_font_small = _icon_font_small;

  // enable window docking
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ImGui::GetIO().ConfigFlags |= ImGuiDockNodeFlags_PassthruCentralNode;
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // enable keyboard tab & arrows navigation
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // for editing parameters with the keyboard
  auto backspace = ImGui::GetIO().KeyMap[ImGuiKey_Backspace];
  ImGui::GetIO().AddInputCharacter(backspace);
  ImGui::GetIO().AddKeyEvent(ImGuiKey_Backspace, true);
  ImGui::GetIO().AddKeyEvent(ImGuiKey_Backspace, false);

  _imgui_context = ImGuiIntegration::Context(
      *ImGui::GetCurrentContext(), Vector2(windowSize()) / dpiScaling(),
      windowSize(), framebufferSize());

  // Set up the link from imgui to the system clipboard using SDL
  ImGui::GetIO().SetClipboardTextFn = [](void *, const char *text) {
    SDL_SetClipboardText(text);
  };
  ImGui::GetIO().GetClipboardTextFn = [](void *) -> const char * {
    return SDL_GetClipboardText();
  };

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
  _scene_root = std::make_unique<SceneGraph::DrawableGroup3D>();

  _ground_grid =
      std::make_unique<WireframeGrid>(_scene.get(), _scene_root.get());
  _ground_grid->transform(Matrix4::scaling(Vector3(1.0f)) *
                          Matrix4::translation(Vector3(0, 0, 0)));

//   const auto fetch_devices = [this] {
//     std::lock_guard lock(this->_session_devices_mutex);
//     return this->workspace.devices;
//   };

// #ifndef WIN32
//   const auto fetch_usb_config = [this] {
//     std::lock_guard lock(this->_usb_config_mutex);
//     return this->workspace.usb.value();
//   };
//   _usb_monitor =
//       std::make_unique<UsbMonitor>(fetch_usb_config, fetch_devices);
// #endif

  OrbbecDevice::init_context();
  _session_operator_graph = std::make_unique<OperatorGraph>("Session");

  // load last session
  auto data_dir = path::get_or_create_data_directory();

  for (const auto &entry : std::filesystem::directory_iterator(data_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".toml")
      continue;

    auto write_time = std::filesystem::last_write_time(entry);
    if (last_modified_workspace_file.empty() || write_time > last_write_time) {
      last_modified_workspace_file = entry.path();
      last_write_time = write_time;
    }
  }

  if (last_modified_workspace_file.empty()) {
    pc::logger->info("No previous workspace found. Creating new workspace.");
    workspace = {.sessions = {{.id = uuid::word(), .label = "session_0"}}};
    auto file_path = data_dir / (std::format("{}.toml", uuid::word()));
    save_workspace(file_path);
  } else {
    pc::logger->info("Found previous session file");
    load_workspace(last_modified_workspace_file);
  }
  sync_session_instances();

  // if there is no usb configuration, initialise a default
  if (!workspace.usb.has_value()) {
    workspace.usb = pc::devices::UsbConfiguration{};
  }
  // TODO, should this parameter declaration be done inside UsbConfiguration
  // constructor?
  // parameters::declare_parameters("usb", workspace.usb.value());

  const auto viewport_size = GL::defaultFramebuffer.viewport().size();

  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  // render particles as sprites
  GL::Renderer::enable(GL::Renderer::Feature::ProgramPointSize);

  // set background color
  GL::Renderer::setClearColor(0x000000_rgbf);

  _point_cloud_renderer = std::make_unique<PointCloudRenderer>();
  _sphere_renderer = std::make_unique<SphereRenderer>();

  // set up the spheres that render skeleton joints
  /* Setup points (render as spheres) */
  {
    const std::size_t total_joint_count = K4ABT_JOINT_COUNT * 5;
    const Vector3 start_pos{-999, -999, -999};
    const Vector3 joint_size{0.015};

    _sphere_instance_data =
        Containers::Array<SphereInstanceData>{NoInit, total_joint_count};

    for (std::size_t i = 0; i < total_joint_count; i++) {
      /* Fill in the instance data. Most of this stays the same, except
         for the translation */
      _sphere_instance_data[i].transformationMatrix =
          Matrix4::translation(start_pos) * Matrix4::scaling(joint_size);
      _sphere_instance_data[i].normalMatrix =
          _sphere_instance_data[i].transformationMatrix.normalMatrix();
      _sphere_instance_data[i].color =
          Color3{Vector3(std::rand(), std::rand(), std::rand()) /
                 Magnum::Float(RAND_MAX)};
    }

    _sphere_shader =
        Shaders::PhongGL{Shaders::PhongGL::Configuration{}.setFlags(
            Shaders::PhongGL::Flag::VertexColor |
            Shaders::PhongGL::Flag::InstancedTransformation)};
    _sphere_instance_buffer = GL::Buffer{};
    _sphere_mesh = MeshTools::compile(Primitives::icosphereSolid(2));
    _sphere_mesh.addVertexBufferInstanced(
        _sphere_instance_buffer, 1, 0, Shaders::PhongGL::TransformationMatrix{},
        Shaders::PhongGL::NormalMatrix{}, Shaders::PhongGL::Color3{});
    _sphere_mesh.setInstanceCount(_sphere_instance_data.size());
  }

  // Start the timer, loop at 144 Hz max
  setSwapInterval(1);
  setMinimalLoopPeriod(7);

  try_init_unique_member<MqttClient>(workspace.mqtt, _mqtt);
  try_init_unique_member<MidiDevice>(workspace.midi, _midi);
  try_init_unique_member<SyncServer>(workspace.sync_server, _sync_server);

  try_init_unique_member<Radio>(workspace.radio, _radio,
                                session_operator_hosts[0], *_sync_server);

#ifdef WITH_OSC
  try_init_unique_member<OscClient>(workspace.osc_client, _osc_client);
  try_init_unique_member<OscServer>(workspace.osc_server, _osc_server);
#endif

  _snapshots_context = std::make_unique<Snapshots>();

  TweenManager::create();
  _timeline.start();
}

void PointCaster::quit() {
  std::lock_guard lock(pc::devices::devices_access);
  pc::devices::attached_devices.clear();
  OrbbecDevice::destroy_context();
  exit(0);
}

void PointCaster::save_and_quit() {
  save_workspace();
  quit();
}

void PointCaster::save_workspace() {
  if (last_modified_workspace_file.empty()) {
    auto data_dir = path::get_or_create_data_directory();
    auto file_path = data_dir / (uuid::word() + ".toml");
    save_workspace(file_path);
  } else {
    save_workspace(last_modified_workspace_file);
  }
}

void PointCaster::save_workspace(std::filesystem::path file_path) {
  pc::logger->info("Saving session to {}", file_path.string());

  // save imgui layout to an adjacent file
  std::size_t imgui_layout_size;
  auto imgui_layout_data = ImGui::SaveIniSettingsToMemory(&imgui_layout_size);
  std::vector<char> layout_data(imgui_layout_data,
                                imgui_layout_data + imgui_layout_size);
  std::filesystem::path layout_file_path = file_path;
  layout_file_path.replace_extension(".layout");
  std::ofstream layout_file(layout_file_path, std::ios::binary);
  layout_file.write(layout_data.data(), layout_data.size());

  // create output copy
  PointcasterWorkspace output_workspace = workspace;

  // serialize configs from camera controllers & operator hosts, since they own
  // the config for the app's lifetime
  for (size_t i = 0; i < camera_controllers.size(); i++) {
    if (output_workspace.sessions.size() <= i) {
      pc::logger->warn("Mismatched camera controllers & workspace sessions count during workspace save");
      break;
    }
    output_workspace.sessions[i].camera = camera_controllers[i]->config();
    output_workspace.sessions[i].session_operator_host =
        session_operator_hosts[i]._config;
  }

  // parse parameter topic lists
  output_workspace.published_params = parameters::published_parameter_topics();
  output_workspace.pushed_params = parameters::pushed_parameter_topics();

  // save any friendly names assigned to operators
  output_workspace.operator_names.clear();
  const auto &op_names = pc::operators::operator_friendly_names;
  std::for_each(op_names.begin(), op_names.end(), [&](auto &&kvp) {
    const auto &[id, name] = kvp;
    output_workspace.operator_names[name] = id;
  });

  // similarly save any colors assigned to operators' bounding boxes
  output_workspace.operator_colors.clear();
  const auto &boxes = pc::operators::operator_bounding_boxes;
  std::for_each(boxes.begin(), boxes.end(), [&](auto &&kvp) {
    const auto &[id, box] = kvp;
    const auto &name = pc::operators::operator_friendly_names[id];
    auto color = box->getColor();
    output_workspace.operator_colors[name] = {color.r(), color.g(), color.b(),
                                              color.a()};
  });

  auto workspace_toml = serde::serialize<serde::toml_v>(output_workspace);

  // add anything that isn't serlializable using serdepp (like variant types)
  // directly to the toml
  workspace_toml["devices"] = toml::array{};
  auto &devices = workspace_toml["devices"].as_array();
  {
    std::lock_guard lock(devices::device_configs_access);
    for (const auto &device_config_ref : devices::device_configs) {
      toml::table device_table{};
      std::visit(
          [&](auto &&config) {
            using T = std::decay_t<decltype(config)>;
            const auto type_info = serde::type_info<T>;
            const auto member_names = type_info.member_names().members();

            // insert the variant type name to the device table
            device_table.insert_or_assign("device_variant", toml::value(T::Name));
            // iterate each struct member to add to the device table
            auto handle_member = [&](std::string_view member_name,
                                                auto index) {
              using MemberType = pc::reflect::type_at_t<typename T::MemberTypes,
                                                        decltype(index)::value>;
              // retrieve the actual member reference at runtime
              auto &member =
                  type_info.template member<decltype(index)::value>(config);
              toml::value member_value =
                  serde::serialize<serde::toml_v>(member);
              device_table.insert_or_assign(std::string{member_name},
                                            member_value);
            };
            // using an immediately invoked lambda to provide member names and
            // the associated index as a compile-time constant to the handler
            // above
            [&handle_member,
             &member_names]<std::size_t... Is>(std::index_sequence<Is...>) {
              ((handle_member(member_names[Is],
                              std::integral_constant<std::size_t, Is>{})),
               ...);
            }(std::make_index_sequence<T::MemberCount>{});
          },
          device_config_ref.get());
      devices.emplace_back(std::move(device_table));
    }
  }

  // write the finished toml
  std::ofstream(file_path, std::ios::binary) << toml::format(workspace_toml);

  last_modified_workspace_file = file_path;
  // last_write_time = std::filesystem::last_write_time(file_path);
}

template <typename T>
void deserialize_device_toml(toml::table &device_table,
                             DeviceConfigurationVariant &variant) {
  variant = T{};
  auto &config = std::get<T>(variant);

  constexpr auto info = serde::type_info<T>;
  const auto member_names = info.member_names().members();

  // read each field from toml and write into struct
  auto assign_field = [&](std::string_view name, auto idx) {
    using I = decltype(idx);
    using MemberType =
        pc::reflect::type_at_t<typename T::MemberTypes, I::value>;

    auto &member = device_table.at(std::string{name}); // toml node
    MemberType val = serde::deserialize<MemberType>(member);
    info.template member<I::value>(config) = std::move(val);
  };

  // expand for each member index
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (assign_field(member_names[Is], std::integral_constant<std::size_t, Is>{}),
     ...);
  }(std::make_index_sequence<T::MemberCount>{});
}

void PointCaster::load_workspace(std::filesystem::path file_path) {
  pc::logger->info("Loading workspace from {}", file_path.string());

  std::ifstream file(file_path, std::ios::binary);

  if (!file) {
    pc::logger->warn("Failed to open file");
    workspace = {};
    return;
  }

  toml::value file_toml;
  try {
    file_toml = toml::parse(file);
  } catch (const toml::syntax_error &e) {
    pc::logger->error("Failed to parse config file toml");
    pc::logger->error(e.what());
    workspace = {};
    return;
  }

  // deserialize anything that was serialized manually and doesnt just exist in
  // the PointcasterWorkspace struct

  // deserialize devices
  if (file_toml.contains("devices")) {
    for (auto &device_entry : file_toml["devices"].as_array()) {
      auto &device_table = device_entry.as_table();
      auto type_name = device_table.at("device_variant").as_string();
      device_table.erase("device_variant");

      DeviceConfigurationVariant device_config;

      try {

        if (type_name == OrbbecDeviceConfiguration::Name) {
          deserialize_device_toml<OrbbecDeviceConfiguration>(device_table,
                                                             device_config);
        } else if (type_name == AzureKinectConfiguration::Name) {
          deserialize_device_toml<AzureKinectConfiguration>(device_table,
                                                            device_config);
        } else if (type_name == PlySequencePlayerConfiguration::Name) {
          deserialize_device_toml<PlySequencePlayerConfiguration>(
              device_table, device_config);
        }

      } catch (std::exception &e) {
        pc::logger->error("Unable to deserialize Device configuration: {}",
                          e.what());
        workspace = {};
        return;
      }

      // TODO
      // mutable required because load_device can't take a const ref to
      // config... pretty hacky
      run_async(
          [this, config = device_config]() mutable { load_device(config); });
    }
  }

  // deserialize everything else automatically
  PointcasterWorkspace result;

  try {
    result = serde::deserialize<PointcasterWorkspace>(file_toml);
  } catch (const std::exception &e) {
    pc::logger->error("Failed to deserialize config file toml");
    pc::logger->error(e.what());
    workspace = {};
    return;
  }

  // check if there is an adjacent .layout file
  // and load if so
  std::filesystem::path layout_file_path = file_path;
  layout_file_path.replace_extension(".layout");

  // and load if so
  std::ifstream layout_file(layout_file_path, std::ios::binary | std::ios::ate);
  if (layout_file.is_open()) {
    std::streamsize layout_size = layout_file.tellg();
    layout_file.seekg(0, std::ios::beg);

    std::vector<char> layout_data(layout_size);
    layout_file.read(layout_data.data(), layout_size);

    ImGui::LoadIniSettingsFromMemory(layout_data.data(), layout_size);

  } else {
    pc::logger->warn("Failed to open adjacent .layout file");
  }

  try {

    if (!result.usb.has_value()) { result.usb = UsbConfiguration{}; }

    // if ((*_session.usb).open_on_launch) {
    //   if (_session.devices.empty()) {
    //     open_kinect_sensors();
    //   } else {
    //     // get saved device configurations and populate the device list
    //     run_async([this] {
    //       for (auto &[device_id, device_config] : _session.devices) {
    //         load_device(device_config);
    //       }
    //     });
    //   }
    // }

    // run_async([this] {
    //   for (auto &[device_id, device_entry] : workspace.devices) {
    //     auto& [device_variant, variant] = device_entry;
    //     load_device(device_variant, variant);
    //   }
    // });

    // unpack published parameters, initialising parameter state
    if (result.published_params.has_value()) {
      for (auto &parameter_id : *result.published_params) {
        parameters::parameter_states.emplace(
            parameter_id, parameters::ParameterState::Publish);
      }
    }
    // unpack pushed parameters
    for (auto &parameter_id : result.pushed_params) {
      parameters::parameter_states.emplace(parameter_id,
                                           parameters::ParameterState::Push);
    }

    // unpack any serialized friendly operator names
    for (const auto &[name, operator_id] : result.operator_names) {
      set_operator_friendly_name(operator_id, name);
    }

    // unpack any serialized operator colors as bounding boxes that are
    // hidden - this reserves the color for use by the operator that owns the
    // bounding box

    auto all_workspace_operators =
        result.sessions | std::views::transform([](auto &session) {
          return std::views::all(session.session_operator_host.operators);
        }) |
        std::views::join;

    for (auto &operator_variant : all_workspace_operators) {
      std::visit(
          [&](auto &&operator_config) {
            // search through our serialized operator colors to find if this
            // operator_config is present
            for (const auto &[friendly_name, color] : result.operator_colors) {
              if (!operator_friendly_name_exists(friendly_name)) continue;
              // if we have an existing operator, grab its ID
              const auto operator_id =
                  operator_ids_from_friendly_names.at(friendly_name);
              // and create its bounding box
              set_or_create_bounding_box(
                  operator_id, {}, {}, *_scene.get(), *_scene_root.get(), false,
                  Magnum::Color4{color[0], color[1], color[2], color[3]});
            }
          },
          operator_variant);
    }

    workspace = result;
    pc::logger->info("Loaded workspace '{}'", file_path.filename().string());

  } catch (...) {
    workspace = {};
    pc::logger->error("Failed to load workspace (Unknown exception)");
  }
}

void PointCaster::load_device(DeviceConfigurationVariant &config) {
  std::visit(
      [&](auto &&device_config) {
        using T = std::decay_t<decltype(device_config)>;
        std::string device_id(device_config.id);
        pc::logger->info("Loading device '{}'", device_id);
        std::lock_guard lock(devices::devices_access);
        devices::attached_devices.emplace_back(
            std::make_shared<typename T::DeviceType>(device_config));
        // if the device is not already present in each session's
        // active_device_map, make sure it's in there as active
        for (auto &session : workspace.sessions) {
          if (!session.active_devices.contains(device_config.id)) {
            session.active_devices.emplace(device_config.id, true);
          }
        }
      },
      config);
}

void PointCaster::render_cameras() {
  auto skeletons = devices::scene_skeletons();

  for (size_t i = 0; i < workspace.sessions.size(); i++) {
    auto &session = workspace.sessions[i];
    auto &session_operator_host = session_operator_hosts[i];
    auto &camera_controller = camera_controllers[i];
    auto& camera = (*camera_controller).camera();
    auto &rendering_config = camera_controller->config().rendering;

    const auto frame_size =
        Vector2i{int(rendering_config.resolution[0] / dpiScaling().x()),
                 int(rendering_config.resolution[1] / dpiScaling().y())};

    if (frame_size.x() < 1 || frame_size.y() < 1) continue;

    camera_controller->setup_frame(frame_size);

    PointCloud &points = devices::compute_or_get_point_cloud(
        session.id, session.active_devices,
        session_operator_host._config.operators);

    _point_cloud_renderer->points = points;
    _point_cloud_renderer->setDirty();

    // enable or disable wireframe ground depending on camera settings
    _ground_grid->set_visible(rendering_config.ground_grid);

    // draw shaders
    _point_cloud_renderer->draw(camera, rendering_config);

    if (rendering_config.skeletons) {
      if (!skeletons.empty()) {
        int i = 0;
        for (auto &skeleton : skeletons) {
          for (auto &joint : skeleton) {
            auto pos = joint.first;
            _sphere_instance_data[i].transformationMatrix.translation() = {
                pos.x / 1000.0f, pos.y / 1000.0f, pos.z / 1000.0f};
            i++;
          }
        }
        _sphere_instance_buffer.setData(_sphere_instance_data,
                                        GL::BufferUsage::DynamicDraw);
        GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
        _sphere_shader.setProjectionMatrix(camera.projectionMatrix())
            .setTransformationMatrix(camera.cameraMatrix())
            .setNormalMatrix(camera.cameraMatrix().normalMatrix())
            .draw(_sphere_mesh);
        GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
      }
    }

    // render camera
    camera.draw(*_scene_root);
    // TODO analysis needs to happen every session not just the selected onee
    camera_controller->dispatch_analysis();

    // if (i == workspace.selected_session_index) {
    //   selected_session_points = points;
    //   selected_camera_controller = *camera_controller;
    //   selected_renderer_config = rendering_config;
    // }

    // if (points.empty()) {

      // TODO


      // // TODO this vector should defs be removed
      // // the synthesized_point_cloud api just requires a reference to a
      // // list which we dont have at the moment
      // std::vector<operators::OperatorHostConfiguration> fix_me_operator_list{
      //     _session_operator_host->_config};

      // // using namespace std::chrono;
      // // using namespace std::chrono_literals;

      // // auto begin = high_resolution_clock::now();

      // points = devices::synthesized_point_cloud(fix_me_operator_list);

      // // auto end = high_resolution_clock::now();
      // // auto duration_ms = duration_cast<milliseconds>(end - begin).count();
      // // pc::logger->warn("devices::synthesized_point_cloud: {}ms",
      // // duration_ms);

      // _session_operator_host->_config = fix_me_operator_list[0];

      // // TODO
      // // get the points from THIS camera
      // // auto& config = camera_controller->config();
      // // int node_id = _session_operator_graph->node_id_from_reference(config);
      // // auto path = _session_operator_graph->path_to(node_id);
      // // if (path.size() > 1) pc::logger->info(path.size());

      // if (rendering_config.snapshots) points += snapshots::point_cloud();

    // }
  }

  GL::defaultFramebuffer.bind();
}

void PointCaster::sync_session_instances() {
  // build a map of session labels per id
  session_label_from_id.clear();
  for (auto &session : workspace.sessions) {
    session_label_from_id[session.id] = session.label;
  }

  // build a lookup of existing controllers by session-id
  std::unordered_map<std::string, std::unique_ptr<CameraController>>
      controller_map;
  controller_map.reserve(camera_controllers.size());
  for (auto &camera_controller : camera_controllers) {
    controller_map.emplace(camera_controller->session_id,
                           std::move(camera_controller));
  }

  // rebuild in workspace order, reusing existing camera controllers, or
  // creating any new ones
  std::vector<std::unique_ptr<CameraController>> new_controllers;
  new_controllers.reserve(workspace.sessions.size());
  for (auto &session : workspace.sessions) {
    auto it = controller_map.find(session.id);
    if (it != controller_map.end()) {
      new_controllers.push_back(std::move(it->second));
    } else {
      new_controllers.push_back(std::make_unique<CameraController>(
          this, _scene.get(), session.id, session.camera));
    }
  }
  camera_controllers.swap(new_controllers);

  // same idea for session_operator_hosts
  std::unordered_map<std::string, SessionOperatorHost> operator_host_map;
  operator_host_map.reserve(session_operator_hosts.size());
  for (auto &operator_host : session_operator_hosts) {
    operator_host_map.emplace(operator_host.session_id,
                              std::move(operator_host));
  }

  std::vector<SessionOperatorHost> new_operator_hosts;
  new_operator_hosts.reserve(workspace.sessions.size());
  for (auto &session : workspace.sessions) {
    auto it = operator_host_map.find(session.id);
    if (it != operator_host_map.end()) {
      new_operator_hosts.push_back(std::move(it->second));
    } else {
      new_operator_hosts.emplace_back(*this, *_scene.get(), *_scene_root.get(),
                                      session.id,
                                      session.session_operator_host);
    }
  }
  session_operator_hosts.swap(new_operator_hosts);
}

// void PointCaster::load_k4a_device(AzureKinectConfiguration &config,
//                                   std::string_view target_id) {
//   loading_device = true;
//   try {
//     auto device = std::make_shared<K4ADevice>(config, target_id);
//     std::lock_guard lock(pc::devices::devices_access);
//     pc::devices::attached_devices.push_back(device);
//     for (auto &session : workspace.sessions) {
//       if (!session.active_devices.contains(config.id)) {
//         session.active_devices[config.id] = true;
//       }
//     }
//   } catch (k4a::error e) { pc::logger->error(e.what()); } catch (...) {
//     pc::logger->error("Failed to open device. (Unknown exception)");
//   }
//   loading_device = false;
// }

void PointCaster::open_kinect_sensors() {
  run_async([this] {
    loading_device = true;
    size_t attached_device_count = 0;
    {
      std::lock_guard lock(K4ADevice::devices_access);
      attached_device_count = K4ADevice::attached_devices.size();
    }
    const auto installed_device_count = k4a::device::get_installed_count();
    pc::logger->info("Found {} k4a devices", (int)installed_device_count);
    for (std::size_t i = attached_device_count; i < installed_device_count;
         i++) {
      DeviceConfigurationVariant config = AzureKinectConfiguration{
          .id = uuid::word(), .serial_number = K4ADevice::get_serial_number(i)};
      load_device(config);
    }
    loading_device = false;
  });
}

void PointCaster::open_orbbec_sensor(std::string_view ip) {
  pc::logger->info("Opening Orbbec sensor at {}", ip);
  DeviceConfigurationVariant config =
      OrbbecDeviceConfiguration{.id = uuid::word(), .ip = std::string(ip)};
  load_device(config);
}

void PointCaster::open_ply_sequence() {
  auto ply_sequence_config = PlySequencePlayer::load_directory();
  if (!ply_sequence_config.directory.empty()) {
    std::lock_guard lock(devices::devices_access);
    devices::attached_devices.emplace_back(
        std::make_shared<PlySequencePlayer>(ply_sequence_config));
    for (auto &session : workspace.sessions) {
      if (!session.active_devices.contains(ply_sequence_config.id)) {
        session.active_devices[ply_sequence_config.id] = true;
      }
    }
  } else {
    pc::logger->error("No directory");
  }
}

void PointCaster::draw_onscreen_log() {
  // if draw log
  ImVec2 log_window_size{400, 150};
  const auto viewport_size = ImGui::GetWindowSize();

  using namespace std::chrono_literals;
  auto latest_messages = pc::logger_lines(10, 10s);

  if (latest_messages.size() > 0) {
    ImGui::SetNextWindowPos({viewport_size.x - log_window_size.x,
                             viewport_size.y - log_window_size.y});
    ImGui::SetNextWindowSize({log_window_size.x, log_window_size.y});
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushFont(_mono_font);
    constexpr auto log_window_flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
    ImGui::Begin("log", nullptr, log_window_flags);
    ImGui::PushTextWrapPos(log_window_size.x);
    for (auto log_entry : latest_messages) {
      ImGui::Spacing();

      auto [log_level, message] = log_entry;

      switch (log_level) {
      case spdlog::level::info:
        ImGui::PushStyleColor(ImGuiCol_Text,
                              ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); // white
        ImGui::Text(" [info]");
        ImGui::SameLine();
        break;
      case spdlog::level::warn:
        ImGui::PushStyleColor(ImGuiCol_Text,
                              ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // yellow
        ImGui::Text(" [warn]");
        ImGui::SameLine();
        break;
      case spdlog::level::err:
        ImGui::PushStyleColor(ImGuiCol_Text,
                              ImVec4(1.0f, 0.0f, 0.0f, 1.0f)); // red
        ImGui::Text("[error]");
        ImGui::SameLine();
        break;
      case spdlog::level::debug:
        ImGui::PushStyleColor(ImGuiCol_Text,
                              ImVec4(0.0f, 1.0f, 0.0f, 1.0f)); // Red
        ImGui::Text("[debug]");
        ImGui::SameLine();
        break;
      case spdlog::level::critical:
        ImGui::PushStyleColor(ImGuiCol_Text,
                              ImVec4(1.0f, 0.0f, 1.0f, 1.0f)); // Red
        ImGui::Text(" [crit]");
        ImGui::SameLine();
        break;
      default: break;
      }
      ImGui::TextUnformatted(message.c_str());
      ImGui::PopStyleColor();
      ImGui::SetScrollHereY();
    }
    ImGui::End();
    ImGui::PopFont();
    ImGui::PopStyleVar();
  }
}

void PointCaster::draw_modeline() {
  constexpr auto modeline_height = 20;
  constexpr auto modeline_color = catpuccin::imgui::mocha_crust;

  ImGui::PushID("modeline");

  ImGui::PushStyleColor(ImGuiCol_WindowBg, modeline_color);
  ImGui::PushStyleColor(ImGuiCol_Border, modeline_color);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});

  const auto viewport_size = ImGui::GetMainViewport()->Size;
  const ImVec2 modeline_size{viewport_size.x, modeline_height};
  const ImVec2 modeline_min{0, viewport_size.y - modeline_size.y};
  const ImVec2 modeline_max{viewport_size.x, viewport_size.y};

  ImGui::SetNextWindowPos(modeline_min);
  ImGui::SetNextWindowSize(modeline_size);

  auto window_flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings;
  if (_current_mode != Mode::Find) window_flags |= ImGuiWindowFlags_NoInputs;

  ImGui::Begin("modeline", nullptr, window_flags);

  if (_current_mode == Mode::Find) {
    ImGui::Text("/");
    ImGui::SameLine();
    ImGui::SetKeyboardFocusHere();

    if (ImGui::InputText("##modeline.find", _modeline_input.data(),
                         modeline_buffer_size)) {};
  }

  ImGui::End();

  ImGui::PopStyleVar();
  ImGui::PopStyleVar();

  ImGui::PopStyleColor();
  ImGui::PopStyleColor();

  ImGui::PopID();
  ;
}

void PointCaster::find_mode_keypress(KeyEvent &event) {
  const auto &key = event.key();
  if (key == KeyEvent::Key::Enter) {
    _current_mode = Mode::NavigateMatch;
    _modeline_input.fill({});
  } else if (key == KeyEvent::Key::Esc) {
    _current_mode = Mode::Normal;
    _modeline_input.fill({});
  } else if (_imgui_context.handleKeyPressEvent(event)) {
    event.setAccepted(true);
  }
}

void PointCaster::draw_stats(const float delta_time) {
  ImGui::PushID("FrameStats");
  ImGui::SetNextWindowPos({50.0f, 200.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({200.0f, 100.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Frame Stats", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);

  // calculate the mean, min and max frame times from our last 60 frames
  frame_durations.push_back(delta_time);
  constexpr auto frames_to_average = 60 * 2; // 2 seconds
  if (frame_durations.size() < frames_to_average) {
    ImGui::Text("Gathering data...");
  } else {
    frame_durations.erase(frame_durations.begin()); // pop_front
    const float avg_duration =
        std::reduce(frame_durations.begin(), frame_durations.end()) /
        frame_durations.size();
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

    for (size_t i = 0; i < workspace.sessions.size(); i++) {
      auto& session = workspace.sessions[i];
      auto &camera_controller = camera_controllers[i];
      ImGui::PushID(session.id.c_str());
      if (ImGui::CollapsingHeader(session.label.c_str())) {
        if (camera_controller->config().analysis.enabled) {
          ImGui::Text("Analysis Duration");
          ImGui::BeginTable("analysis_duration", 2);
          ImGui::TableNextColumn();
          ImGui::Text("Current");
          ImGui::TableNextColumn();
          auto duration = camera_controller->analysis_time();
          ImGui::Text("%ims", duration);
          ImGui::EndTable();
          ImGui::Text("%.0f FPS", 1000.0f / duration);
        }
      }
      ImGui::PopID();
    }
  }

  ImGui::PopItemWidth();
  ImGui::End();
  ImGui::PopID();
}

auto output_count = 0;

void PointCaster::drawEvent() {

  pc::devices::reset_pointcloud_frames();

  const auto delta_time = _timeline.previousFrameDuration();
  const auto delta_ms = static_cast<int>(delta_time * 1000);
  TweenManager::instance()->tick(delta_ms);

  std::function<void()> main_thread_callback;
  while (MainThreadDispatcher::try_dequeue(main_thread_callback)) {
    main_thread_callback();
  }

  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
                               GL::FramebufferClear::Depth);

  render_cameras();

  _imgui_context.newFrame();
  ImGuizmo::SetOrthographic(false);
  ImGuizmo::BeginFrame();
  pc::gui::begin_gui_helpers(_current_mode, _modeline_input);

  // Enable text input, if needed/
  if (ImGui::GetIO().WantTextInput && !isTextInputActive()) startTextInput();
  else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
    stopTextInput();

  // Draw gui windows

  gui::draw_main_viewport(*this);

  if (!workspace.layout.hide_ui) {
    if (workspace.layout.show_devices_window) {
      pc::gui::draw_devices_window(*this);
    }

    if (workspace.layout.show_camera_window) {
      camera_controllers.at(workspace.selected_session_index)
          ->draw_imgui_controls();
    }

    if (workspace.layout.show_stats) draw_stats(delta_time);
    if (workspace.layout.show_radio_window) _radio->draw_imgui_window();
    if (workspace.layout.show_snapshots_window)
      _snapshots_context->draw_imgui_window();

    // clear gizmos e.g. bounding boxes, voxels
    for (auto &session_operator_host : session_operator_hosts) {
      session_operator_host.clear_gizmos();
    }
    // render the selected session's gizmos
    auto &selected_session_operator_host =
        session_operator_hosts.at(workspace.selected_session_index);
    if (workspace.layout.show_global_transform_window) {
      selected_session_operator_host.draw_imgui_window();
    }
    selected_session_operator_host.draw_gizmos();

    if (workspace.layout.show_session_operator_graph_window)
      _session_operator_graph->draw();

    if (workspace.mqtt.has_value() && (*workspace.mqtt).show_window)
      _mqtt->draw_imgui_window();

    if (workspace.midi.has_value() && (*workspace.midi).show_window) {
      ImGui::SetNextWindowSize({600, 400}, ImGuiCond_FirstUseEver);
      ImGui::Begin("MIDI");
      _midi->draw_imgui_window();
      ImGui::End();
    }

#ifdef WITH_OSC
    if (workspace.osc_client.has_value() && (*workspace.osc_client).show_window)
      _osc_client->draw_imgui_window();

    if (workspace.osc_server.has_value() && (*workspace.osc_server).show_window)
      _osc_server->draw_imgui_window();
#endif

    // draw_modeline();
  }

  _imgui_context.updateApplicationCursor(*this);

  // Render ImGui window
  GL::Renderer::enable(GL::Renderer::Feature::Blending);
  GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
  GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

  _imgui_context.drawFrame();

  if (ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }

  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);

  // TODO this can be removed and if we want GL errors we can set
  // MAGNUM_GPU_VALIDATION=ON instead or run the application with
  // --magnum-gpu-validation on
  auto error = GL::Renderer::error();
  if (error == GL::Renderer::Error::InvalidFramebufferOperation)
    pc::logger->warn("InvalidFramebufferOperation");
  if (error == GL::Renderer::Error::InvalidOperation)
    pc::logger->warn("InvalidOperation");
  if (error == GL::Renderer::Error::InvalidValue)
    pc::logger->warn("InvalidValue");
  if (error == GL::Renderer::Error::StackOverflow)
    pc::logger->warn("StackOverflow");
  if (error == GL::Renderer::Error::StackUnderflow)
    pc::logger->warn("StackUnderflow");

  redraw();

  swapBuffers();

  auto delta_secs = static_cast<float>(_timeline.previousFrameDuration());
  {
    // TODO make this some thing inside the PlyPlayer namespace maybe
    // or some more abstract class or function that handles all things that need
    // a tick each frame
    std::lock_guard lock(PlySequencePlayer::devices_access);
    for (auto &player : PlySequencePlayer::attached_devices) {
      player.get().tick(delta_secs);
    }
  }

  parameters::publish(delta_secs);

  _timeline.nextFrame();
  FrameMark;
}

void PointCaster::set_full_screen(bool full_screen) {
  if (full_screen && _display_resolution.has_value()) {
    pc::logger->debug("going full screen");
    if (!_full_screen) {
      _restore_window_size = windowSize() / dpiScaling();
      SDL_GetWindowPosition(window(), &_restore_window_position.x(),
                            &_restore_window_position.y());
    }
    setWindowSize(_display_resolution.value() / dpiScaling());
    SDL_SetWindowPosition(window(), 0, 0);
    _full_screen = true;
  } else if (!full_screen) {
    pc::logger->debug("restoring out");
    setWindowSize(_restore_window_size);
    SDL_SetWindowPosition(window(), _restore_window_position.x(),
                          _restore_window_position.y());
    _full_screen = false;
  }
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

  if (ImGui::GetIO().WantTextInput) {
    if (_imgui_context.handleKeyPressEvent(event)) event.setAccepted(true);
    return;
  }

  if (_current_mode == Mode::Find) {
    find_mode_keypress(event);
    if (_imgui_context.handleKeyPressEvent(event)) event.setAccepted(true);
    return;
  }
  if (_current_mode != Mode::Normal && event.key() == KeyEvent::Key::Esc) {
    _current_mode = Mode::Normal;
    return;
  } else if (_current_mode == Mode::Normal &&
             event.key() == KeyEvent::Key::Slash) {
    _current_mode = Mode::Find;
    return;
  }

  switch (event.key()) {
  case KeyEvent::Key::C: {
    workspace.layout.show_camera_window = !workspace.layout.show_camera_window;
    break;
  }
  case KeyEvent::Key::D: {
    workspace.layout.show_devices_window = !workspace.layout.show_devices_window;
    break;
  }
  case KeyEvent::Key::F: {
    set_full_screen(!_full_screen);
    break;
  }
  case KeyEvent::Key::G: {
    if (event.modifiers() == InputEvent::Modifier::Shift) {
      //workspace.layout.hide_ui = !workspace.layout.hide_ui;
      workspace.layout.show_session_operator_graph_window =
          !workspace.layout.show_session_operator_graph_window;
    } else {
      workspace.layout.show_global_transform_window =
          !workspace.layout.show_global_transform_window;
    }
    break;
  }
  case KeyEvent::Key::M: {
    if (event.modifiers() == InputEvent::Modifier::Shift) {
      if (workspace.mqtt.has_value()) {
        auto &mqtt_conf = workspace.mqtt.value();
        mqtt_conf.show_window = !mqtt_conf.show_window;
      }
    } else {
      if (workspace.midi.has_value()) {
        auto &midi_conf = workspace.midi.value();
        midi_conf.show_window = !midi_conf.show_window;
      }
    }
    break;
  }
#ifdef WITH_OSC
  case KeyEvent::Key::O: {
    if (event.modifiers() == InputEvent::Modifier::Shift) {
      if (workspace.osc_client.has_value()) {
        auto &osc_client_conf = workspace.osc_client.value();
        osc_client_conf.show_window = !osc_client_conf.show_window;
      }
    } else {
      if (workspace.osc_server.has_value()) {
        auto &osc_server_conf = workspace.osc_server.value();
        osc_server_conf.show_window = !osc_server_conf.show_window;
      }
    }
    break;
  }
#endif
  case KeyEvent::Key::Q: {
    quit();
    break;
  }
  case KeyEvent::Key::R: {
    workspace.layout.show_radio_window = !workspace.layout.show_radio_window;
    break;
  }
  case KeyEvent::Key::S: {
    if (event.modifiers() == InputEvent::Modifier::Shift) {
      workspace.layout.show_snapshots_window =
          !workspace.layout.show_snapshots_window;
    } else {
      save_workspace();
    }
    break;
  }
  case KeyEvent::Key::T: {
    workspace.layout.show_stats = !workspace.layout.show_stats;
    break;
  }
  default: {
    if (_imgui_context.handleKeyPressEvent(event)) event.setAccepted(true);
  }
  }
}

void PointCaster::keyReleaseEvent(KeyEvent &event) {
  if (_imgui_context.handleKeyReleaseEvent(event)) {
    event.setAccepted(true);
    return;
  }
}

void PointCaster::textInputEvent(TextInputEvent &event) {
  if (_imgui_context.handleTextInputEvent(event)) event.setAccepted(true);
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
  if (_imgui_context.handleMouseMoveEvent(event) &&
      !interacting_camera_controller) {
    event.setAccepted(true);
    return;
  }

  if (!interacting_camera_controller) {
    event.setAccepted(true);
    return;
  }

  if (ImGuizmo::IsUsing()) {
    event.setAccepted(true);
    return;
  }

  auto &camera_controller = interacting_camera_controller->get();

  // rotate / orbit
  if (event.buttons() == MouseMoveEvent::Button::Left) {
    camera_controller.mouse_orbit(event);
  }
  // translate
  else if (event.buttons() == MouseMoveEvent::Button::Right) {
    auto lock_y_axis = event.modifiers() == InputEvent::Modifier::Shift;
    camera_controller.mouse_translate(event, lock_y_axis);
  }

  event.setAccepted();
}

void PointCaster::mouseScrollEvent(MouseScrollEvent &event) {
  if (_imgui_context.handleMouseScrollEvent(event) &&
      !interacting_camera_controller) {
    /* Prevent scrolling the page */
    event.setAccepted(true);
    return;
  }

  if (!interacting_camera_controller) return;
  auto &camera_controller = interacting_camera_controller->get();

  const Magnum::Float delta = event.offset().y();
  if (Math::abs(delta) < 1.0e-2f) return;

  camera_controller.dolly(event);
}

} // namespace pc

MAGNUM_APPLICATION_MAIN(pc::PointCaster);
