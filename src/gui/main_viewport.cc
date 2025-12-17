#include "main_viewport.h"
#include "widgets.h"
#include "windows.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ImGuizmo.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Vector.h>
#include <cmath>
#include <imgui.h>
#include <regex>

namespace pc::gui {

void draw_main_viewport(PointCaster &app) {

  static std::optional<std::string> remove_session_id;
  static ImVec2 remove_session_pos;
  static std::string remove_session_window_id;
  static std::optional<int> rename_session_index;
  static std::optional<int> new_session_index;
  static std::optional<int> duplicate_session_index;
  static ImVec2 rename_tab_pos;
  static std::string rename_edit_buffer;

  constexpr float tab_bar_height = 20;
  static ImVec2 tab_bar_padding = {10.0f, 5.0f};
  ImGuiViewport *vp = ImGui::GetMainViewport();

  auto &workspace = app.workspace;

  // draw the tab bar at the top which can't be docked into and isn't part of
  // the workspace
  ImGui::SetNextWindowPos({vp->Pos.x, vp->Pos.y});
  ImGui::SetNextWindowSize({vp->Size.x, tab_bar_height});
  ImGui::SetNextWindowViewport(vp->ID);
  ImGuiWindowFlags tabbar_flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_NoDocking;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 5});
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, tab_bar_padding);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::Begin("##TabBar", nullptr, tabbar_flags);
  if (ImGui::BeginTabBar("Sessions", ImGuiTabBarFlags_Reorderable)) {

    auto &style = ImGui::GetStyle();
    ImDrawList *dl = ImGui::GetWindowDrawList();

    // app/file menu button at the start of the tab bar
    if (gui::draw_icon_tab_button(ICON_FA_BARS)) { gui::show_app_menu(); }

    // add new session button at the end of the tab bar
    if (gui::draw_icon_tab_button(ICON_FA_CIRCLE_PLUS,
                                  ImGuiTabItemFlags_Trailing, {-7, 0})) {
      ImGui::OpenPopup("AddSessionPopup");
    }

    gui::draw_app_menu(app);

    gui::push_context_menu_styles();
    if (ImGui::BeginPopup("AddSessionPopup",
                          ImGuiWindowFlags_AlwaysAutoResize)) {
      if (ImGui::MenuItem("Add new session to workspace")) {
        new_session_index = workspace.sessions.size();
        auto new_label_num = *new_session_index;
        auto new_label = std::format("session_{}", new_label_num);
        while (std::ranges::any_of(workspace.sessions, [&](auto &session) {
          return session.label == new_label;
        })) {
          new_label = std::format("session_{}", ++new_label_num);
        }
        workspace.sessions.push_back({.id = uuid::word(), .label = new_label});
        app.sync_session_instances();
        workspace.selected_session_index = workspace.sessions.size() - 1;
      }
      if (ImGui::MenuItem("Load session from disk")) {
        pc::logger->info("Load?");
      }
      ImGui::EndPopup();
    }
    gui::pop_context_menu_styles();

    // tab items for each open session

    constexpr auto tab_width = 125;

    for (int i = 0; i < workspace.sessions.size(); i++) {
      auto &session = workspace.sessions[i];
      ImGuiTabItemFlags item_flags = ImGuiTabItemFlags_None;
      if (new_session_index && i == *new_session_index) {
        item_flags |= ImGuiTabItemFlags_SetSelected;
        new_session_index.reset();
      }

      ImGui::SetNextItemWidth(tab_width);
      auto tab_name = std::format("{}###{}", session.label, session.id);
      if (ImGui::BeginTabItem(tab_name.c_str(), nullptr, item_flags)) {
        workspace.selected_session_index = i;
        ImGui::EndTabItem();
      }

      bool wants_to_rename = false;

      gui::push_context_menu_styles();
      if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Rename")) { wants_to_rename = true; }
        if (ImGui::MenuItem("Duplicate")) { duplicate_session_index = i; }
        ImGui::EndPopup();
      }
      gui::pop_context_menu_styles();

      // rename a session on a double click of its tab
      if (wants_to_rename ||
          (ImGui::IsItemHovered() &&
           ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) {
        rename_session_index = i;
        rename_tab_pos = ImGui::GetItemRectMin();
        rename_edit_buffer = session.label;
        ImGui::OpenPopup("RenameSession");
      }

      auto *session_tab_bar = ImGui::GetCurrentTabBar();

      // calculate close button positions and ids for each tab
      struct CloseButton {
        ImVec2 pos;
        ImGuiID tab_id;
      };
      std::vector<CloseButton> close_buttons;
      close_buttons.resize(session_tab_bar->Tabs.size());
      // iterate the tab count minus a few due to the
      // buttons that are also "tabs" in the bar
      for (int i = 0; i < session_tab_bar->Tabs.size() - 2; i++) {
        auto tab_index = i + 1;
        float close_button_x =
            tab_index * (tab_width + tab_bar_padding.x / 2) - 3;
        float close_button_y = tab_bar_padding.y + 1;
        const auto &tab = session_tab_bar->Tabs[tab_index];
        close_buttons[i] = {{close_button_x, close_button_y}, tab.ID};
      }

      // draw close buttons over top of our tab bar
      // and handle setting up removal of session when clicked
      for (const auto &button : close_buttons) {
        // draw a close session button on the right hand side of each tab
        if (button.pos.x == 0) continue;
        ImGui::SetCursorScreenPos(button.pos);
        if (gui::draw_icon_button(
                ICON_FA_XMARK, false,
                catpuccin::with_alpha(catpuccin::imgui::mocha_text, 0.25f),
                catpuccin::with_alpha(catpuccin::imgui::mocha_text, 0.5f))) {

          for (int i = 0; i < workspace.sessions.size(); ++i) {
            const auto &session = workspace.sessions[i];
            auto tab_name = std::format("{}###{}", session.label, session.id);
            if (ImGui::GetID(tab_name.c_str()) == button.tab_id) {
              pc::logger->debug("Should remove: {}", session.label);
              remove_session_id = session.id;
              remove_session_pos = {button.pos.x - 30, button.pos.y + 30};
              remove_session_window_id =
                  std::format("Delete Session?###{}", uuid::digit());
              ImGui::OpenPopup(remove_session_window_id.c_str());
              break;
            }
          }
        }
      }

      // handle reordering of sessions by dragging tabs around
      // TODO this is buggy and maybe doesn't serialize
      if (session_tab_bar->ReorderRequestTabId) {
        int source_index = -1;
        for (int i = 0; i < workspace.sessions.size(); i++) {
          const auto &session = workspace.sessions[i];
          auto tab_name = std::format("{}###{}", session.label, session.id);
          if (ImGui::GetID(tab_name.c_str()) ==
              session_tab_bar->ReorderRequestTabId) {
            source_index = i;
            break;
          }
        }
        int destination_index =
            source_index + session_tab_bar->ReorderRequestOffset;
        if (source_index >= 0 && destination_index >= 0 &&
            destination_index < workspace.sessions.size()) {
          std::swap(workspace.sessions[source_index],
                    workspace.sessions[destination_index]);
        }
      }

      // duplicate a session
      if (duplicate_session_index) {
        const auto &original_session =
            workspace.sessions[*duplicate_session_index];
        const std::string &original_label = original_session.label;
        // duplicate the target session
        workspace.sessions.push_back(original_session);
        auto &new_session = workspace.sessions.back();
        // give it and its camera a unique id
        new_session.id = uuid::word();
        new_session.camera.id = uuid::word();
        // copy operators and give them unique ids
        auto& new_operators = new_session.session_operator_host.operators;
        new_operators = original_session.session_operator_host.operators;
        for (auto &operator_variant : new_operators) {
          std::visit([](auto &&op) { op.id = uuid::digit(); },
                     operator_variant);
        }
        // while the new label is equal to any existing session label...
        std::smatch match;
        while (
            std::ranges::any_of(workspace.sessions, [&](const auto &session) {
              return (&session != &new_session) &&
                     (session.label == new_session.label);
            })) {
          // match a suffix number if it exists and increment it
          static const std::regex trailing_number_regex(R"(^(.*?)(\d+)$)");
          if (std::regex_match(new_session.label, match,
                               trailing_number_regex)) {
            const std::string prefix = match[1].str();
            const int suffix_value = std::stoi(match[2].str());
            new_session.label = std::format("{}{}", prefix, suffix_value + 1);
          } else {
            // otherwise append a suffix number
            new_session.label = std::format("{}_1", new_session.label);
          }
        }
        // sync the app state and focus the duplicate
        app.sync_session_instances();
        workspace.selected_session_index = workspace.sessions.size() - 1;
        duplicate_session_index.reset();
      }
    }

    // draw popup windows & modals

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10, 5});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {10, 5});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {10, 5});
    ImGui::SetNextWindowPos(remove_session_pos);
    if (ImGui::BeginPopupModal(remove_session_window_id.c_str(), nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize) &&
        remove_session_id) {
      auto target_session_it = std::ranges::find(
          workspace.sessions, remove_session_id, &PointcasterSession::id);
      auto &target_session = *target_session_it;
      ImGui::Dummy({0, 5});
      ImGui::Text("%s",
                  std::format("Really delete '{}' from the current workspace?",
                              target_session.label)
                      .c_str());
      ImGui::Dummy({0, 10});

      if (ImGui::Button("Delete", ImVec2(80, 0))) {
        int removed_index =
            std::distance(workspace.sessions.begin(), target_session_it);
        workspace.sessions.erase(target_session_it);

        if (workspace.sessions.empty()) {
          workspace.selected_session_index = 0;
        } else if (removed_index == workspace.selected_session_index) {
          if (workspace.selected_session_index >=
              int(workspace.sessions.size()))
            workspace.selected_session_index =
                int(workspace.sessions.size()) - 1;
        } else if (removed_index < workspace.selected_session_index) {
          workspace.selected_session_index--;
        }
        remove_session_id.reset();
        rename_session_index.reset();
        duplicate_session_index.reset();
        new_session_index.reset();
        ImGui::CloseCurrentPopup();
      }

      ImGui::SameLine();

      if (ImGui::Button("Cancel", ImVec2(80, 0))) {
        ImGui::CloseCurrentPopup();
      }
      ImGui::Dummy({0, 5});

      ImGui::EndPopup();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();

    ImGui::SetNextWindowPos(rename_tab_pos, ImGuiCond_Appearing);
    ImGui::SetNextWindowSize({tab_width, -FLT_MIN});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    if (ImGui::BeginPopup("RenameSession", ImGuiWindowFlags_NoDecoration) &&
        rename_session_index) {
      auto &target_session = workspace.sessions.at(*rename_session_index);
      auto rename_input_label =
          fmt::format("##rename_session_{}", *rename_session_index);
      if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
      ImGui::PushItemWidth(tab_width);
      if (ImGui::InputText(rename_input_label.c_str(), &rename_edit_buffer,
                           ImGuiInputTextFlags_AutoSelectAll |
                               ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (std::ranges::any_of(workspace.sessions, [&](auto &session) {
              return session.label == rename_edit_buffer;
            })) {
          // can't name two sessions the same name
          rename_edit_buffer.clear();
        }
        target_session.label = rename_edit_buffer.empty() ? target_session.label
                                                          : rename_edit_buffer;
        session_label_from_id[target_session.id] = target_session.label;
        ImGui::CloseCurrentPopup();
        new_session_index = *rename_session_index;
        rename_session_index.reset();
        gui::refresh_tooltips = true;
      }
      ImGui::EndPopup();
    }
    ImGui::PopStyleVar();

    ImGui::EndTabBar(); // session tab bar
  }

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();

  // main docking workspace below the tab bar
  ImVec2 session_viewport_pos{vp->Pos.x, vp->Pos.y + tab_bar_height};
  ImGui::SetNextWindowPos(session_viewport_pos);

  ImVec2 session_viewport_size{vp->Size.x, vp->Size.y - tab_bar_height};
  ImGui::SetNextWindowSize(session_viewport_size);

  ImGui::SetNextWindowViewport(vp->ID);
  ImGuiWindowFlags host_flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoBringToFrontOnFocus;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::Begin("##DockSpaceHost", nullptr, host_flags);
  ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
  ImGui::DockSpace(dockspace_id, {0, 0},
                   ImGuiDockNodeFlags_PassthruCentralNode |
                       ImGuiDockNodeFlags_NoDockingInCentralNode);
  ImGui::End();
  ImGui::PopStyleVar();

  // draw the central content inside the workspace - the selected session's
  // camera controller output frames
  ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_Always);
  ImGuiWindowClass wc = {};
  wc.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::SetNextWindowClass(&wc);
  ImGui::Begin("CentralContent", nullptr,
               ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);
  {
    auto &selected_session =
        workspace.sessions.at(workspace.selected_session_index);
    auto &selected_camera_controller =
        app.camera_controllers.at(workspace.selected_session_index);
    const auto camera_config = selected_camera_controller->config();

    const auto window_size = ImGui::GetContentRegionAvail();

    if (workspace.layout.hide_ui) { ImGui::SetCursorPosY(0); }

    const auto draw_frame_labels =
        [&selected_camera_controller](ImVec2 viewport_offset) {
          ImGui::PushStyleColor(ImGuiCol_Text, {1.0f, 1.0f, 0.0f, 1.0f});
          for (auto label : selected_camera_controller->labels()) {
            auto &pos = label.position;
            auto x = static_cast<float>(viewport_offset.x + pos.x);
            auto y = static_cast<float>(viewport_offset.y + pos.y);
            ImGui::SetCursorPos({x, y});
            ImGui::Text("%s", label.text.data());
          }
          ImGui::PopStyleColor();
        };

    const auto draw_imguizmo_layer = [&](auto layer_pos_x, auto layer_pos_y,
                                         auto layer_size_x, auto layer_size_y) {
      ImGuizmo::SetDrawlist();
      ImGuizmo::SetRect(layer_pos_x, layer_pos_y, layer_size_x, layer_size_y);

      auto &camera = selected_camera_controller->camera();
      std::array<float, 16> camera_view_matrix;
      std::memcpy(camera_view_matrix.data(), camera.cameraMatrix().data(),
                  sizeof(camera_view_matrix));
      std::array<float, 16> camera_projection_matrix;
      std::memcpy(camera_projection_matrix.data(),
                  camera.projectionMatrix().data(),
                  sizeof(camera_projection_matrix));

      if (!devices::selected_device_config) return;

      using namespace Magnum::Math::Literals;
      using Magnum::Matrix4;
      using Magnum::Vector3;
      using Magnum::Math::Quaternion;
      using Magnum::Math::Rad;
      using Magnum::Math::Deg;

      Matrix4 object_model_matrix;
      std::visit(
          [&](auto &&config_variant) {
            auto &t = config_variant.transform;
            Quaternion orientation_quaternion{
                {t.rotation[0], t.rotation[1], t.rotation[2]}, t.rotation[3]};
            auto rotation_matrix_3x3 = orientation_quaternion.toMatrix();
            object_model_matrix =
                Matrix4::translation(
                    Vector3{t.offset.x, t.offset.y, t.offset.z} / 1000.0f) *
                Matrix4::from(rotation_matrix_3x3, {}) *
                Matrix4::scaling(Vector3{t.scale});
          },
          devices::selected_device_config->get());

      std::array<float, 16> object_model_matrix_flat;
      std::memcpy(object_model_matrix_flat.data(), object_model_matrix.data(),
                  sizeof(object_model_matrix_flat));

      ImGuizmo::Enable(true);

      std::array<float, 16> delta_rotation_matrix_array;
      std::array<float, 16> delta_translation_matrix_array;

      ImGuizmo::Manipulate(
          camera_view_matrix.data(), camera_projection_matrix.data(),
          ImGuizmo::ROTATE, ImGuizmo::MODE::WORLD,
          object_model_matrix_flat.data(), delta_rotation_matrix_array.data());

      ImGuizmo::Manipulate(camera_view_matrix.data(),
                           camera_projection_matrix.data(), ImGuizmo::TRANSLATE,
                           ImGuizmo::MODE::WORLD,
                           object_model_matrix_flat.data(),
                           delta_translation_matrix_array.data());

      const ImVec2 view_manipulate_gizmo_size { 128, 128 };
      const ImVec2 view_manipulate_gizmo_padding{0, 10};
      const ImVec2 view_manipulate_gizmo_position = {
          layer_pos_x + layer_size_x - view_manipulate_gizmo_size.x -
              view_manipulate_gizmo_padding.x,
          layer_pos_y + view_manipulate_gizmo_padding.y};
      auto camera_distance =
          selected_camera_controller->config().transform.distance;

      const auto pre_view = camera_view_matrix;

      ImGuizmo::ViewManipulate(camera_view_matrix.data(), camera_distance,
                               view_manipulate_gizmo_position,
                               view_manipulate_gizmo_size, 0x10101010);

      // handle updates from the view manipulate gizmo
      if (pre_view != camera_view_matrix) {
        Eigen::Matrix4f view;
        std::memcpy(view.data(), camera_view_matrix.data(), sizeof(view));

        const Eigen::Matrix4f world = view.inverse();
        const Eigen::Vector3f new_translation = world.block<3, 1>(0, 3) / 1000.0f;
        const Eigen::Matrix3f new_rotation = world.block<3, 3>(0, 0);
        const Eigen::Vector3f new_euler = new_rotation.eulerAngles(1, 0, 2);

        constexpr auto rad2deg = 180.0f / static_cast<float>(M_PI);
        const float new_yaw = new_euler[0] * rad2deg;
        const float new_pitch = -new_euler[1] * rad2deg;
        const float new_roll = new_euler[2] * rad2deg;

        auto& selected_camera = *selected_camera_controller;
        selected_camera.set_orbit({ new_yaw, new_pitch });
        selected_camera.set_roll(new_roll);
        selected_camera.set_translation(
            {new_translation.x(), new_translation.y(), new_translation.z()});
        // selected_camera.set_distance(new_translation.z());
      }
      // handle updates from the selected device transform gizmo
      if (ImGuizmo::IsUsing()) {
        Magnum::Matrix4 rotation_delta_matrix =
            Magnum::Matrix4::from(delta_rotation_matrix_array.data());
        Magnum::Matrix4 translation_delta_matrix =
            Magnum::Matrix4::from(delta_translation_matrix_array.data());

        auto rotation_delta_3x3 = rotation_delta_matrix.rotationNormalized();
        auto rotation_delta_quaternion =
            Quaternion<float>::fromMatrix(rotation_delta_3x3).normalized();

        auto translation_delta_vector = translation_delta_matrix.translation();
        float uniform_scale_delta = rotation_delta_matrix.uniformScaling();

        std::lock_guard lock(devices::device_configs_access);
        std::visit(
            [&](auto &&config_variant) {
              auto &transform = config_variant.transform;
              Quaternion<float> current_quaternion{{transform.rotation[0],
                                                    transform.rotation[1],
                                                    transform.rotation[2]},
                                                   transform.rotation[3]};
              auto updated_quaternion =
                  (rotation_delta_quaternion * current_quaternion).normalized();
              transform.rotation[0] = updated_quaternion.vector().x();
              transform.rotation[1] = updated_quaternion.vector().y();
              transform.rotation[2] = updated_quaternion.vector().z();
              transform.rotation[3] = updated_quaternion.scalar();
              transform.offset =
                  transform.offset + (Float3{translation_delta_vector[0],
                                             translation_delta_vector[1],
                                             translation_delta_vector[2]} *
                                      1000.0f);
              transform.scale *= uniform_scale_delta;
              // also update euler values
              auto euler_rad = updated_quaternion.toEuler();
              Deg<float> deg_x{euler_rad.x()};
              Deg<float> deg_y{euler_rad.y()};
              Deg<float> deg_z{euler_rad.z()};
              transform.rotation_deg[0] = static_cast<float>(deg_x);
              transform.rotation_deg[1] = static_cast<float>(deg_y);
              transform.rotation_deg[2] = static_cast<float>(deg_z);
            },
            devices::selected_device_config->get());
      }
    };

    auto rendering = camera_config.rendering;
    auto scale_mode = rendering.scale_mode;
    const Vector2 frame_space{window_size.x, window_size.y};
    selected_camera_controller->viewport_size = frame_space;

    if (scale_mode == (int)ScaleMode::Span) {

      auto image_pos = ImGui::GetCursorPos();
      ImGuiIntegration::image(selected_camera_controller->color_frame(),
                              {frame_space.x(), frame_space.y()});

      draw_imguizmo_layer(image_pos.x, image_pos.y, frame_space.x(),
                          frame_space.y());

      auto analysis = camera_config.analysis;

      if (analysis.enabled && (analysis.contours.draw)) {
        ImGui::SetCursorPos(image_pos);
        ImGuiIntegration::image(selected_camera_controller->analysis_frame(),
                                {frame_space.x(), frame_space.y()});
        draw_frame_labels(image_pos);
      }

    } else if (scale_mode == (int)ScaleMode::Letterbox) {

      float width, height, horizontal_offset, vertical_offset;

      auto frame_aspect_ratio =
          rendering.resolution[0] / static_cast<float>(rendering.resolution[1]);
      auto space_aspect_ratio = frame_space.x() / frame_space.y();

      constexpr auto frame_spacing = 5.0f;

      if (frame_aspect_ratio > space_aspect_ratio) {
        width = frame_space.x();
        height = std::round(width / frame_aspect_ratio);
        horizontal_offset = 0.0f;
        vertical_offset =
            std::max(0.0f, (frame_space.y() - height) / 2.0f - frame_spacing);
      } else {
        height = frame_space.y() - 2 * frame_spacing;
        width = std::round(height * frame_aspect_ratio);
        vertical_offset = 0.0f;
        horizontal_offset = std::max(0.0f, (frame_space.x() - width) / 2.0f);
      }

      ImGui::Dummy({horizontal_offset, vertical_offset});
      if (horizontal_offset != 0.0f) ImGui::SameLine();

      constexpr auto border_size = 1.0f;
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, border_size);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});

      ImGui::BeginChildFrame(ImGui::GetID("letterboxed"), {width, height});

      auto image_pos = ImGui::GetCursorPos();
      ImGuiIntegration::image(selected_camera_controller->color_frame(),
                              {width, height});

      draw_imguizmo_layer(image_pos.x, image_pos.y, frame_space.x(),
                          frame_space.y());

      auto &analysis = selected_camera_controller->config().analysis;

      if (analysis.enabled && (analysis.contours.draw)) {
        ImGui::SetCursorPos(image_pos);
        ImGuiIntegration::image(selected_camera_controller->analysis_frame(),
                                {width, height});
        draw_frame_labels(image_pos);
      }

      ImGui::EndChildFrame();

      ImGui::PopStyleVar();
      ImGui::PopStyleVar();

      if (horizontal_offset != 0.0f) ImGui::SameLine();
      ImGui::Dummy({horizontal_offset, vertical_offset});
    }

    if (!workspace.layout.hide_ui) {
      // draw_viewport_controls(*camera_controller);
      if (workspace.layout.show_log) { app.draw_onscreen_log(); }
    }

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootWindow)) {
      app.interacting_camera_controller = *selected_camera_controller;
    } else {
      app.interacting_camera_controller.reset();
    }
  }
  ImGui::End();
  ImGui::PopStyleVar();
}
} // namespace pc::gui