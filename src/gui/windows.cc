#include "windows.h"
#include "catpuccin.h"
#include "widgets.h"
#include <imgui.h>
#include <Magnum/Math/Quaternion.h>

namespace pc::gui {

void draw_devices_window(PointCaster &app) {
  ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({300.0f, 600.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  const ImVec2 window_padding{10, 10};
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, window_padding);
  ImGui::Begin("Devices", nullptr);

  static unsigned int selected_device_row = 0;
  static int editing_device_row_id = -1;

  // this ensures newly added devices are 'selected'
  static int last_device_count = -1;
  {
    std::lock_guard lock(devices::device_configs_access);
    if (last_device_count == -1) {
      last_device_count = devices::device_configs.size();
    }
    if (last_device_count != devices::device_configs.size()) {
      selected_device_row = devices::device_configs.size() - 1;
    }
  }

  ImGui::PushFont(pc::gui::icon_font.get());
  const auto icon_width = ImGui::CalcTextSize(ICON_FA_CIRCLE).x +
                          ImGui::GetStyle().FramePadding.x * 2.0f;
  const auto row_height = ImGui::GetTextLineHeightWithSpacing() + 5;
  ImGui::PopFont();
  const auto window_width = ImGui::GetWindowWidth();

  // top row of devices window

  // includes the name of the selected device type
  if (last_device_count > 0) {
    ImGui::Dummy({2, 0});
    ImGui::SameLine();
    ImGui::BeginDisabled();
    {
      std::lock_guard lock(devices::device_configs_access);
      if (selected_device_row < devices::device_configs.size()) {
        std::visit(
            [](auto &&config) {
              using T = std::decay_t<decltype(config)>;
              ImGui::Text(T::Name);
            },
            devices::device_configs[selected_device_row].get());
      }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
  }

  // and a button to add a new device
  ImGui::SetCursorPosX(window_width - icon_width - window_padding.x);
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 3);
  if (gui::draw_icon_button(
          ICON_FA_CIRCLE_PLUS, false,
          catpuccin::with_alpha(catpuccin::imgui::mocha_green, 0.75f),
          catpuccin::imgui::mocha_green)) {
    ImGui::OpenPopup("AddDevicePopup");
  }
  if (ImGui::IsItemHovered()) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, tooltip_padding);
    ImGui::SetTooltip("Add Device");
    ImGui::PopStyleVar();
  }

  ImGui::Dummy({0, 5});

  // create the scrollable list of device names

  static constexpr size_t visible_device_rows = 6;
  std::optional<int> device_to_delete;
  std::optional<int> device_to_toggle_active;
  std::optional<int> device_to_toggle_session_active;
  std::optional<std::string> edited_device_id;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
  ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0f);
  if (ImGui::BeginChild("DevicesScrollable",
                        ImVec2(0, visible_device_rows * row_height), true)) {
    if (ImGui::BeginTable("DevicesTable", 4, ImGuiTableFlags_RowBg,
                          ImVec2(-FLT_MIN, visible_device_rows * row_height))) {
      ImGui::TableSetupColumn("active_button", ImGuiTableColumnFlags_WidthFixed,
                              icon_width + 5);
      ImGui::TableSetupColumn("session_active_button",
                              ImGuiTableColumnFlags_WidthFixed, icon_width + 2);
      ImGui::TableSetupColumn("device", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("delete_button", ImGuiTableColumnFlags_WidthFixed,
                              icon_width + 5);

      // get the table's total width from the available content region.
      float table_width = ImGui::GetContentRegionAvail().x;

      // TODO this device_configs and device access is not thread safe

      for (int i = 0; i < devices::device_configs.size(); i++) {
        std::string_view device_id;
        devices::DeviceStatus device_status{DeviceStatus::Missing};
        if (i < devices::attached_devices.size()) {
          if (auto *device = devices::attached_devices[i].get()) {
            device_status = device->status();
          }
        }
        bool is_selected = (i == selected_device_row);
        bool is_active;
        bool is_session_active;
        std::visit(
            [&](auto &&device_config) {
              device_id = device_config.id;
              is_active = device_config.active;
              is_session_active =
                  app.workspace.sessions[app.workspace.selected_session_index]
                      .active_devices[device_config.id];
            },
            devices::device_configs[i].get());

        ImGui::TableNextRow();

        size_t column_count = 0;

        // column 1: active button
        ImGui::TableSetColumnIndex(column_count++);
        ImVec2 active_button_content_min = ImGui::GetCursorPos();
        ImGui::SameLine();
        std::string_view active_icon;
        if (is_active) {
          ImGui::SetCursorPos(
              {active_button_content_min.x + 5, active_button_content_min.y + 1.0f});
          active_icon = ICON_FA_TOGGLE_ON;
        } else {
          ImGui::SetCursorPos(
              {active_button_content_min.x + 5, active_button_content_min.y + 1.0f});
          active_icon = ICON_FA_TOGGLE_OFF;
        }
        if (gui::draw_icon_button(active_icon.data(), false,
                                  catpuccin::imgui::mocha_overlay,
                                  catpuccin::imgui::mocha_overlay2)) {
          device_to_toggle_active = i;
        }
        if (ImGui::IsItemHovered()) {
          ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, tooltip_padding);
          ImGui::SetTooltip("Toggle device active");
          ImGui::PopStyleVar();
        }

        // column 2: session active button
        ImGui::TableSetColumnIndex(column_count++);
        ImVec2 session_active_button_content_min = ImGui::GetCursorPos();
        ImGui::SameLine();
        std::string_view session_active_icon;
        if (is_session_active) {
          ImGui::SetCursorPos({session_active_button_content_min.x - 1,
                               session_active_button_content_min.y + 1.0f});
          session_active_icon = ICON_FA_EYE;
        } else {
          ImGui::SetCursorPos({session_active_button_content_min.x - 2,
                               session_active_button_content_min.y + 1.0f});
          session_active_icon = ICON_FA_EYE_SLASH;
        }
        if (gui::draw_icon_button(session_active_icon.data(), false,
                                  catpuccin::imgui::mocha_overlay,
                                  catpuccin::imgui::mocha_overlay2)) {
          device_to_toggle_session_active = i;
        }
        if (ImGui::IsItemHovered()) {
          ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, tooltip_padding);
          ImGui::SetTooltip("Toggle device session visibility");
          ImGui::PopStyleVar();
        }

        if (!is_active || !is_session_active) ImGui::BeginDisabled();

        constexpr auto color_from_status = [](DeviceStatus status) -> ImVec4 {
          switch (status) {
          case DeviceStatus::Active: return catpuccin::imgui::mocha_green;
          case DeviceStatus::Inactive: return catpuccin::imgui::mocha_surface2;
          case DeviceStatus::Loaded: return catpuccin::imgui::mocha_blue;
          case DeviceStatus::Missing: return catpuccin::imgui::mocha_red;
          default: return catpuccin::imgui::mocha_text;
          }
        };
        auto status_color = color_from_status(device_status);

        // column 3: device id with status icon
        ImGui::TableSetColumnIndex(column_count++);
        ImVec2 device_id_cell_min = ImGui::GetCursorScreenPos();
        {
          // status icon, grouped for positioning
          ImGui::BeginGroup();
          ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
          ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 2);
          gui::draw_icon_button(ICON_FA_CIRCLE, true, status_color,
                                status_color);
          ImGui::EndGroup();
        }
        // device id
        ImGui::SameLine();
        ImGui::Dummy({2, 0});
        ImGui::SameLine();
        ImGui::TextUnformatted(device_id.data());

        // compute the width for the device cell: table width minus active,
        // extra and delete columns and spacing
        float device_id_cell_width = table_width -
                                     ((icon_width + 5) * (column_count - 1)) -
                                     ImGui::GetStyle().ItemSpacing.x;
        ImVec2 device_id_cell_max =
            ImVec2(device_id_cell_min.x + device_id_cell_width,
                   device_id_cell_min.y + row_height);

        if (editing_device_row_id == i) {
          auto popup_label = fmt::format("##device_edit_popup/{}", device_id);
          static std::string edit_buf_device;
          static bool device_edit_popup_opened = false;
          if (!device_edit_popup_opened) {
            edit_buf_device = std::string(device_id);
            device_edit_popup_opened = true;
          }
          ImGui::SetNextWindowPos(
              {device_id_cell_min.x - 2.0f, device_id_cell_min.y + 1.0f});
          ImGui::SetNextWindowSize({device_id_cell_width - 20.0f, row_height});
          ImGui::OpenPopup(popup_label.c_str());
          if (ImGui::BeginPopup(popup_label.c_str())) {
            // use &edit_buf_device so the std::string overload is used
            if (ImGui::InputText(fmt::format("##rename_device_{}", i).c_str(),
                                 &edit_buf_device,
                                 ImGuiInputTextFlags_AutoSelectAll |
                                     ImGuiInputTextFlags_EnterReturnsTrue)) {
              edited_device_id = edit_buf_device;
              if (edited_device_id.value().empty())
                edited_device_id = std::string(device_id);
              ImGui::CloseCurrentPopup();
              device_edit_popup_opened = false;
            }
            if (!ImGui::IsItemActive() && ImGui::IsMouseClicked(0)) {
              edited_device_id = std::string(device_id);
              ImGui::CloseCurrentPopup();
              device_edit_popup_opened = false;
            }
            ImGui::EndPopup();
          }
        }

        // extend the cell's rect across the entire row (for background)
        ImVec2 row_min = device_id_cell_min;
        ImVec2 row_max = ImVec2(device_id_cell_min.x + table_width,
                                device_id_cell_min.y + row_height);
        bool row_hovered = ImGui::IsMouseHoveringRect(row_min, row_max);

        ImU32 bg_color;
        if (is_selected || row_hovered) {
          bg_color = ImGui::GetColorU32(catpuccin::imgui::mocha_surface);
        } else {
          bg_color = ImGui::GetColorU32(catpuccin::imgui::mocha_base);
        }
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, bg_color);

        // detect clicks for selection using the full cell's rectangle
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
            ImGui::IsMouseHoveringRect(device_id_cell_min,
                                       device_id_cell_max)) {
          selected_device_row = i;
        }
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
            ImGui::IsMouseHoveringRect(device_id_cell_min,
                                       device_id_cell_max)) {
          editing_device_row_id = i;
        }

        if (!is_active || !is_session_active) ImGui::EndDisabled();

        // column 3: delete icon button
        ImGui::TableSetColumnIndex(column_count++);
        ImVec2 delete_cell_content_min = ImGui::GetCursorPos();
        ImGui::SameLine();
        ImGui::SetCursorPos(
            {delete_cell_content_min.x, delete_cell_content_min.y + 1.0f});
        if (gui::draw_icon_button(ICON_FA_CIRCLE_XMARK, false,
                                  catpuccin::imgui::mocha_overlay,
                                  catpuccin::imgui::mocha_overlay2)) {
          device_to_delete = i;
        }
        ImGui::SameLine();
        ImGui::Dummy({5, 0});
      }
      ImGui::EndTable();
    }
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();

  // after the list, process any modifications
  if (device_to_toggle_active.has_value()) {
    std::lock_guard lock(devices::device_configs_access);
    auto &device_config =
        devices::device_configs[device_to_toggle_active.value()];
    std::visit([](auto &&config) { config.active = !config.active; },
               device_config.get());
  }
  if (device_to_toggle_session_active.has_value()) {
    std::lock_guard lock(devices::device_configs_access);
    auto &device_config =
        devices::device_configs[device_to_toggle_session_active.value()];
    auto &selected_session =
        app.workspace.sessions[app.workspace.selected_session_index];
    auto &session_active_device_map = selected_session.active_devices;
    std::visit(
        [&](auto &&config) {
          auto &id = config.id;
          auto old_session_active_val = session_active_device_map.contains(id)
                                            ? session_active_device_map[id]
                                            : true;
          selected_session.active_devices[id] = !old_session_active_val;
        },
        device_config.get());
  }
  if (device_to_delete.has_value()) {
    std::lock_guard lock(devices::devices_access);
    if (device_to_delete < devices::attached_devices.size()) {
      devices::attached_devices.erase(attached_devices.begin() +
                                      device_to_delete.value());
      // we can't select a device row if it no longer exists
      selected_device_row = std::min(static_cast<size_t>(selected_device_row),
                                     devices::attached_devices.size() - 1);
    }
  }
  if (edited_device_id.has_value()) {
    std::lock_guard lock(devices::device_configs_access);
    if (editing_device_row_id < devices::device_configs.size()) {
      auto &device_config = devices::device_configs[editing_device_row_id];
      std::visit(
          [&](auto &&config) {
            parameters::unbind_parameters(config.id);
            config.id = edited_device_id.value();
            parameters::declare_parameters("devices", config.id, config);
          },
          device_config.get());
    }
    editing_device_row_id = -1;
    edited_device_id.reset();
  }

  ImGui::Dummy({0, 10});

  // the scrollable child that contains the device configuration parameters of
  // the selected device in the list

  if (!devices::device_configs.empty()) {
    auto &selected_device_config = devices::device_configs[selected_device_row];
    if (selected_device_row < devices::attached_devices.size()) {
      auto selected_device = devices::attached_devices[selected_device_row];
      std::visit(
          [&](auto &&device_config) {
            selected_device->draw_controls();
            ImGui::Dummy({0, 10});
            const auto available_space = ImGui::GetContentRegionAvail();
            ImGui::PushID("device_config");
            ImGui::BeginChild(device_config.id.c_str(),
                              {-FLT_MIN, available_space.y});
            auto previous_config = device_config;
            auto did_update_config = pc::gui::draw_parameters(device_config.id);
            if (did_update_config) {
              // if we updated the rotation via eulers, they need to get
              // converted into quaternion
              if (previous_config.transform.rotation_deg !=
                  device_config.transform.rotation_deg) {
                using Magnum::Math::Deg;
                using Magnum::Math::Quaternion;
                using Magnum::Math::Vector3;
                auto &transform = device_config.transform;
                float yaw = transform.rotation_deg[0];
                float pitch = transform.rotation_deg[1];
                float roll = transform.rotation_deg[2];

                // Eigen/ImGuizmo stored pitch as –x, so invert it here
                Quaternion<float> qy = Quaternion<float>::rotation(
                    Deg<float>(yaw), Vector3<float>::yAxis());
                Quaternion<float> qx = Quaternion<float>::rotation(
                    Deg<float>(-pitch), Vector3<float>::xAxis());
                Quaternion<float> qz = Quaternion<float>::rotation(
                    Deg<float>(roll), Vector3<float>::zAxis());

                Quaternion<float> quat_out = qy * qx * qz;
                transform.rotation[0] = quat_out.vector().x();
                transform.rotation[1] = quat_out.vector().y();
                transform.rotation[2] = quat_out.vector().z();
                transform.rotation[3] = quat_out.scalar();
              }
            }
            ImGui::EndChild();
            ImGui::PopID();
          },
          selected_device_config.get());
    }
  }

  // the popup for adding devices if the add device button is pressed
  gui::push_context_menu_styles();
  ImGui::SetNextWindowSize({180, 0});
  if (ImGui::BeginPopup("AddDevicePopup", ImGuiWindowFlags_AlwaysAutoResize)) {
    // USB devices
    ImGui::SeparatorText("USB Devices");
    if (ImGui::MenuItem("Azure Kinect")) { app.open_kinect_sensors(); }
    if (ImGui::MenuItem("Orbbec (USB)")) {
      pc::logger->error("Orbbec USB device is not yet implemented");
    }

    // Network devices
    ImGui::SeparatorText("Network Devices");

    // Start discovery once if needed
    static std::atomic_bool requested_orbbec_discovery = false;
    if (ImGui::IsWindowAppearing()) requested_orbbec_discovery = false;
    if (!requested_orbbec_discovery) {
      requested_orbbec_discovery = true;
      app.run_async([] {
        pc::logger->info("Starting Orbbec device discovery...");
        try {
          OrbbecDevice::discover_devices();
        } catch (const std::exception &e) {
          pc::logger->error("Orbbec discovery failed: {}", e.what());
        }
        pc::logger->info("Orbbec device discovery finished.");
      });
    }

    if (OrbbecDevice::discovering_devices) {
      ImGui::MenuItem("Searching for Orbbec devices...", nullptr, false, false);
    } else {
      size_t orbbec_list_size = 0;
      {
        for (auto &dev : OrbbecDevice::discovered_devices) {
          const bool already_connected = std::ranges::any_of(
              OrbbecDevice::attached_devices, [&](const auto &attached) {
                return attached.get().ip() == dev.ip;
              });

          if (!already_connected) {
            auto label = fmt::format("Orbbec - {}", dev.ip);
            if (ImGui::MenuItem(label.c_str())) {
              ImGui::CloseCurrentPopup();
              app.open_orbbec_sensor(dev.ip);
            }
            orbbec_list_size++;
          }
        }
      }
      if (OrbbecDevice::discovered_devices.empty() || orbbec_list_size == 0) {
        ImGui::MenuItem("None", nullptr, false, false);
      }
    }

    // Sequences
    ImGui::SeparatorText("Files");
    if (ImGui::MenuItem("PLY Sequence")) {
      app.run_async([&] { app.open_ply_sequence(); });
    }

    ImGui::EndPopup();
  }
  gui::pop_context_menu_styles();

  {
    std::lock_guard lock(devices::device_configs_access);
    last_device_count = devices::device_configs.size();
    if (last_device_count > 0) {
      devices::selected_device_config =
          devices::device_configs[selected_device_row];
    } else {
      devices::selected_device_config.reset();
    }
  }

  ImGui::End();
  ImGui::PopStyleVar();
}

void show_app_menu() { ImGui::OpenPopup("AppMenuPopup"); }

void draw_app_menu(PointCaster &app) {
  gui::push_context_menu_styles();
  ImGui::SetNextWindowSize({125, -FLT_MIN});
  if (ImGui::BeginPopup("AppMenuPopup")) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("New Workspace")) {
        ImGui::CloseCurrentPopup();
        pc::logger->info("New workspace...");
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Open Workspace")) {
        ImGui::CloseCurrentPopup();
        pc::logger->info("Open workspace...");
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Save", "S")) {
        ImGui::CloseCurrentPopup();
        app.run_async([&] { app.save_workspace(); });
      }
      if (ImGui::MenuItem("Save As...")) { pc::logger->info("Save as..."); }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit", "Ctrl-Q")) {
        ImGui::CloseCurrentPopup();
        // TODO ask to save & quit modal
        app.run_async([&] { app.quit(); });
      }
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Window")) { 
      auto& w = app.workspace;
      auto& l = app.workspace.layout;
      constexpr auto window_item = [&](auto label, auto shortcut, bool &open) {
        if (ImGui::MenuItem(label, shortcut, open)) open = !open;
      };
      window_item("Camera", "C", l.show_camera_window);
      window_item("Devices", "D", l.show_devices_window);
      window_item("Session Operators", "G", l.show_global_transform_window);
      window_item("Pointcloud Radio", "R", l.show_radio_window);
      window_item("Frame Statistics", "T", l.show_stats);
      ImGui::Separator();
      bool inactive = false;
      window_item("MIDI", "M", w.midi ? w.midi->show_window : inactive);
      window_item("MQTT", "Shift+M", w.mqtt ? w.mqtt->show_window : inactive);
      window_item("OSC Client", "O",
                  w.osc_client ? w.osc_client->show_window : inactive);
      window_item("OSC Server", "Shift+O",
                  w.osc_server ? w.osc_server->show_window : inactive);
      ImGui::Separator();
      auto fs = app._full_screen;
      window_item("Fullscreen", "F", fs);
      if (fs != app._full_screen) app.set_full_screen(fs);
      window_item("Maintain window focus", "", w.maintain_window_focus);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
      if (ImGui::MenuItem("About")) { ImGui::CloseCurrentPopup(); }
      if (ImGui::MenuItem("View third-party licenses")) {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndMenu();
    }
    ImGui::EndPopup();
  }
  gui::pop_context_menu_styles();
}

} // namespace pc::gui