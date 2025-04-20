#include "windows.h"
#include "catpuccin.h"
#include "widgets.h"
#include <imgui.h>

namespace pc::gui {

void draw_devices_window(PointCaster &app) {
  ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({250.0f, 400.0f}, ImGuiCond_FirstUseEver);
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
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Add Device");

  ImGui::Dummy({0, 5});

  // create the scrollable list of device names

  static constexpr size_t visible_device_rows = 6;
  std::optional<int> device_to_delete;
  std::optional<int> device_to_toggle_active;
  std::optional<std::string> edited_device_id;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
  ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0f);
  if (ImGui::BeginChild("DevicesScrollable",
                        ImVec2(0, visible_device_rows * row_height), true)) {
    if (ImGui::BeginTable("DevicesTable", 3, ImGuiTableFlags_RowBg,
                          ImVec2(-FLT_MIN, visible_device_rows * row_height))) {
      ImGui::TableSetupColumn("active_button", ImGuiTableColumnFlags_WidthFixed,
                              icon_width + 5);
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
        std::visit(
            [&](auto &&device_config) {
              device_id = device_config.id;
              is_active = device_config.active;
            },
            devices::device_configs[i].get());

        ImGui::TableNextRow();

        // column 1: view button
        ImGui::TableSetColumnIndex(0);
        ImVec2 cell0_content_min = ImGui::GetCursorPos();
        ImGui::SameLine();
        std::string_view active_icon;
        if (is_active) {
          ImGui::SetCursorPos(
              {cell0_content_min.x + 5, cell0_content_min.y + 1.0f});
          active_icon = ICON_FA_TOGGLE_ON;
        } else {
          ImGui::SetCursorPos(
              {cell0_content_min.x + 5, cell0_content_min.y + 1.0f});
          active_icon = ICON_FA_TOGGLE_OFF;
        }
        if (gui::draw_icon_button(active_icon.data(), false,
                                  catpuccin::imgui::mocha_overlay,
                                  catpuccin::imgui::mocha_overlay2)) {
          device_to_toggle_active = i;
        }

        if (!is_active) ImGui::BeginDisabled();

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

        // column 2: device id with status icon
        ImGui::TableSetColumnIndex(1);
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
        float device_id_cell_width = table_width - ((icon_width + 5) * 2) -
                                     ImGui::GetStyle().ItemSpacing.x;
        ImVec2 device_id_cell_max =
            ImVec2(device_id_cell_min.x + device_id_cell_width,
                   device_id_cell_min.y + row_height);

        if (editing_device_row_id == i) {
          auto popup_label = fmt::format("##device_edit_popup.{}", device_id);
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

        if (!is_active) ImGui::EndDisabled();

        // column 3: delete icon button
        ImGui::TableSetColumnIndex(2);
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
    ImGui::EndChild();
  }
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
            parameters::declare_parameters(config.id, config);
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
            ImGui::BeginChild(
                std::format("##scroll_", device_config.id).c_str(),
                {-FLT_MIN, available_space.y});
            pc::gui::draw_parameters(device_config.id);
            ImGui::EndChild();
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
    static bool requested_orbbec_discovery = false;
    if (!requested_orbbec_discovery) {
      requested_orbbec_discovery = true;
      app.run_async([] { OrbbecDevice::discover_devices(); });
    }

    if (OrbbecDevice::discovering_devices) {
      ImGui::MenuItem("Searching for Orbbec devices...", nullptr, false, false);
    } else {
      size_t orbbec_list_size = 0;
      {
        std::lock_guard lock(OrbbecDevice::devices_access);
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
  }

  ImGui::End();
  ImGui::PopStyleVar();
}
} // namespace pc::gui