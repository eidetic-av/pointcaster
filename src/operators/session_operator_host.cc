#include "session_operator_host.h"
#include "../aabb.h"
#include "../gui/catpuccin.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../parameters.h"
#include "../string_utils.h"
#include "../uuid.h"
#include "noise_operator.gen.h"
#include "operator_friendly_names.h"
#include "operator_traits.h"
#include "pcl/cluster_extraction_operator.gen.h"
#include "rake_operator.gen.h"
#include "range_filter_operator.gen.h"
#include "rotate_operator.gen.h"
#include "sample_filter_operator.gen.h"
#include "session_bounding_boxes.h"
#include <algorithm>
#include <execution>
#include <functional>
#include <optional>

#include <tbb/tbb.h>

namespace pc::operators {

using namespace pc::types;
using namespace pc::parameters;
using namespace oneapi;

SessionOperatorHost::SessionOperatorHost(
    OperatorHostConfiguration &config, Scene3D &scene,
    Magnum::SceneGraph::DrawableGroup3D &parent_group)
    : _config(config), _scene(scene), _parent_group(parent_group) {

  // for loading an existing list of operators
  for (auto &operator_config_variant : _config.operators) {
    std::visit(
        [&](auto &&operator_config) {
          // declare the serialised operator ID with its config parameters
          declare_parameters(std::to_string(operator_config.id),
                             operator_config);

          using T = std::decay_t<decltype(operator_config)>;

          // call initialisation if its required
          if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {
            RangeFilterOperator::init(operator_config, _scene, _parent_group);
          }

        },
        operator_config_variant);
  }

  using namespace std::chrono_literals;

  instance = this;
};

bool operator_collapsing_header(
    uid operator_id, std::optional<std::function<void()>> delete_callback = {},
    std::optional<std::function<void(std::string_view)>> title_edit_callback = {}) {
  const char *label = get_operator_friendly_name_cstr(operator_id);
  using namespace ImGui;
  PushID(gui::_parameter_index++);
  PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  bool *p_open = GetStateStorage()->GetBoolRef(GetID(label), false);

  // Render arrow button
  bool arrow_clicked =
      ArrowButton(fmt::format("{}##arrow_{}", label, operator_id).c_str(),
                  *p_open ? ImGuiDir_Down : ImGuiDir_Right);
  if (arrow_clicked) { *p_open = !(*p_open); }
  SameLine();

  bool label_clicked = false;
  bool label_right_clicked = false;
  auto button_label = fmt::format("{}##button_{}", label, operator_id);
  if (delete_callback.has_value()) {
    const auto close_button_width = CalcTextSize(" X ").x * 1.5f;
    label_clicked = Button(button_label.c_str(),
                           ImVec2(-FLT_MIN - close_button_width, 0.0f));
    label_right_clicked = IsItemClicked(ImGuiMouseButton_Right);
    SameLine();
    if (Button(" X ")) { delete_callback.value()(); }
  } else {
    label_clicked = Button(button_label.c_str(), ImVec2(-FLT_MIN, 0.0f));
    label_right_clicked = IsItemClicked(ImGuiMouseButton_Right);
  }

  static std::string original_title;
  static bool edit_popup_opened = false;
  auto popup_label = fmt::format("{}##editpopup_{}", label, operator_id);

  if (label_clicked) *p_open = !(*p_open);
  else if (label_right_clicked) {
    original_title = label;
    OpenPopup(popup_label.c_str());
  }

  // Popup: Editable text field for title editing.
  static char edit_buf[256] = "";
  if (BeginPopup(popup_label.c_str())) {
    // Initialize edit_buf with the current label if just opened.
    if (!edit_popup_opened) {
      strncpy(edit_buf, label, sizeof(edit_buf) - 1);
      edit_buf[sizeof(edit_buf) - 1] = '\0';
      edit_popup_opened = true;
    }
    // Render InputText; use flags to auto-select and commit on Enter.
    if (InputText(fmt::format("##rename_{}", popup_label).c_str(), edit_buf,
                  IM_ARRAYSIZE(edit_buf),
                  ImGuiInputTextFlags_AutoSelectAll |
                      ImGuiInputTextFlags_EnterReturnsTrue)) {
      std::string new_title(edit_buf);
      if (new_title.empty()) new_title = original_title;
      if (title_edit_callback.has_value()) {
        title_edit_callback.value()(new_title);
      }
      CloseCurrentPopup();
      edit_popup_opened = false;
    }
    EndPopup();
  } else {
    edit_popup_opened = false;
  }

  PopStyleVar();
  PopID();
  return *p_open;
}

void SessionOperatorHost::draw_gizmos() {

  // operator_bounding_boxes.clear();

  for (const auto &operator_config : _config.operators) {
    std::visit(
        [this](auto &&config) {
          using T = std::decay_t<decltype(config)>;

          if constexpr (std::is_same_v<
                            T, pcl_cpu::ClusterExtractionConfiguration>) {
            auto &pipeline = pcl_cpu::ClusterExtractionPipeline::instance();

            if (config.draw_voxels) {
              auto current_voxels = pipeline.current_voxels.load();
              if (current_voxels.get() != nullptr) {

                // TODO this block is too slow
                // probably should be replaced by some instancing shader similar
                // to how we render point-clouds, just as cubes

                // using namespace std::chrono;
                // using namespace std::chrono_literals;
                // auto start = high_resolution_clock::now();
                auto voxel_count =
                    current_voxels->size(); // pc::logger->debug("voxel count:
                                            // {}", voxel_count);
                if (!_last_voxel_count.has_value()) {
                  _last_voxel_count = voxel_count;
                }
                for (int i = 0; i < *_last_voxel_count; i++) {
                  pcl::PointXYZ p{};
                  bool visible = false;
                  if (i < voxel_count) {
                    p = current_voxels->points[i];
                    visible = true;
                  }
                  // mm to metres
                  set_voxel(i, visible,
                            {p.x / 1000.0f, p.y / 1000.0f, p.z / 1000.0f});
                }
                _last_voxel_count = voxel_count;
                // auto end = high_resolution_clock::now();
                // int duration_us = duration_cast<microseconds>(end -
                // start).count(); float duration_ms = duration_us / 1'000.0f;
                // pc::logger->info("drawing voxels took: {}ms",
                // fmt::format("{:.2f}", duration_ms));
              }
            }

            if (config.draw_clusters) {
              auto latest_clusters_ptr = pipeline.current_clusters.load();
              if (latest_clusters_ptr.get() != nullptr) {
                // using namespace std::chrono;
                // using namespace std::chrono_literals;
                // auto start = high_resolution_clock::now();
                auto &clusters = *latest_clusters_ptr;
                auto cluster_count = clusters.size();
                static size_t last_cluster_count = cluster_count;
                for (size_t i = 0; i < last_cluster_count; i++) {
                  Float3 position, size;
                  bool visible = false;
                  if (i < cluster_count) {
                    pc::AABB aabb = clusters[i].bounding_box;
                    position = aabb.center() / 1000.0f; // mm to metres
                    size = aabb.extents() / 1000.0f;
                    visible = true;
                  }
                  set_cluster(i, visible, position, size);
                }
                last_cluster_count = cluster_count;
                // auto end = high_resolution_clock::now();
                // int duration_us = duration_cast<microseconds>(end -
                // start).count(); float duration_ms = duration_us / 1'000.0f;
                // pc::logger->info("drawing clusters took: {}ms",
                // fmt::format("{:.2f}", duration_ms));
              }
            }
          }
        },
        operator_config);
  }
}

void SessionOperatorHost::draw_imgui_window() {
  ImGui::SetNextWindowSize({600, 400}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Session Operators", nullptr);

  static bool select_node = false;

  ImGui::Dummy({10, 0});
  ImGui::SameLine();
  if (ImGui::Button("Add operator")) { ImGui::OpenPopup("Add operator"); }
  ImGui::Spacing();
  ImGui::Dummy({0, 10});

  if (ImGui::BeginPopup("Add operator")) {
    // populate menu with all Operator types

    apply_to_all_operators([this](auto &&operator_type) {
      // creating a new operator
      using T = std::decay_t<decltype(operator_type)>;
      if (ImGui::Selectable(T::Name)) {

        auto operator_config = T();
        operator_config.id = pc::uuid::digit();
        // create a new instance of our operator configuration and add it to our
        // session operator list
        auto &variant_ref = _config.operators.emplace_back(operator_config);

        // ensure an initial friendly name for this operator is added to our map
        size_t num = 0;
        std::string friendly_name =
            fmt::format("{}_{}", pc::strings::snake_case(T::Name), num);
        while (operator_friendly_name_exists(friendly_name)) {
          friendly_name =
              fmt::format("{}_{}", pc::strings::snake_case(T::Name), ++num);
        }
        set_operator_friendly_name(operator_config.id, friendly_name);

        // declare the instance as parameters to bind to this new operator's id
        declare_parameters(std::to_string(operator_config.id),
                           std::get<T>(variant_ref));

        if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {
          RangeFilterOperator::init(operator_config, _scene, _parent_group);
        }

        ImGui::CloseCurrentPopup();
      }
    });

    ImGui::EndPopup();
  }

  std::optional<uid> marked_for_delete;

  for (auto &operator_config : _config.operators) {
    ImGui::PushID(gui::_parameter_index++);

    std::visit(
        [&](auto &&config) {
          using T = std::decay_t<decltype(config)>;

          auto id = std::to_string(gui::_parameter_index++);
          ImGui::PushID(gui::_parameter_index++);
          ImGui::BeginGroup();

          const auto handle_delete = [&] {
            marked_for_delete = config.id;
            // any delete actions for specific operators
            if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {
              operator_bounding_boxes.erase(config.id);
            }
          };
          const auto handle_title_edit = [&](std::string_view new_title) {
            if (!operator_friendly_name_exists(new_title)) {
              set_operator_friendly_name(config.id, new_title);
            } else {
              pc::logger->error("'{}' already exists", new_title);
            }
          };

          if (operator_collapsing_header(config.id, handle_delete,
                                         handle_title_edit)) {
            config.unfolded = true;

            ImGui::Dummy({0, 6});
            ImGui::Dummy({10, 0});
            ImGui::SameLine();
            ImGui::BeginDisabled(true);
            ImGui::Text(T::Name);
            ImGui::EndDisabled();
            ImGui::Dummy({0, 3});

            const auto previous_config = config;
            bool updated_param = pc::gui::draw_parameters(config.id);

            if (updated_param) {
              if constexpr (std::same_as<T, RangeFilterOperatorConfiguration>) {
                RangeFilterOperator::update(config, _scene, _parent_group);
              } else if constexpr (
                  std::same_as<T, pcl_cpu::ClusterExtractionConfiguration>) {
                if (previous_config.draw_voxels && !config.draw_voxels) {
                  for (int i = 0; i < _last_voxel_count; i++) {
                    set_voxel(i, false);
                  }
                }
              }
            }
          } else {
            config.unfolded = false;
          }
          ImGui::EndGroup();
          ImGui::PopID();
        },
        operator_config);
    ImGui::PopID();
  }

  if (marked_for_delete.has_value()) {
    std::erase_if(_config.operators, [&](const auto &operator_variant) {
      return std::visit(
          [&](const auto &op) { return op.id == marked_for_delete.value(); },
          operator_variant);
    });
  }

  ImGui::End();
}

void SessionOperatorHost::set_voxel(pc::types::Float3 position,
                                    pc::types::Float3 size) {
  set_or_create_bounding_box(pc::uuid::digit(), size, position, _scene,
                             _parent_group, true,
                             catpuccin::magnum::mocha_blue);
}

void SessionOperatorHost::set_voxel(uid id, bool visible,
                                    pc::types::Float3 position,
                                    pc::types::Float3 size) {
  set_or_create_bounding_box(id, size, position, _scene, _parent_group, visible,
                             catpuccin::magnum::mocha_blue);
}

void SessionOperatorHost::set_cluster(uid id, bool visible,
                                      pc::types::Float3 position,
                                      pc::types::Float3 size) {
  set_or_create_bounding_box(10000 + id, size, position, _scene, _parent_group,
                             visible, catpuccin::magnum::mocha_red);
}

} // namespace pc::operators
