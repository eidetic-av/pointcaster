#include "session_operator_host.h"
#include "../gui/widgets.h"
#include "../gui/catpuccin.h"
#include "../logger.h"
#include "../parameters.h"
#include "../uuid.h"
#include "noise_operator.gen.h"
#include "rake_operator.gen.h"
#include "rotate_operator.gen.h"
#include "sample_filter_operator.gen.h"
#include "range_filter_operator.gen.h"
#include <functional>
#include <optional>

#include "../wireframe_objects.h"

namespace pc::operators {

using namespace catpuccin;
using namespace catpuccin::magnum;

static std::map<uid, std::unique_ptr<WireframeBox>> operator_bounding_boxes;

static constexpr std::array<Vector4, 6> bounding_box_colors{
  mocha_red,
  mocha_blue,
  mocha_peach,
  mocha_yellow,
  mocha_lavender,
  mocha_rosewater
};

static Vector3 next_bounding_box_color() {
  static auto current_index = -1;
  current_index = (current_index + 1) % bounding_box_colors.size();
  return bounding_box_colors[current_index].rgb();
}

template <typename T>
static void
set_or_create_bounding_box(const T &operator_config, Scene3D &scene,
			   SceneGraph::DrawableGroup3D &parent_group,
			   std::optional<Color4> color = {}) {
  // get or create the bounding box in the scene
  auto [itr, _] = operator_bounding_boxes.emplace(
      operator_config.id,
      std::make_unique<WireframeBox>(&scene, &parent_group));
  auto &box = itr->second;
  // and set its updated position / scale
  const auto &size = operator_config.size;
  const auto &position = operator_config.position;
  box->set_transformation(Matrix4::scaling({size.x, size.y, size.z}) *
			  Matrix4::translation({0, 0, 0}));
  box->transform(Matrix4::translation({position.x, position.y, position.z}));
  if (color.has_value()) {
    box->set_color(color.value().rgb());
  }
  if constexpr (std::same_as<T, RangeFilterOperatorConfiguration>) {
    box->set_visible(operator_config.draw);
  }
}

SessionOperatorHost::SessionOperatorHost(
    OperatorHostConfiguration &config, Scene3D &scene,
    Magnum::SceneGraph::DrawableGroup3D &parent_group)
    : _config(config), _scene(scene), _parent_group(parent_group) {

  // for loading an existing list of operators
  for (auto &operator_instance : _config.operators) {
    // we need to visit each possible operator variant
    std::visit(
        [&](auto &&operator_config) {
	  // and declare it's saved ID with its parameters
	  declare_parameters(std::to_string(operator_config.id),
			     operator_config);

          using T = std::decay_t<decltype(operator_config)>;

	  if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {
	    set_or_create_bounding_box(operator_config, scene, parent_group,
				       next_bounding_box_color());
          }

        },
        operator_instance);
  }
};

bool operator_collapsing_header(
    const char *label,
    std::optional<std::function<void()>> delete_callback = {}) {
  using namespace ImGui;
  PushID(gui::_parameter_index++);
  PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  bool *p_open = GetStateStorage()->GetBoolRef(GetID(label), false);
  ImGuiStyle &style = ImGui::GetStyle();
  if (ArrowButton(fmt::format("{}##arrow", label).c_str(),
		  *p_open ? ImGuiDir_Down : ImGuiDir_Right)) {
    *p_open ^= 1;
  }
  ImGui::SameLine();
  if (delete_callback.has_value()) {
    const auto close_button_width = ImGui::CalcTextSize(" X ").x * 1.5;
    if (Button(label, ImVec2(-FLT_MIN - close_button_width, 0.0f))) {
      *p_open ^= 1;
    }
    ImGui::SameLine();
    if (ImGui::Button(" X ")) {
      delete_callback.value()();
    };
  } else {
    if (Button(label, ImVec2(-FLT_MIN, 0.0f))) {
      *p_open ^= 1;
    }
  }
  PopStyleVar();
  ImGui::PopID();
  return *p_open;
}

void SessionOperatorHost::draw_imgui_window() {
  ImGui::SetNextWindowSize({600, 400}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Session Operators", nullptr);

  static bool select_node = false;

  if (ImGui::Button("Add session operator")) {
    ImGui::OpenPopup("Add session operator");
  }

  if (ImGui::BeginPopup("Add session operator")) {
    // populate menu with all Operator types

    apply_to_all_operators([this](auto &&operator_type) {
      using T = std::remove_reference_t<decltype(operator_type)>;
      if (ImGui::Selectable(T::Name)) {

        auto operator_config = T();
        operator_config.id = pc::uuid::digit();
	// create a new instance of our operator configuration and add it to our
	// session operator list
	auto &variant_ref = _config.operators.emplace_back(operator_config);

	// declare the instance as parameters to bind to this new operator's id
	auto &config_instance = std::get<T>(variant_ref);
	declare_parameters(std::to_string(operator_config.id), config_instance);

          if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {
	    set_or_create_bounding_box(operator_config, _scene, _parent_group,
				       next_bounding_box_color());
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

          ImGui::PushID(gui::_parameter_index++);
          if (ImGui::CollapsingHeader(T::Name)) {
          // if (operator_collapsing_header(
          //         T::Name, [&] { marked_for_delete = config.id; })) {
	    config.unfolded = true;

            if (pc::gui::draw_parameters(config.id)) {
	      if constexpr (std::is_same_v<T,
					   RangeFilterOperatorConfiguration>) {
		set_or_create_bounding_box(config, _scene, _parent_group);
	      }
            }
          } else {
            config.unfolded = false;
          }
          ImGui::PopID();
        },
        operator_config);
    ImGui::PopID();
  }

  if (marked_for_delete.has_value()) {
    const auto check_marked_id = [&](const auto &config) {
      return config.id == marked_for_delete.value();
    };
    auto new_end = std::remove_if(
	_config.operators.begin(), _config.operators.end(),
	[&](const auto &op) { return std::visit(check_marked_id, op); });
    _config.operators.erase(new_end, _config.operators.end());
  }

  ImGui::End();
}

} // namespace pc::operators
