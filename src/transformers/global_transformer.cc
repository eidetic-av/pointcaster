#include "global_transformer.h"
#include "../logger.h"
#include "../gui/widgets.h"

#include "noise_transformer.h"

namespace pc::transformers {

void GlobalTransformer::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Global Transformers", nullptr);

  if (ImGui::Button("Add")) {
    _config.transformers.push_back(NoiseTransformerConfiguration());
  }

  for (auto &transformer_config : _config.transformers) {
    std::visit(
	[](auto &&transformer_config) {
	  using T = std::decay_t<decltype(transformer_config)>;
	  if constexpr (std::is_same_v<T, NoiseTransformerConfiguration>) {
	    NoiseTransformer::draw_imgui_controls(transformer_config);
	  }
	},
	transformer_config);
  }

  ImGui::End();
}

}
