#pragma once
#include <vector>
#include <functional>
#include <optional>
#include "transformers.h"
#include "../structs.h"

namespace pc::transformers {

class GlobalTransformer {
public:
  GlobalTransformer(TransformerConfiguration &config) : _config(config){};

  void run_transformers(transformer_in_out_t begin,
                        transformer_in_out_t end) const;

  void draw_imgui_window();

private:
  TransformerConfiguration& _config;

  void add_transformer(TransformerConfigurationVariant transformer_config) {
    _config.transformers.push_back(transformer_config);
  }

};

using TransformerList =
    std::vector<std::reference_wrapper<const GlobalTransformer>>;
}
