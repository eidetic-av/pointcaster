#pragma once
#include "../structs.h"
#include "operators.h"
#include <functional>
#include <optional>
#include <vector>

namespace pc::operators {

class SessionOperatorHost {
public:
  SessionOperatorHost(OperatorHostConfiguration &config) : _config(config){};

  operator_in_out_t run_operators(operator_in_out_t begin, operator_in_out_t end) const;

  void draw_imgui_window();

private:
  OperatorHostConfiguration &_config;

  void add_operator(OperatorConfigurationVariant operator_config) {
    _config.operators.push_back(operator_config);
  }
};

using OperatorList =
    std::vector<std::reference_wrapper<const SessionOperatorHost>>;

} // namespace pc::operators
