#pragma once
#include "../structs.h"
#include "operator_host_config.gen.h"
#include <functional>
#include <optional>
#include <vector>

namespace pc::operators {

class SessionOperatorHost {
public:

  static operator_in_out_t run_operators(operator_in_out_t begin,
					 operator_in_out_t end,
					 OperatorHostConfiguration &host_config);

  SessionOperatorHost(OperatorHostConfiguration &config);

  void draw_imgui_window();

  OperatorHostConfiguration &_config;

private:

  void add_operator(OperatorConfigurationVariant operator_config) {
    _config.operators.push_back(operator_config);
  }
};

using OperatorList =
    std::vector<std::reference_wrapper<const SessionOperatorHost>>;

extern operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
			       const OperatorList& operator_list); 

} // namespace pc::operators
