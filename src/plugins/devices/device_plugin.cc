#include "device_plugin.h"
#include <exception>
#include <logger/logger.h>
#include <plugins/operators/operator_plugin.h>
#include <uuid/uuid.h>
#include <workspace/workspace.h>

namespace pc::devices {

void DevicePlugin::add_operator(const std::string_view plugin_name) {
  Corrade::Containers::Pointer<operators::OperatorPlugin> operator_instance;
  try {
    operator_instance = _workspace->operator_plugin_manager->instantiate(
        std::string(plugin_name.data(), plugin_name.size()));
  } catch (const std::exception &e) {
    pc::logger()->error(e.what());
    return;
  }
  if (!operator_instance) {
    pc::logger()->error("Failed to instantiate {}", plugin_name);
    return;
  }
  operator_instance->init();
  operators.push_back(std::move(operator_instance));
}

void DevicePlugin::add_operator(
    const operators::OperatorConfigurationVariant &operator_config) {
  //   //
  //   const auto [operator_id, plugin_name] =
  //       operator_info_from_variant(operator_config);
  //   pc::logger->debug("id: {}, plugin_name: {}", id, plugin_name);

  //   std::visit(
  //       [this, new_operator_config = operator_config](auto device_config) {
  //         if (new_operator_config.id.empty()) {
  //           new_operator_config.id = pc::uuid::word();
  //         }
  //         device_config.operators.push_back(new_operator_config);
  //         update_config(device_config);
  //       },
  //       _config);
}

} // namespace pc::devices