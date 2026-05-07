#pragma once

#include "../operator_plugin.h"

#include "fringe_removal_config.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <plugins/backend/backend_plugin.h>

namespace pc::operators {

class FringeRemovalOperator : public OperatorPlugin {
public:
  explicit FringeRemovalOperator(
      Corrade::PluginManager::AbstractManager &manager,
      Corrade::Containers::StringView plugin)
      : OperatorPlugin(manager, plugin) {}

  ~FringeRemovalOperator() override {}

  FringeRemovalOperator(const FringeRemovalOperator &) = delete;
  FringeRemovalOperator &operator=(const FringeRemovalOperator &) = delete;
  FringeRemovalOperator(FringeRemovalOperator &&) = delete;
  FringeRemovalOperator &operator=(FringeRemovalOperator &&) = delete;

  void init() override;
  void
  process(const PointCloud &input, PointCloud &output,
          const OperatorConfigurationVariant &config_variant) const override;
};

} // namespace pc::operators