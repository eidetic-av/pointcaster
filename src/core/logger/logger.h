#pragma once

#include <memory>
#include <pointcaster/core.h>
#include <spdlog/common.h>
#include <spdlog/logger.h>

namespace pc {
POINTCASTER_CORE_EXPORT std::shared_ptr<spdlog::logger> &logger();
POINTCASTER_CORE_EXPORT void set_log_level(spdlog::level::level_enum lvl);
} // namespace pc
