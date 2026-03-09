#pragma once

#include <memory>
#include <pointcaster/core.h>
#include <spdlog/common.h>
#include <spdlog/logger.h>

#include <chrono>
#include <filesystem>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <string_view>

namespace pc {
POINTCASTER_CORE_EXPORT std::shared_ptr<spdlog::logger> &logger();

POINTCASTER_CORE_EXPORT void set_log_level(spdlog::level::level_enum lvl);

POINTCASTER_CORE_EXPORT void enable_file_logging(std::string_view target = "");
POINTCASTER_CORE_EXPORT void disable_file_logging();

POINTCASTER_CORE_EXPORT const std::string_view log_file_target_name();
POINTCASTER_CORE_EXPORT void set_log_file_target_name(std::string_view name);

struct LogEntry {
  spdlog::level::level_enum level;
  std::string message;
};

POINTCASTER_CORE_EXPORT std::vector<LogEntry>
logger_lines(std::size_t n, std::chrono::system_clock::duration duration = {});

} // namespace pc
