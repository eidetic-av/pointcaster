#pragma once

#include "logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <memory>
#include <string_view>
#include <optional>
#include <chrono>

namespace pc {
  namespace log::detail {
    inline std::shared_ptr<spdlog::logger> instance;
    inline std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> ringbuffer_sink;

    inline std::shared_ptr<spdlog::logger>& get_logger() {
      if (!instance) {
        // Create a ring buffer sink with a fixed capacity
        ringbuffer_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1024);

        // Create a stdout sink
        auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        // Combine sinks
        std::vector<spdlog::sink_ptr> sinks {ringbuffer_sink, stdout_sink};

        // Create the logger
        instance = std::make_shared<spdlog::logger>("pc", sinks.begin(), sinks.end());
      }
      return instance;
    }
  }
  inline std::shared_ptr<spdlog::logger>& logger = pc::log::detail::get_logger();

  struct log_entry {
    spdlog::level::level_enum level;
    std::string message;
  };

  /* Retrieves the last n lines from the ringbuffer
   */
  inline std::vector<log_entry> logger_lines(std::size_t n, std::optional<std::chrono::system_clock::duration> duration = std::nullopt) {
    auto raw_msgs = pc::log::detail::ringbuffer_sink->last_raw(n);
    std::vector<log_entry> log_entries;

    std::chrono::system_clock::time_point now;
    if (duration.has_value()) {
      now = std::chrono::system_clock::now();
    }

    for (const auto& msg : raw_msgs) {
      if (duration.has_value()) {
	auto msg_timestamp = std::chrono::time_point<std::chrono::system_clock>{std::chrono::nanoseconds{msg.time.time_since_epoch().count()}};
	if (now - msg_timestamp > duration.value()) continue; 
      }
      log_entries.emplace_back(log_entry{msg.level, std::string(msg.payload.data(), msg.payload.size())});
    }
    return log_entries;
  }
}
