#include "logger.h"
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace pc::log::detail {
static std::shared_ptr<spdlog::logger> instance;
static std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> ringbuffer_sink;

static std::shared_ptr<spdlog::logger> &get_logger_impl() {
  if (!instance) {
    ringbuffer_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1024);
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks{ringbuffer_sink, stdout_sink};
    instance =
        std::make_shared<spdlog::logger>("pc", sinks.begin(), sinks.end());
    instance->set_level(spdlog::level::info);
  }
  return instance;
}
} // namespace pc::log::detail

namespace pc {
std::shared_ptr<spdlog::logger> &logger() {
  return pc::log::detail::get_logger_impl();
}
void set_log_level(spdlog::level::level_enum lvl) {
  logger()->set_level(lvl);
}
} // namespace pc
