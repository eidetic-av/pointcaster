#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace pc {
namespace log::detail {

inline std::shared_ptr<spdlog::logger> instance;
inline std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> ringbuffer_sink;
inline std::shared_ptr<spdlog::sinks::basic_file_sink_mt> file_sink;

inline std::shared_ptr<spdlog::logger> &get_logger() {
  if (!instance) {
    ringbuffer_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1024);
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    std::vector<spdlog::sink_ptr> sinks{ringbuffer_sink, stdout_sink};

    instance =
        std::make_shared<spdlog::logger>("pc", sinks.begin(), sinks.end());

#ifdef NDEBUG
    instance->set_level(spdlog::level::info);
    instance->info("Release build - set log level to info");
#else
    instance->set_level(spdlog::level::debug);
    instance->info("Debug build - set log level to debug");
#endif
  }
  return instance;
}

inline std::string &log_target_name_storage() {
  static std::string name = "pointcaster";
  return name;
}

inline std::string_view log_target_name() { return log_target_name_storage(); }

inline void set_log_target_name(std::string_view name) {
  if (!name.empty()) log_target_name_storage() = std::string(name);
}

inline std::filesystem::path default_logs_dir(std::string_view app) {
  const auto get_env_path =
      [](const char *name) -> std::optional<std::filesystem::path> {
    if (!name) return std::nullopt;
    const char *v = std::getenv(name);
    if (!v || *v == '\0') return std::nullopt;
    return std::filesystem::path(v);
  };

#if defined(_WIN32)
  // %LOCALAPPDATA%\matth\<app>
  if (auto p = get_env_path("LOCALAPPDATA"))
    return (*p) / "matth" / std::string(app);
  if (auto p = get_env_path("APPDATA"))
    return (*p) / "matth" / std::string(app);
  return std::filesystem::current_path() / "Log" / std::string(app);
#else
  // $XDG_STATE_HOME/matth/<app> (preferred for logs)
  // else $XDG_DATA_HOME/matth/<app>
  // else ~/.local/state/matth/<app>
  if (auto p = get_env_path("XDG_STATE_HOME"))
    return (*p) / "matth" / std::string(app);
  if (auto p = get_env_path("XDG_DATA_HOME"))
    return (*p) / "matth" / std::string(app);
  if (auto home = get_env_path("HOME"))
    return (*home) / ".local" / "state" / "matth" / std::string(app);
  return std::filesystem::current_path() / "matth" / std::string(app);
#endif
}

inline std::filesystem::path default_log_file_path(std::string_view app) {
  const auto dir = default_logs_dir(app);
  return dir / (std::string(app) + ".log");
}

inline void rotate_run_logs(const std::filesystem::path &base_file,
                            std::size_t keep_count) {
  if (keep_count < 2) {
    std::error_code ec_rm;
    std::filesystem::remove(base_file, ec_rm);
    return;
  }

  const auto make_rotated = [&](std::size_t idx) -> std::filesystem::path {
    return base_file.string() + "." + std::to_string(idx);
  };

  std::filesystem::remove(make_rotated(keep_count - 1));

  for (std::size_t i = keep_count - 1; i-- > 1;) {
    const auto src = make_rotated(i);
    const auto dst = make_rotated(i + 1);
    if (std::filesystem::exists(src)) std::filesystem::rename(src, dst);
  }

  if (std::filesystem::exists(base_file))
    std::filesystem::rename(base_file, make_rotated(1));
}

} // namespace log::detail

inline std::shared_ptr<spdlog::logger> &logger() {
  return pc::log::detail::get_logger();
}

struct log_entry {
  spdlog::level::level_enum level;
  std::string message;
};

inline std::vector<log_entry>
logger_lines(std::size_t n,
             std::optional<std::chrono::system_clock::duration> duration =
                 std::nullopt) {
  (void)pc::log::detail::get_logger();

  auto raw_msgs = pc::log::detail::ringbuffer_sink->last_raw(n);
  std::vector<log_entry> log_entries;

  std::chrono::system_clock::time_point now;
  if (duration.has_value()) now = std::chrono::system_clock::now();

  for (const auto &msg : raw_msgs) {
    if (duration.has_value() && (now - msg.time > duration.value())) continue;

    log_entries.emplace_back(log_entry{
        msg.level, std::string(msg.payload.data(), msg.payload.size())});
  }
  return log_entries;
}

inline void
enable_file_logging(std::optional<std::string_view> target = std::nullopt) {
  auto &lg = pc::log::detail::get_logger();

  if (target.has_value() && !target->empty())
    pc::log::detail::set_log_target_name(*target);

  constexpr std::size_t keep_runs = 5;

  const auto app = pc::log::detail::log_target_name();
  const auto path = pc::log::detail::default_log_file_path(app);

  // If already enabled to same file, no-op.
  if (pc::log::detail::file_sink) {
    try {
      const auto current_filename = pc::log::detail::file_sink->filename();
      if (std::filesystem::path(current_filename) == path) return;
    } catch (...) {}

    auto &sinks = lg->sinks();
    sinks.erase(std::remove_if(sinks.begin(), sinks.end(),
                               [](const spdlog::sink_ptr &s) {
                                 return s == pc::log::detail::file_sink;
                               }),
                sinks.end());
    pc::log::detail::file_sink.reset();
  }

  // Ensure directory exists.
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  // TODO errors?

  pc::log::detail::rotate_run_logs(path, keep_runs);

  try {
    constexpr bool truncate = true;

    pc::log::detail::file_sink =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.string(),
                                                            truncate);

    pc::log::detail::file_sink->set_level(spdlog::level::trace);
    // pc::log::detail::file_sink->set_pattern(
    //     "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    lg->sinks().push_back(pc::log::detail::file_sink);

    lg->flush_on(spdlog::level::trace);

    lg->info("enabled file logging: {}", path.string());
  } catch (const std::exception &e) {
    lg->error("failed to enable file logging {}: {}", path.string(), e.what());
  } catch (...) {
    lg->error("failed to enable file logging {}: <unknown error>",
              path.string());
  }
}

inline void disable_file_logging() {
  auto &lg = pc::log::detail::get_logger();
  if (!pc::log::detail::file_sink) return;

  auto &sinks = lg->sinks();
  sinks.erase(std::remove_if(sinks.begin(), sinks.end(),
                             [](const spdlog::sink_ptr &s) {
                               return s == pc::log::detail::file_sink;
                             }),
              sinks.end());

  pc::log::detail::file_sink.reset();
  lg->info("disabled file logging");
}

} // namespace pc
