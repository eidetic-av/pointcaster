#include "logger.h"
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace pc::log::detail {
static std::shared_ptr<spdlog::logger> instance;
static std::shared_ptr<spdlog::sinks::basic_file_sink_mt> file_sink;
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

static std::filesystem::path logs_dir(std::string_view target_name) {
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
  // $XDG_STATE_HOME/matth/<app>
  // else ~/.local/state/matth/<app>
  if (auto p = get_env_path("XDG_STATE_HOME"))
    return (*p) / "matth" / std::string(target_name);
  if (auto home = get_env_path("HOME"))
    return (*home) / ".local" / "state" / "matth" / std::string(target_name);
  return std::filesystem::current_path() / "matth" / std::string(target_name);
#endif
}

static std::filesystem::path log_file_path(std::string_view target_name) {
  const auto dir = logs_dir(target_name);
  return dir / (std::string(target_name) + ".log");
}

static void rotate_file_logs(const std::filesystem::path &base_file,
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

} // namespace pc::log::detail

namespace pc {

std::shared_ptr<spdlog::logger> &logger() {
  return pc::log::detail::get_logger_impl();
}

void set_log_level(spdlog::level::level_enum lvl) {
  logger()->set_level(lvl);
}

void enable_file_logging(std::string_view target) {
  auto &logger = pc::logger();

  constexpr std::size_t keep_runs = 5;

  assert(!target.empty());
  const auto path = pc::log::detail::log_file_path(target);

  // If already enabled to same file, no-op.
  if (pc::log::detail::file_sink) {
    try {
      const auto current_filename = pc::log::detail::file_sink->filename();
      if (std::filesystem::path(current_filename) == path) return;
    } catch (...) {
    }

    auto &sinks = logger->sinks();
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

  pc::log::detail::rotate_file_logs(path, keep_runs);

  try {
    constexpr bool truncate = true;

    pc::log::detail::file_sink =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.string(),
                                                            truncate);

    pc::log::detail::file_sink->set_level(spdlog::level::trace);
    // pc::log::detail::file_sink->set_pattern(
    //     "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->sinks().push_back(pc::log::detail::file_sink);

    logger->flush_on(spdlog::level::trace);

    logger->info("enabled file logging: {}", path.string());
  } catch (const std::exception &e) {
    logger->error("failed to enable file logging {}: {}", path.string(),
                  e.what());
  } catch (...) {
    logger->error("failed to enable file logging {}: <unknown error>",
                  path.string());
  }
}

void disable_file_logging() {
  if (!pc::log::detail::file_sink) return;

  auto &logger = pc::logger();

  auto &sinks = logger->sinks();
  sinks.erase(std::remove_if(sinks.begin(), sinks.end(),
                             [](const spdlog::sink_ptr &s) {
                               return s == pc::log::detail::file_sink;
                             }),
              sinks.end());

  pc::log::detail::file_sink.reset();
  logger->info("disabled file logging");
}

std::vector<LogEntry>
logger_lines(std::size_t n, std::chrono::system_clock::duration duration) {

  (void)pc::logger();

  auto raw_msgs = pc::log::detail::ringbuffer_sink->last_raw(n);
  std::vector<LogEntry> log_entries;

  const bool has_duration =
      duration != std::chrono::system_clock::duration::zero();

  std::chrono::system_clock::time_point now;
  if (has_duration) now = std::chrono::system_clock::now();

  for (const auto &msg : raw_msgs) {
    if (has_duration && (now - msg.time > duration)) continue;

    log_entries.emplace_back(LogEntry{
        msg.level, std::string(msg.payload.data(), msg.payload.size())});
  }
  return log_entries;
}

} // namespace pc
