#pragma once

#include <string_view>
#include <reckless/policy_log.hpp>
#include <reckless/stdout_writer.hpp>

namespace bob {

namespace __log {

class severity_field {
public:
  severity_field(std::string_view severity) : severity_(severity) {}

  void format(reckless::output_buffer *poutput_buffer) const {
    const char* p = severity_.c_str();
    size_t n = severity_.length();
    char* buffer = poutput_buffer->reserve(n);
    std::memcpy(buffer, p, n);
    poutput_buffer->commit(n);
  }

private:
  std::string severity_;
};

namespace detail {
  template <class HeaderField> HeaderField construct_header_field(std::string_view) {
  return HeaderField();
}

template <>
inline severity_field construct_header_field<severity_field>(std::string_view severity) {
  return severity_field(severity);
}
} // namespace detail

template <class IndentPolicy, char FieldSeparator, class... HeaderFields>
class severity_log : public reckless::basic_log {
public:
  using basic_log::basic_log;

  template <typename... Args> void debug(char const *fmt, Args &&...args) {
    write("[\033[32mdebug\033[0m]", fmt, std::forward<Args>(args)...);
  }
  template <typename... Args> void info(char const *fmt, Args &&...args) {
    write("[\033[37m info\033[0m]", fmt, std::forward<Args>(args)...);
  }
  template <typename... Args> void warn(char const *fmt, Args &&...args) {
    write("[\033[35m warn\033[0m]", fmt, std::forward<Args>(args)...);
  }
  template <typename... Args> void error(char const *fmt, Args &&...args) {
    write("[\033[31merror\033[0m]", fmt, std::forward<Args>(args)...);
  }

private:
  template <typename... Args>
  void write(std::string_view severity, char const *fmt, Args &&...args) {
    basic_log::write<reckless::policy_formatter<IndentPolicy, FieldSeparator,
						HeaderFields...>>(
	detail::construct_header_field<HeaderFields>(severity)...,
	IndentPolicy(), fmt, std::forward<Args>(args)...);
  }
};

using log_t = severity_log<reckless::indent<4>, ' ', severity_field,
			   reckless::timestamp_field>;

inline reckless::stdout_writer stdout_writer;

} // namespace __log

inline __log::log_t log(&__log::stdout_writer);

} // namespace bob
