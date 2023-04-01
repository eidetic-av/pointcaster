#include <reckless/severity_log.hpp>
#include <reckless/stdout_writer.hpp>

using log_t = reckless::severity_log<
  reckless::indent<4>,       // 4 spaces of indent
  ' ',                       // Field separator
  reckless::severity_field,  // Show severity marker (D/I/W/E) first
  reckless::timestamp_field  // Then timestamp field
  >;

reckless::stdout_writer __stdout_writer;
log_t g_log(&__stdout_writer);

