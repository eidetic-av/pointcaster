include(GenerateExportHeader)

find_package(concurrentqueue CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")

add_library(pointreceiver SHARED
    pointreceiver.cc
)

target_link_libraries(pointreceiver PRIVATE
    cppzmq-static
    concurrentqueue::concurrentqueue
    pointcaster::core
)

target_include_directories(pointreceiver PRIVATE
    ${ZPP_BITS_INCLUDE_DIRS}
)

# ----- public C api

set_target_properties(pointreceiver
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
)

generate_export_header(pointreceiver
    EXPORT_FILE_NAME
        ${PROJECT_BINARY_DIR}/include/pointreceiver_export.h
)

target_include_directories(pointreceiver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ----- install

# an integration gets handed one directory, so the installed library looks for
# its dependencies beside itself rather than on a system path
set_target_properties(pointreceiver PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS pointreceiver
    RUNTIME_DEPENDENCY_SET pointreceiver_runtime_deps
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    "${CMAKE_SOURCE_DIR}/include/pointreceiver.h"
    "${PROJECT_BINARY_DIR}/include/pointreceiver_export.h"
    DESTINATION include
)

# pull in the shared libraries it was linked against, skipping anything the
# host already provides
install(RUNTIME_DEPENDENCY_SET pointreceiver_runtime_deps
    DIRECTORIES
        $<TARGET_FILE_DIR:pointcaster_core>
    PRE_EXCLUDE_REGEXES
        "^api-ms-" "^ext-ms-"
        "^(lib)?(c|m|dl|rt|pthread|gcc_s|stdc\\+\\+|mvec)\\.so"
        "^ld-linux"
    POST_EXCLUDE_REGEXES
        "^/(usr/)?lib(32|64)?/"
        "^[A-Za-z]:[\\\\/][Ww]indows[\\\\/]"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION lib
)

# -----

set(pointreceiver_precompiled_headers
  <algorithm>
  <chrono>
  <condition_variable>
  <memory>
  <mutex>
  <optional>
  <ranges>
  <set>
  <span>
  <stop_token>
  <string>
  <string_view>
  <thread>
  <unordered_map>
  <unordered_set>
  <variant>
  <vector>
  <zmq.hpp>
  <zpp_bits.h>
)

target_precompile_headers(pointreceiver PRIVATE
    ${pointreceiver_precompiled_headers}
)
