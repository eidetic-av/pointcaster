include(GenerateExportHeader)

find_package(concurrentqueue CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")

add_library(pointreceiver SHARED
    pointreceiver.cc
)

add_executable(pointreceiver_test
    pointreceiver.cc
)

target_link_libraries(pointreceiver PRIVATE
    cppzmq-static
    concurrentqueue::concurrentqueue
    pointcaster::core
)

target_link_libraries(pointreceiver_test PRIVATE
    cppzmq-static
    concurrentqueue::concurrentqueue
    pointcaster::core
)

target_include_directories(pointreceiver PRIVATE
    ${ZPP_BITS_INCLUDE_DIRS}
)
target_include_directories(pointreceiver_test PRIVATE
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

# the test executable compiles the same sources directly, so the api symbols
# are its own rather than imported from the shared library
target_include_directories(pointreceiver_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/include
)
target_compile_definitions(pointreceiver_test PRIVATE
    POINTRECEIVER_STATIC_DEFINE
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
target_precompile_headers(pointreceiver_test PRIVATE
    ${pointreceiver_precompiled_headers}
)
