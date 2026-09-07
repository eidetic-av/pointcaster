include(GenerateExportHeader)

# the core library is a shared library so that singletons like 
# the logger and  the profiler can be accessed from inside 
# other shared libraries, like dynamically loaded device plugins
add_library(pointcaster_core SHARED
    point_cloud.cc
    core_types.cc
    task_pool.cc
    plugin_manager_lock.cc
    codec/draco.cc
    uuid/uuid.cc
    uuid/wtf/wtf.cc
    logger/logger.cc
    profiling/profiling_zone.cc
    profiling/profiler.cc
    util/string_utils.cc
    util/geometry_utils.cc
    ../config/transform_config.cc
    ../config/config_registry.cc
)

add_library(pointcaster::core ALIAS pointcaster_core)

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
target_include_directories(pointcaster_core PRIVATE ${ZPP_BITS_INCLUDE_DIRS})
if (NOT WIN32)
  # zpp-bits' inlined templates show up as string overflow warnings in gcc
  set_source_files_properties(point_cloud.cc PROPERTIES
    COMPILE_OPTIONS "-Wno-stringop-overflow"
  )
endif()

find_package(draco CONFIG REQUIRED)
target_link_libraries(pointcaster_core PRIVATE draco::draco)

# shared configs need reflectcpp
find_package(reflectcpp CONFIG REQUIRED)
target_link_libraries(pointcaster_core PRIVATE reflectcpp::reflectcpp)

find_path(BSHOSHANY_THREAD_POOL_INCLUDE_DIRS "BS_thread_pool.hpp")
target_include_directories(pointcaster_core PUBLIC ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS})

find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
target_link_libraries(pointcaster_core PUBLIC spdlog::spdlog fmt::fmt)

find_package(Tracy CONFIG REQUIRED)
target_link_libraries(pointcaster_core PUBLIC Tracy::TracyClient)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(pointcaster_core PUBLIC Eigen3::Eigen)

target_include_directories(pointcaster_core
    ${WARNING_GUARD}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(pointcaster_core
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        AUTOMOC OFF
)

generate_export_header(pointcaster_core
    EXPORT_FILE_NAME
        ${PROJECT_BINARY_DIR}/include/pointcaster/core.h
)

install(TARGETS pointcaster_core
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION lib
  # import libs are link-time only, keep them out of the runtime install
  ARCHIVE DESTINATION lib EXCLUDE_FROM_ALL
)

target_precompile_headers(pointcaster_core PRIVATE
  <string>
  <string_view>
  <vector>
  <memory>
  <mutex>
  <thread>
  <optional>
  <variant>
  <functional>
  <chrono>
  <atomic>
  <filesystem>
  <utility>
  <algorithm>
  <span>
  <cstddef>
  <cstdint>
  <ranges>
  <expected>
  <tracy/Tracy.hpp>
)
