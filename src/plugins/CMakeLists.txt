set(DEVICES_PLUGIN_DIR "plugins/devices")

add_library(pointcaster_plugins
    plugin_loader.cc
)
add_library(pointcaster::plugins ALIAS pointcaster_plugins)

target_link_libraries(pointcaster_plugins PRIVATE
    pointcaster::core 
    Corrade::Main Corrade::PluginManager
)

target_include_directories(pointcaster_plugins PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    find_package(cpplocate CONFIG REQUIRED)
    target_link_libraries(pointcaster_plugins PRIVATE cpplocate::cpplocate)
endif()

# subdirectories are the dynamic corrade plugins
add_subdirectory(devices/orbbec)

# the $CORRADE_PLUGINS list should be populated after adding subdirectories
add_dependencies(pointcaster ${CORRADE_PLUGINS})

target_precompile_headers(pointcaster_plugins PRIVATE
  <string>
  <string_view>
  <vector>
  <memory>
  <filesystem>
  <optional>
  <Corrade/PluginManager/Manager.h>
  <Corrade/PluginManager/AbstractManager.h>
  <Corrade/Containers/StringView.h>
)