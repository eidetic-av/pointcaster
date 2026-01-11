set(DEVICES_PLUGIN_DIR "plugins/devices")

find_package(cpplocate CONFIG REQUIRED)

add_library(pointcaster_plugins
    plugin_loader.cc
)
add_library(pointcaster::plugins ALIAS pointcaster_plugins)

target_link_libraries(pointcaster_plugins PRIVATE
    pointcaster::core cpplocate::cpplocate
    Corrade::Main Corrade::PluginManager
)

# subdirectories are the dynamic corrade plugins
add_subdirectory(devices/orbbec)

# the $CORRADE_PLUGINS list should be populated after adding subdirectories
add_dependencies(pointcaster ${CORRADE_PLUGINS})