set(DEVICES_PLUGIN_DIR "plugins/devices")

add_subdirectory(devices/orbbec)

# the $CORRADE_PLUGINS list should be populated after adding subdirectories

add_dependencies(pointcaster ${CORRADE_PLUGINS})
