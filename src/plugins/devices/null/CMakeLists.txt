# TODO this should be a static plugin
# then we wouldnt need to 'install' some plugin for code that does nothing
# ... but corrade_add_static_plugin introduced more linker errors so cbf for now

# corrade_add_plugin(NullDevice
#   ${DEVICES_PLUGIN_DIR}
#   ${DEVICES_PLUGIN_DIR}
#   NullDevice.conf
#   null_device.cc
#   ${CMAKE_SOURCE_DIR}/src/ui/models/device_status.h
# )

corrade_add_static_plugin(NullDevice
  NullDevice
  ""
  NullDevice.conf
  null_device.cc
  ${CMAKE_SOURCE_DIR}/src/ui/models/device_status.h
)

set_target_properties(NullDevice PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${DEVICES_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${DEVICES_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${DEVICES_PLUGIN_DIR}
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/null"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/null"
)

target_link_libraries(NullDevice PRIVATE
  pointcaster::core
  Corrade::PluginManager
  # TODO: this qt inclusion is wierd...
  Qt6::Core 
)

target_include_directories(NullDevice 
  PRIVATE 
    "${CMAKE_SOURCE_DIR}/src" 
    "${CMAKE_SOURCE_DIR}/include"
)

# add_custom_command(TARGET NullDevice POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E make_directory "${DEVICES_PLUGIN_DIR}"
#   COMMAND ${CMAKE_COMMAND} -E copy_if_different
#   "${CMAKE_CURRENT_SOURCE_DIR}/NullDevice.conf"
#   "${DEVICES_PLUGIN_DIR}/NullDevice.conf"
# )

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} NullDevice PARENT_SCOPE)
