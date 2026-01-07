corrade_add_plugin(OrbbecDevice
  ${DEVICES_PLUGIN_DIR}
  ${DEVICES_PLUGIN_DIR}
  OrbbecDevice.conf 
    orbbec_device.cc
    orbbec_context.cc
)

add_qml_adapters(
    TARGETS
        pointcaster
        OrbbecDevice
    CONFIG_INPUTS
        orbbec_device_config.h
)

set_target_properties(OrbbecDevice PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${DEVICES_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${DEVICES_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${DEVICES_PLUGIN_DIR}
  BUILD_RPATH   "\$ORIGIN;\$ORIGIN/orbbec"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/orbbec"
)

find_package(OrbbecSDK CONFIG REQUIRED)

target_link_libraries(OrbbecDevice PRIVATE
  pointcaster::core 
  Corrade::PluginManager
  Orbbec::OrbbecSDK
)

# for pointcaster::core headers in public include dir
target_include_directories(OrbbecDevice PRIVATE "${CMAKE_SOURCE_DIR}/include")

add_custom_command(TARGET OrbbecDevice POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DEVICES_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecDevice.conf"
  "${DEVICES_PLUGIN_DIR}/OrbbecDevice.conf"
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")

  add_custom_command(TARGET OrbbecDevice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DEVICES_PLUGIN_DIR}/orbbec"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/OrbbecSDK.dll"
    "${DEVICES_PLUGIN_DIR}/orbbec/OrbbecSDK.dll"
  )

else()

  add_custom_command(TARGET OrbbecDevice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DEVICES_PLUGIN_DIR}/orbbec"
    COMMAND ${CMAKE_COMMAND}
    -DINPUT_EXECUTABLE="$<TARGET_FILE:OrbbecDevice>"
    -DOUTPUT_DIRECTORY="${DEVICES_PLUGIN_DIR}/orbbec"
    -DVCPKG_INSTALLED_DIR="${VCPKG_INSTALLED_DIR}"
    -DVCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET}"
    -DMOVE_EXISTING_DEPS=ON
    -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeDependencies.cmake"
  )

endif()

install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/${DEVICES_PLUGIN_DIR}/orbbec/"
    DESTINATION "${DEVICES_PLUGIN_DIR}/orbbec"
)


set(CORRADE_PLUGINS ${CORRADE_PLUGINS} OrbbecDevice PARENT_SCOPE)