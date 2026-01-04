set(ORBBEC_PLUGIN_DIR "plugins/orbbec")

corrade_add_plugin(OrbbecDevice
  ${ORBBEC_PLUGIN_DIR}
  ${ORBBEC_PLUGIN_DIR}
  OrbbecDevice.conf 
  orbbec_device.cc
)
set_target_properties(OrbbecDevice PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${ORBBEC_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${ORBBEC_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${ORBBEC_PLUGIN_DIR}
  BUILD_RPATH   "\$ORIGIN;\$ORIGIN/deps"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/deps"
)

find_package(OrbbecSDK CONFIG REQUIRED)

target_link_libraries(OrbbecDevice PRIVATE 
    Corrade::PluginManager Orbbec::OrbbecSDK
)

add_custom_command(TARGET OrbbecDevice POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ORBBEC_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecDevice.conf"
  "${ORBBEC_PLUGIN_DIR}/OrbbecDevice.conf"
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")

  add_custom_command(TARGET OrbbecDevice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ORBBEC_PLUGIN_DIR}/deps"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/OrbbecSDK.dll"
    "${ORBBEC_PLUGIN_DIR}/deps/OrbbecSDK.dll"
  )

else()

  add_custom_command(TARGET OrbbecDevice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ORBBEC_PLUGIN_DIR}/deps"
    COMMAND ${CMAKE_COMMAND}
    -DINPUT_EXECUTABLE="$<TARGET_FILE:OrbbecDevice>"
    -DOUTPUT_DIRECTORY="${ORBBEC_PLUGIN_DIR}/deps"
    -DVCPKG_INSTALLED_DIR="${VCPKG_INSTALLED_DIR}"
    -DVCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET}"
    -DMOVE_EXISTING_DEPS=ON
    -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeDependencies.cmake"
  )

endif()

install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/${ORBBEC_PLUGIN_DIR}/deps/"
    DESTINATION "${ORBBEC_PLUGIN_DIR}/deps"
)


set(CORRADE_PLUGINS ${CORRADE_PLUGINS} OrbbecDevice PARENT_SCOPE)
