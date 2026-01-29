corrade_add_plugin(OrbbecDevice
  ${DEVICES_PLUGIN_DIR}
  ${DEVICES_PLUGIN_DIR}
  OrbbecDevice.conf
  orbbec_device.cc
  orbbec_context.cc
  ${CMAKE_SOURCE_DIR}/src/ui/models/device_status.h
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
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/orbbec"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/orbbec"
)

find_package(OrbbecSDK CONFIG REQUIRED)
find_package(unofficial-concurrentqueue CONFIG REQUIRED)

target_link_libraries(OrbbecDevice PRIVATE
  pointcaster::core
  pointcaster::metrics
  Corrade::PluginManager
  ob::OrbbecSDK
  unofficial::concurrentqueue::concurrentqueue
  # TODO: this gui inclusion is wierd, it seems to require it for 
  # deserialization of pc::float3 into QVector3D?
  Qt6::Gui 
)

target_include_directories(OrbbecDevice PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/include")

add_custom_command(TARGET OrbbecDevice POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DEVICES_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecDevice.conf"
  "${DEVICES_PLUGIN_DIR}/OrbbecDevice.conf"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")

  # add_custom_command(TARGET OrbbecDevice POST_BUILD
  #   COMMAND ${CMAKE_COMMAND} -E make_directory "${DEVICES_PLUGIN_DIR}/extensions"
  #   COMMAND ${CMAKE_COMMAND} -E copy_directory
  #   "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/extensions"
  #   "${DEVICES_PLUGIN_DIR}/extensions"
  # )
  # message(FATAL_ERROR ${CMAKE_CURRENT_BINARY_DIR})

  install(DIRECTORY
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/extensions"
    DESTINATION "${DEVICES_PLUGIN_DIR}"
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

  add_custom_command(TARGET OrbbecDevice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DEVICES_PLUGIN_DIR}/orbbec/extensions"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/extensions"
    "${DEVICES_PLUGIN_DIR}/orbbec/extensions"
  )

  install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/${DEVICES_PLUGIN_DIR}/orbbec/"
    DESTINATION "${DEVICES_PLUGIN_DIR}/orbbec"
  )

endif()


set(CORRADE_PLUGINS ${CORRADE_PLUGINS} OrbbecDevice PARENT_SCOPE)

target_precompile_headers(OrbbecDevice PRIVATE
  <string>
  <string_view>
  <vector>
  <memory>
  <mutex>
  <thread>
  <chrono>
  <atomic>
  <optional>
  <libobsensor/ObSensor.hpp>
  <libobsensor/hpp/Context.hpp>
  <QObject>
  <QString>
  <QVariant>
)