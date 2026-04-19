set(ORBBEC_PLUGIN_DIR ${DEVICES_PLUGIN_DIR}/orbbec)

corrade_add_plugin(OrbbecDevice
  ${ORBBEC_PLUGIN_DIR}
  ${ORBBEC_PLUGIN_DIR}
  OrbbecDevice.conf
  orbbec_device.cc
  orbbec_context.cc
  ${CMAKE_SOURCE_DIR}/src/ui/models/device_status.h
)

set_target_properties(OrbbecDevice PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${ORBBEC_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${ORBBEC_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${ORBBEC_PLUGIN_DIR}
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/orbbec"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/orbbec"
  AUTOMOC OFF
)

find_package(OrbbecSDK CONFIG REQUIRED)
find_package(TBB REQUIRED)

# TODO
# don't link to this. dynamically link to cuda plugin
# i guess just load headers into this ? 
# AND it needs to be compiled by nvcc
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

target_link_libraries(OrbbecDevice PRIVATE
  pointcaster::core
  pointcaster::metrics
  Corrade::PluginManager
  # ob::OrbbecSDK
  ${OrbbecSDK_LIBS}
  TBB::tbb

  # TODO
  Thrust::Thrust
)

find_path(READERWRITERQUEUE_INCLUDE_DIR "readerwriterqueue/readerwriterqueue.h")

target_include_directories(OrbbecDevice 
  PRIVATE 
    "${CMAKE_SOURCE_DIR}/src" 
    "${CMAKE_SOURCE_DIR}/include"
    "${READERWRITERQUEUE_INCLUDE_DIR}"
    "$ENV{TBB_DIR}/include"
)

add_custom_command(TARGET OrbbecDevice POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ORBBEC_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecDevice.conf"
  "${ORBBEC_PLUGIN_DIR}/OrbbecDevice.conf"
)

set(ORBBEC_INSTALLED_LIB_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(ORBBEC_INSTALLED_LIB_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
endif()

file(GLOB ORBBEC_RUNTIME_DEPENDENCIES "${ORBBEC_INSTALLED_LIB_DIR}/*OrbbecSDK*")

install(FILES ${ORBBEC_RUNTIME_DEPENDENCIES} DESTINATION ${ORBBEC_PLUGIN_DIR})

# install(DIRECTORY "${ORBBEC_INSTALLED_LIB_DIR}/extensions" DESTINATION ${ORBBEC_PLUGIN_DIR})

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
  <libobsensor/h/ObTypes.h>
  <libobsensor/hpp/Context.hpp>
  <libobsensor/hpp/Error.hpp>
  <libobsensor/hpp/Pipeline.hpp>
  <libobsensor/hpp/Utils.hpp>
)
