set(ORBBEC_PLUGIN_DIR ${DEVICES_PLUGIN_DIR}/orbbec)


find_package(OrbbecSDK CONFIG REQUIRED)
find_package(OrbbecSDKv1 CONFIG REQUIRED)

find_package(TBB REQUIRED)
find_package(cpplocate CONFIG REQUIRED)

# TODO
# don't link to this. dynamically link to cuda plugin
# i guess just load headers into this ?
# AND it needs to be compiled by nvcc
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

find_path(READERWRITERQUEUE_INCLUDE_DIR "readerwriterqueue/readerwriterqueue.h")

set(ORBBEC_PLUGIN_SOURCES
  orbbec_device.cc
  orbbec_context.cc
  ${CMAKE_SOURCE_DIR}/src/ui/models/enum_adapters.h
)

set(ORBBEC_INSTALLED_LIB_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib")
set(ORBBEC_INSTALLED_BIN_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")

# adds one plugin variant target compiled against the given sdk target
function(add_orbbec_plugin_variant target sdk_target sdk_major)
  set(variant_dir "${ORBBEC_PLUGIN_DIR}/sdk-v${sdk_major}")

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_library(${target} SHARED ${ORBBEC_PLUGIN_SOURCES})
  else()
    add_library(${target} MODULE ${ORBBEC_PLUGIN_SOURCES})
  endif()

  target_compile_definitions(${target} PRIVATE
    CORRADE_DYNAMIC_PLUGIN
    POINTCASTER_ORBBEC_SDK_VERSION=${sdk_major}
  )

  set_target_properties(${target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME OrbbecDevice
    RUNTIME_OUTPUT_DIRECTORY ${variant_dir}
    LIBRARY_OUTPUT_DIRECTORY ${variant_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${variant_dir}
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
    AUTOMOC OFF
  )

  target_link_libraries(${target} PRIVATE
    pointcaster::core
    pointcaster::pipeline
    pointcaster::metrics
    pointcaster::plugins
    Corrade::PluginManager
    ${sdk_target}
    cpplocate::cpplocate
  )

  target_include_directories(${target} BEFORE PRIVATE
    "$<TARGET_PROPERTY:${sdk_target},INTERFACE_INCLUDE_DIRECTORIES>"
  )

  target_include_directories(${target}
    PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
    "${READERWRITERQUEUE_INCLUDE_DIR}"
    "$ENV{TBB_DIR}/include"
  )

  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${variant_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecDevice.conf"
    "${variant_dir}/OrbbecDevice.conf"
  )

  install(TARGETS ${target}
    RUNTIME DESTINATION ${variant_dir}
    LIBRARY DESTINATION ${variant_dir}
  )
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecDevice.conf"
    DESTINATION ${variant_dir}
  )

  target_precompile_headers(${target} PRIVATE
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
endfunction()

# copies the given runtime files beside the variant's plugin binary
function(add_orbbec_variant_runtime_files target sdk_major)
  set(variant_dir "${ORBBEC_PLUGIN_DIR}/sdk-v${sdk_major}")

  set(runtime_files "")
  foreach(runtime_file IN LISTS ARGN)
    if(NOT IS_DIRECTORY "${runtime_file}")
      list(APPEND runtime_files "${runtime_file}")
    endif()
  endforeach()

  foreach(runtime_file IN LISTS runtime_files)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${runtime_file}" "${variant_dir}"
    )
  endforeach()
  install(FILES ${runtime_files} DESTINATION ${variant_dir})
endfunction()

# --- sdk v2 variant

add_orbbec_plugin_variant(OrbbecDevice ob::OrbbecSDK 2)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  file(GLOB ORBBEC_V2_RUNTIME_DEPENDENCIES "${ORBBEC_INSTALLED_BIN_DIR}/*OrbbecSDK*")
  set(ORBBEC_V2_EXTENSIONS_SOURCE_DIR "${ORBBEC_INSTALLED_BIN_DIR}/extensions")
else()
  file(GLOB ORBBEC_V2_RUNTIME_DEPENDENCIES "${ORBBEC_INSTALLED_LIB_DIR}/libOrbbecSDK*")
  set(ORBBEC_V2_EXTENSIONS_SOURCE_DIR "${ORBBEC_INSTALLED_LIB_DIR}/extensions")
endif()
add_orbbec_variant_runtime_files(OrbbecDevice 2 ${ORBBEC_V2_RUNTIME_DEPENDENCIES})

# sdk v2 loads its 'extensions' libs from a directory beside the plugin at
# runtime (the plugin points the sdk there via setExtensionsDirectory)
add_custom_command(TARGET OrbbecDevice POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${ORBBEC_V2_EXTENSIONS_SOURCE_DIR}"
  "${ORBBEC_PLUGIN_DIR}/sdk-v2/extensions"
)
install(DIRECTORY "${ORBBEC_V2_EXTENSIONS_SOURCE_DIR}"
  DESTINATION "${ORBBEC_PLUGIN_DIR}/sdk-v2"
)

# the plugin's preferences qml ships beside its binaries
add_custom_command(TARGET OrbbecDevice POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecSettingsPage.qml"
  "${ORBBEC_PLUGIN_DIR}/OrbbecSettingsPage.qml"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/OrbbecSettingsPage.qml"
  DESTINATION ${ORBBEC_PLUGIN_DIR}
)

# --- sdk v1 variant

add_orbbec_plugin_variant(OrbbecDeviceSdk1 ob::OrbbecSDKv1 1)

# copy sdk dependencies
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  file(GLOB ORBBEC_V1_RUNTIME_DEPENDENCIES "${ORBBEC_INSTALLED_BIN_DIR}/orbbecsdk-v1/*.dll")
else()
  file(GLOB ORBBEC_V1_RUNTIME_DEPENDENCIES "${ORBBEC_INSTALLED_LIB_DIR}/orbbecsdk-v1/*")
endif()
add_orbbec_variant_runtime_files(OrbbecDeviceSdk1 1 ${ORBBEC_V1_RUNTIME_DEPENDENCIES})

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} OrbbecDevice OrbbecDeviceSdk1 PARENT_SCOPE)
