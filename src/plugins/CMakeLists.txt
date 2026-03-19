set(DEVICES_PLUGIN_DIR "plugins/devices")

add_library(pointcaster_plugins
    plugin_loader.cc
)
add_library(pointcaster::plugins ALIAS pointcaster_plugins)

target_link_libraries(pointcaster_plugins PRIVATE
    pointcaster::core 
    Corrade::Main Corrade::PluginManager
)

target_include_directories(pointcaster_plugins PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/include
)

find_package(cpplocate CONFIG REQUIRED)
target_link_libraries(pointcaster_plugins PRIVATE cpplocate::cpplocate)

# subdirectories are the corrade plugins
add_subdirectory(devices/orbbec)
add_subdirectory(devices/null)

# the $CORRADE_PLUGINS and $CORRADE_STATIC_PLUGINS lists should 
# be populated after adding plugin subdirectories
add_dependencies(pointcaster ${CORRADE_PLUGINS})
target_link_libraries(pointcaster_plugins PRIVATE ${CORRADE_STATIC_PLUGINS})

# TODO for some reason our static libs .a are being deployed with pointcaster install...
# not sure of the CMAKE switch to turn this off... 
# so after install step, just delete the installed files
install(CODE [===[
  foreach(plugin ${CORRADE_STATIC_PLUGINS})
    file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}/${plugin}/")
  endforeach()
]===])

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