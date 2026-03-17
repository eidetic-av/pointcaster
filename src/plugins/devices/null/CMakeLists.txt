
corrade_add_static_plugin(NullDevice
  NullDevice
  ""
  NullDevice.conf
  null_device.cc
  ${CMAKE_SOURCE_DIR}/src/ui/models/device_status.h
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

# TODO for some reason our static lib .a is being deployed with pointcaster
# not sure of the CMAKE switch to turn this off though... 
# so after install step, just delete the installed files
# AND I don't think the following code is garunteed to run after the archive
# dir actually exists
install(CODE [===[
  file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}/NullDevice/")
]===])

set(CORRADE_STATIC_PLUGINS ${CORRADE_STATIC_PLUGINS} NullDevice PARENT_SCOPE)
set(CORRADE_PLUGINS ${CORRADE_PLUGINS} NullDevice PARENT_SCOPE)