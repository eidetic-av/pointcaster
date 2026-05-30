corrade_add_static_plugin(
  NullDevice
  NullDevice
  ""
  NullDevice.conf
  null_device.cc
  ${CMAKE_SOURCE_DIR}/src/ui/models/device_status.h
)

set_target_properties(NullDevice PROPERTIES AUTOMOC OFF)

# TODO ??
find_package(TBB REQUIRED)

target_link_libraries(NullDevice PRIVATE
  pointcaster::core
  Corrade::PluginManager
  #
  TBB::tbb
)

target_include_directories(NullDevice
  PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
  ${TBB_INCLUDE_DIRS}
)

install(CODE [===[
  file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}/NullDevice/")
]===])

set(CORRADE_STATIC_PLUGINS ${CORRADE_STATIC_PLUGINS} NullDevice PARENT_SCOPE)
set(CORRADE_PLUGINS ${CORRADE_PLUGINS} NullDevice PARENT_SCOPE)
