corrade_add_static_plugin(
  PlyDevice
  PlyDevice
  ""
  PlyDevice.conf
  ply_device.cc
  ${CMAKE_SOURCE_DIR}/src/ui/models/device_status.h
)

set_target_properties(PlyDevice PROPERTIES AUTOMOC OFF)

find_package(TBB REQUIRED)

target_link_libraries(PlyDevice PRIVATE
  pointcaster::core
  Corrade::PluginManager
  TBB::tbb
)

find_path(HAPPLY_INCLUDE_DIRS "happly.h")

target_include_directories(PlyDevice 
  PRIVATE 
    "${CMAKE_SOURCE_DIR}/src" 
    "${CMAKE_SOURCE_DIR}/include"
    ${HAPPLY_INCLUDE_DIRS}
)

install(CODE [===[
  file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}/PlyDevice/")
]===])

set(CORRADE_STATIC_PLUGINS ${CORRADE_STATIC_PLUGINS} PlyDevice PARENT_SCOPE)
set(CORRADE_PLUGINS ${CORRADE_PLUGINS} PlyDevice PARENT_SCOPE)