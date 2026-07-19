# operator-hosting & pipeline infrastructure shared by both the main executable
# and pointcaster_plugins
find_package(Corrade CONFIG REQUIRED PluginManager)
find_package(TBB REQUIRED)
find_package(cpplocate CONFIG REQUIRED)

add_library(pointcaster_pipeline STATIC
  concurrent_operator_pipeline.cc
  ${CMAKE_SOURCE_DIR}/src/plugins/operators/operator_host.cc
)
add_library(pointcaster::pipeline ALIAS pointcaster_pipeline)

set_target_properties(pointcaster_pipeline PROPERTIES
  AUTOMOC OFF
)

target_link_libraries(pointcaster_pipeline PUBLIC
  pointcaster::core
  Corrade::PluginManager
  TBB::tbb
  cpplocate::cpplocate
)

target_include_directories(pointcaster_pipeline PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
)
