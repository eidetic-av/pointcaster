include(GenerateExportHeader)

# metrics lib uses a singleton to publish all metrics on one http server,
# this means if we want plugins and other modules to use it,
# it must be a shared library for that single instance 

add_library(pointcaster_metrics SHARED
    prometheus_server.cc
)

add_library(pointcaster::metrics ALIAS pointcaster_metrics)

# - prometheus gathers metrics pointcaster sessions can expose 
# (e.g. connected devices, health, framerate, scene density)
find_package(prometheus-cpp CONFIG REQUIRED)

# lock free queues
find_package(unofficial-concurrentqueue CONFIG REQUIRED)

target_link_libraries(pointcaster_metrics PRIVATE
  pointcaster::core
  pointcaster::app_settings
  prometheus-cpp::core 
  prometheus-cpp::pull
  unofficial::concurrentqueue::concurrentqueue
)

target_compile_definitions(pointcaster_metrics PRIVATE METRICS_DLL)

target_include_directories(pointcaster_metrics PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)

install(TARGETS pointcaster_metrics
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib                                                            )
