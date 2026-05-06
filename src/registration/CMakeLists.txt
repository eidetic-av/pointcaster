include(GenerateExportHeader)

add_library(pointcaster_registration
    registration.cc
)

add_library(pointcaster::registration ALIAS pointcaster_registration)

find_package(PCL CONFIG REQUIRED)

target_link_libraries(pointcaster_registration PRIVATE
  pointcaster::core
  ${PCL_LIBRARIES}
)

target_include_directories(pointcaster_registration PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)

install(TARGETS pointcaster_registration
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib                                                            )
