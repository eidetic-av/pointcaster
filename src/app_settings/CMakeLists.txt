include(GenerateExportHeader)

add_library(pointcaster_app_settings
    app_settings.cc
)

add_library(pointcaster::app_settings ALIAS pointcaster_app_settings)

find_package(Qt6 REQUIRED COMPONENTS Core)

target_link_libraries(pointcaster_app_settings PUBLIC
  Qt6::Core
)

target_include_directories(pointcaster_app_settings PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)