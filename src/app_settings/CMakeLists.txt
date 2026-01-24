include(GenerateExportHeader)

# since plugins can get access to application settings, this module must be a shared lib

add_library(pointcaster_app_settings SHARED
    app_settings.cc
)

add_library(pointcaster::app_settings ALIAS pointcaster_app_settings)

find_package(Qt6 REQUIRED COMPONENTS Core)

find_package(spdlog CONFIG REQUIRED)

target_link_libraries(pointcaster_app_settings PUBLIC
  Qt6::Core
  spdlog::spdlog
)

target_include_directories(pointcaster_app_settings PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)