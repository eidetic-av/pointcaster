include(GNUInstallDirs)

qt_standard_project_setup(REQUIRES 6.11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_INSTALL_QMLDIR "qml")
set(QT_QML_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qml")

add_subdirectory(core)

qt_add_executable(pointcaster 
    pointcaster.cc
    workspace/workspace.cc
    workspace/workspace_config.cc
)

add_subdirectory(ui)
add_subdirectory(app_settings)
add_subdirectory(metrics)

# corrade lib used for dynamic plugin / lib loading
find_package(Corrade CONFIG REQUIRED Main PluginManager)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

# dynamic runtime loaded dependencies
add_subdirectory(plugins)

#----- Dependencies

# - reflectcpp is used for de/serialization of workspace configuration,
find_package(reflectcpp CONFIG REQUIRED)

# TODO this is probably not what we want re: qml?
# - application settings initialisation requires Qml dependency linked in the main application
find_package(Qt6 REQUIRED COMPONENTS Qml)

target_link_libraries(pointcaster PRIVATE 
  pointcaster::core 
  pointcaster::plugins
  pointcaster::metrics
  pointcaster::app_settings
  pointcaster::ui
  Corrade::Main 
  Corrade::PluginManager
  reflectcpp::reflectcpp
  Qt6::Qml
)

target_include_directories(pointcaster PRIVATE "${CMAKE_SOURCE_DIR}/src")

target_precompile_headers(pointcaster PRIVATE
  <string>
  <string_view>
  <vector>
  <memory>
  <mutex>
  <thread>
  <chrono>
  <Corrade/PluginManager/Manager.h>
  <Corrade/PluginManager/AbstractManager.h>
  <Corrade/Containers/StringView.h>
  <QCoreApplication>
  <QUrl>
)

# we set qt automoc off because QObjects we include should be
# rendered inside the ui lib not the exe itself (otherwise Qt MOC runs twice)
set_target_properties(pointcaster PROPERTIES
  AUTOMOC OFF
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_custom_command(TARGET pointcaster POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    -DROOT=$<TARGET_FILE_DIR:pointcaster>
    -P ${CMAKE_SOURCE_DIR}/cmake/PruneMsvcDependencies.cmake
  )
endif()

install(TARGETS pointcaster
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
