include(GNUInstallDirs)

qt_standard_project_setup(REQUIRES 6.11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_INSTALL_QMLDIR "qml")
set(QT_QML_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qml")

add_subdirectory(core)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(app_icon_resource "${CMAKE_SOURCE_DIR}/packaging/icon.rc")
  # set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  # adds PDB export to msvc
  add_compile_options("/Zi")
endif()

qt_add_executable(pointcaster
  #WIN32
  pointcaster.cc
  workspace/workspace.cc
  session/session.cc
  workspace/workspace_config.cc
  camera/look_at_camera.cc
  camera/camera_frame.cc
  recorder/session_recorder.cc
  pipeline/concurrent_operator_pipeline.cc
  networking/point_streamer.cc
  networking/osc/osc_receiver.cc
  config/config_registry.cc
  # TODO should probs be part of plugin lib? but other deps cant find it if it is atm
  plugins/operators/operator_host.cc
  ${app_icon_resource}
)

# we set qt automoc off because QObjects we include should be
# rendered inside the ui lib not the exe itself (otherwise Qt MOC runs twice)
set_target_properties(pointcaster PROPERTIES
  AUTOMOC OFF
  ENABLE_EXPORTS ON
)

add_subdirectory(ui)
add_subdirectory(app_settings)
add_subdirectory(metrics)
add_subdirectory(registration)

# corrade lib used for dynamic plugin / lib loading
find_package(Corrade CONFIG REQUIRED Main PluginManager)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

# dynamic runtime loaded dependencies
add_subdirectory(plugins)

#----- Dependencies

find_package(reflectcpp CONFIG REQUIRED)
find_package(liblo CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(PCL CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED
  COMPONENTS
  opencv_core
  opencv_imgproc
)

target_link_libraries(pointcaster PRIVATE
  pointcaster::core
  pointcaster::plugins
  pointcaster::metrics
  pointcaster::registration
  pointcaster::app_settings
  pointcaster::ui
  Corrade::Main
  Corrade::PluginManager
  reflectcpp::reflectcpp
  liblo::liblo
  TBB::tbb
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

target_include_directories(pointcaster PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  ${OpenCV_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
)

target_compile_definitions(pointcaster PRIVATE POINTCASTER_BUILD)

target_precompile_headers(pointcaster PRIVATE
  <string>
  <string_view>
  <vector>
  <tuple>
  <memory>
  <mutex>
  <thread>
  <chrono>
  <Corrade/PluginManager/Manager.h>
  <Corrade/PluginManager/AbstractManager.h>
  <Corrade/Containers/StringView.h>
  <QCoreApplication>
  <QUrl>
  <Eigen/Core>
  <Eigen/Dense>
  <pcl/point_cloud.h>
  <pcl/point_types.h>
  <pcl/registration/transformation_estimation_svd.h>
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_custom_command(TARGET pointcaster POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    -DROOT=$<TARGET_FILE_DIR:pointcaster>
    -P ${CMAKE_SOURCE_DIR}/cmake/PruneMsvcDependencies.cmake
  )
endif()

install(TARGETS pointcaster
  BUNDLE DESTINATION .
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
