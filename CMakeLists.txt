cmake_minimum_required(VERSION 3.22.0)
project(bob-pointcaster LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(WITH_K4A "Enable Azure Kinect support" ON)

# ----- imgui lib -----

find_package(imgui CONFIG REQUIRED)
add_library(ImGui::ImGui ALIAS imgui::imgui)
list(APPEND LINK_LIBS imgui::imgui)

# ----- Eigen lib for geometry -----

find_package(Eigen3 CONFIG REQUIRED)
list(APPEND LINK_LIBS Eigen3::Eigen)

# ----- Point cloud streaming libs -----

find_package(bob-pointclouds CONFIG REQUIRED)
list(APPEND LINK_LIBS bob::pointclouds)

find_package(draco CONFIG REQUIRED)
list(APPEND LINK_LIBS draco::draco)

find_package(ZeroMQ CONFIG REQUIRED)
# zmq needs to be a shared lib on windows
if (WIN32)
  list(APPEND LINK_LIBS libzmq)
  # and it needs to include the system bcrypt
  set(PLATFORM_LIBS ${PLATFORM_LIBS} bcrypt)
else()
  list(APPEND LINK_LIBS libzmq-static)
endif()

# ----- Magnum 3D engine -----

find_package(Corrade CONFIG REQUIRED Main)
list(APPEND LINK_LIBS Corrade::Main)

find_package(Magnum CONFIG REQUIRED
  GL MeshTools Primitives SceneGraph
  Shaders Trade Sdl2Application)
list(APPEND LINK_LIBS
  Magnum::Application Magnum::GL
  Magnum::Magnum Magnum::MeshTools
  Magnum::Primitives Magnum::SceneGraph
  Magnum::Shaders Magnum::Trade)

find_package(MagnumIntegration CONFIG REQUIRED
  ImGui Eigen)
list(APPEND LINK_LIBS
  MagnumIntegration::ImGui MagnumIntegration::Eigen)

# ----- Sensor driver libraries -----

find_package(libusb CONFIG REQUIRED)
list(APPEND LINK_LIBS ${LIBUSB_LIBRARIES})
list(APPEND INCLUDE_DIRS ${LIBUSB_INCLUDE_DIRS})

# Azure Kinect
if (WITH_K4A)

  find_package(k4a CONFIG REQUIRED)
  list(APPEND OPTIONAL_LIBS k4a::k4a)

  list(APPEND OPTIONAL_SRC
    src/devices/k4a/k4a_driver.cu
    src/devices/k4a/k4a_device.cc)

  find_package(k4abt REQUIRED)
  list(APPEND OPTIONAL_LIBS k4a::k4abt)
  
  add_compile_definitions(WITH_K4A)

  if (WIN32)
    add_compile_definitions(NOMINMAX)
  endif()

endif()

# ----- OpenCV -----

find_package(OpenCV CONFIG REQUIRED)
list(APPEND LINK_LIBS ${OpenCV_LIBS})

# ----- Utility libs -----

find_package(fmt CONFIG REQUIRED)
list(APPEND LINK_LIBS fmt::fmt)

find_package(spdlog CONFIG REQUIRED)
list(APPEND LINK_LIBS spdlog::spdlog_header_only)

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
list(APPEND INCLUDE_DIRS ${ZPP_BITS_INCLUDE_DIRS})

# ----- Application setup -----

set(SOURCE_FILES
  src/pointcaster.cc
  src/devices/device.cc
  src/devices/usb.cc
  src/camera/camera_controller.cc
  src/radio.cc
  src/snapshots.cc
  src/point_cloud_renderer.cc
  src/sphere_renderer.cc
  src/shaders/particle_sphere.cc
  src/gui_helpers.cc
  src/path.cc
  src/uuid.cc
  src/wtf/wtf.cc
  ${OPTIONAL_SRC})

corrade_add_resource(RESOURCES resources.conf)

add_executable(pointcaster ${SOURCE_FILES} ${RESOURCES})
set_target_properties(pointcaster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(pointcaster PRIVATE
    ${INCLUDE_DIRS} ${OPTIONAL_INCLUDE_DIRS} 
    ${PLATFORM_INCLUDE_DIRS})

target_link_libraries(pointcaster PRIVATE 
    ${LINK_LIBS} ${OPTIONAL_LIBS} ${PLATFORM_LIBS})

# ----- Post-build actions -----

if (WIN32 AND WITH_K4A)
  # shared k4abt dll must be moved into executable directory manually
  add_custom_command(TARGET pointcaster POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${K4ABT_ROOT}/sdk/windows-desktop/amd64/release/bin/k4abt.dll
    $<TARGET_FILE_DIR:pointcaster>)
endif()

if (NOT WIN32)
  # copy compile_commands.json for lsp integration on linux
  add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR})
endif()
