cmake_minimum_required(VERSION 3.22.0)
project(bob-pointcaster LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add vcpkg ports from this repo
set(VCPKG_OVERLAY_PORTS "${VCPKG_OVERLAY_PORTS};${CMAKE_CURRENT_LIST_DIR}/ports")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")

option(WITH_K4A "Enable Azure Kinect support" ON)
option(WITH_RS2 "Enable Realsense device support" OFF)
option(WITH_SKYBRIDGE "Enable global sessions using the BoB Skybridge server" OFF)
option(WITH_PLY_PLAYBACK "Enable .ply file pointcloud loading and playback" ON)

find_package(spdlog CONFIG REQUIRED)
find_package(EBML CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

find_package(ZeroMQ CONFIG REQUIRED)
set(ZMQ_LIB libzmq-static)

find_package(imgui CONFIG REQUIRED)
add_library(ImGui::ImGui ALIAS imgui::imgui)

find_package(Eigen3 CONFIG REQUIRED)

# Corrade won't build on windows without the following flag
if (WIN32)
    set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
endif()
find_package(Corrade CONFIG REQUIRED Main)
find_package(Magnum CONFIG REQUIRED
  GL MeshTools Primitives SceneGraph
  Shaders Trade Sdl2Application)
find_package(MagnumIntegration CONFIG REQUIRED ImGui Eigen)
corrade_add_resource(RESOURCES resources.conf)

find_package(libusb CONFIG REQUIRED)

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
# find_package(bob-pointclouds CONFIG REQUIRED)
include_directories(~/bob/bob-pointclouds/include)
link_directories(~/bob/bob-pointclouds/build)

if (WITH_K4A)
  find_package(k4a CONFIG REQUIRED)
  set(OPTIONAL_SRC ${OPTIONAL_SRC}
      src/devices/k4a/k4a_driver.cu
      src/devices/k4a/k4a_device.cc)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} k4a::k4a)

  # TODO create a port for using k4abt
  find_package(k4abt CONFIG REQUIRED)
  set(OPTIONAL_LIBS ${OPTIONAL_LIBS} k4abt)
  
  add_compile_definitions(WITH_K4A)
endif()

if (WITH_RS2)
  find_package(realsense2 CONFIG REQUIRED)
  set(OPTIONAL_SRC ${OPTIONAL_SRC}
      src/devices/rs2/rs2_driver.cc 
      src/devices/rs2/rs2_device.cc)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} realsense2::realsense2)
  add_compile_definitions(WITH_RS2)
endif()

if (WITH_PLY_PLAYBACK)
    find_path(HAPPLY_INCLUDE_DIRS "happly.h")
    set(OPTIONAL_INCLUDE_DIRS ${OPTIONAL_INCLUDE_DIRS} ${HAPPY_INCLUDE_DIRS})
    find_package(unofficial-nativefiledialog CONFIG REQUIRED)
    set(OPTIONAL_LIBS ${OPTIONAL_LIBS} unofficial::nativefiledialog::nfd)
endif()

if (WITH_SKYBRIDGE)
  find_package(ixwebsocket CONFIG REQUIRED)
  set(OPTIONAL_SRC ${OPTIONAL_SRC} src/skybridge.cc)
  set(OPTIONAL_LIBS ${OPTIONAL_LIBS} ixwebsocket::ixwebsocket)
  add_compile_definitions(WITH_SKYBRIDGE)
endif()

set(SOURCE_FILES
  src/pointcaster.cc
  src/camera_controller.cc
  src/radio.cc
  src/devices/device.cc
  src/gui_helpers.cc
  src/point_cloud_renderer.cc
  src/sphere_renderer.cc
  src/shaders/particle_sphere.cc
  src/devices/usb.cc
  ${OPTIONAL_SRC})

if (WIN32)
  # have to link dynamically to zeromq on windows
  set(ZMQ_LIB libzmq)
  # include bcrypt from the Windows SDK
  set(PLATFORM_LIBS ${PLATFORM_LIBS} bcrypt)
endif()

add_executable(pointcaster ${SOURCE_FILES} ${RESOURCES})

set_target_properties(pointcaster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(pointcaster PRIVATE
  ${LIBUSB_INCLUDE_DIRS} ${ZPP_BITS_INCLUDE_DIRS}
  ${OPTIONAL_INCLUDE_DIRS})

target_link_libraries(pointcaster PRIVATE ${PLATFORM_LIBS}
  imgui::imgui #libremidi 
  Corrade::Main
  Magnum::Application
  Magnum::GL
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::Primitives
  Magnum::SceneGraph
  Magnum::Shaders
  Magnum::Trade
  MagnumIntegration::ImGui
  MagnumIntegration::Eigen
  Eigen3::Eigen
  spdlog::spdlog EBML::ebml
  # ${BOB_POINTCLOUDS_LIB}
  pointclouds
  ${LIBUSB_LIBRARIES} ${ZMQ_LIB}
  ${OPTIONAL_LIBS} ${PLATFORM_LIBS})

# copy compile_commands.json for lsp integration 
# (assume we don't need this on Windows)
if (NOT WIN32)
  add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR})
endif()
