cmake_minimum_required(VERSION 3.22.0)
project(bob-pointcaster)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add vcpkg ports from this repo
set(VCPKG_OVERLAY_PORTS "${VCPKG_OVERLAY_PORTS};${CMAKE_CURRENT_LIST_DIR}/ports")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")

option(WITH_K4A "Enable Azure Kinect support" ON)
option(WITH_RS2 "Enable Realsense device support" OFF)
option(WITH_SKYBRIDGE "Enable global sessions using the BoB Skybridge server" OFF)

find_package(spdlog CONFIG REQUIRED)
find_package(EBML CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

find_package(ZeroMQ CONFIG REQUIRED)
set(ZMQ_LIB libzmq-static)

find_package(imgui CONFIG REQUIRED)
add_library(ImGui::ImGui ALIAS imgui::imgui)
find_package(Corrade CONFIG REQUIRED Main)
find_package(Magnum CONFIG REQUIRED
  GL MeshTools Primitives SceneGraph
  Shaders Sdl2Application)
find_package(MagnumIntegration CONFIG REQUIRED ImGui)
corrade_add_resource(RESOURCES resources.conf)

find_package(libusb CONFIG REQUIRED)

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
# find_package(bob-pointclouds CONFIG REQUIRED)
include_directories(~/bob/bob-pointclouds/include)
link_directories(~/bob/bob-pointclouds/build)

include(FetchContent)
FetchContent_Declare(yaclib
  GIT_REPOSITORY https://github.com/YACLib/YACLib.git
  GIT_TAG main
  )
FetchContent_MakeAvailable(yaclib)
link_libraries(yaclib)


if (WITH_K4A)
  find_package(k4a CONFIG REQUIRED)
  set(OPTIONAL_SRC ${OPTIONAL_SRC}
      src/devices/k4a/k4a_driver.cc 
      src/devices/k4a/k4a_device.cc)
  set (OPTIONAL_LIBS ${OPTIONAL_LIBS} k4a::k4a)
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

if (WITH_SKYBRIDGE)
  find_package(ixwebsocket CONFIG REQUIRED)
  set(OPTIONAL_SRC ${OPTIONAL_SRC} src/skybridge.cc)
  set(OPTIONAL_LIBS ${OPTIONAL_LIBS} ixwebsocket::ixwebsocket)
  add_compile_definitions(WITH_SKYBRIDGE)
endif()

set(SOURCE_FILES
  src/pointcaster.cc
  src/radio.cc
  src/gui_helpers.cc
  src/point_cloud_renderer.cc
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

target_include_directories(pointcaster PRIVATE
  ${LIBUSB_INCLUDE_DIRS} ${ZPP_BITS_INCLUDE_DIRS})

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
  MagnumIntegration::ImGui
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
