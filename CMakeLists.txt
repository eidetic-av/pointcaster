cmake_minimum_required(VERSION 3.22.0)
project(bob-pointcaster)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_K4A "Build with Azure Kinect support" ON)
option(WITH_K4W2 "Build with Kinect for Windows v2 support" ON)
option(WITH_RS2 "Build with Realsense device support" ON)

find_package(spdlog CONFIG REQUIRED)
find_package(EBML CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

find_package(ZeroMQ CONFIG REQUIRED)
find_package(ixwebsocket CONFIG REQUIRED)

find_package(imgui CONFIG REQUIRED)
add_library(ImGui::ImGui ALIAS imgui::imgui)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
find_package(Corrade CONFIG REQUIRED Main)
find_package(Magnum CONFIG REQUIRED
  GL MeshTools Primitives SceneGraph
  Shaders Sdl2Application)
find_package(MagnumIntegration CONFIG REQUIRED ImGui)
corrade_add_resource(RESOURCES resources.conf)

find_package(libusb CONFIG REQUIRED)

if (WITH_K4A)
  find_package(k4a CONFIG REQUIRED)
  set(DEVICES_SRC ${DEVICES_SRC}
      src/devices/k4a/k4a_driver.cc 
      src/devices/k4a/k4a_device.cc)
  set (DEVICES_LIBS ${DEVICES_LIBS} k4a::k4a)
  add_compile_definitions(WITH_K4A)
endif()

if (WITH_K4W2)
  find_package(unofficial-kinectsdk2 CONFIG REQUIRED)
  set(DEVICES_SRC ${DEVICES_SRC}
      src/devices/k4w2/k4w2_driver.cc 
      src/devices/k4w2/k4w2_device.cc)
  set (DEVICES_LIBS ${DEVICES_LIBS} unofficial::kinectsdk2::kinectsdk2)
  add_compile_definitions(WITH_K4W2)
endif()

if (WITH_RS2)
  find_package(realsense2 CONFIG REQUIRED)
  set(DEVICES_SRC ${DEVICES_SRC}
      src/devices/rs2/rs2_driver.cc 
      src/devices/rs2/rs2_device.cc)
  set (DEVICES_LIBS ${DEVICES_LIBS} realsense2::realsense2)
  add_compile_definitions(WITH_RS2)
endif()

set(SOURCE_FILES
  src/pointcaster.cc src/point_cloud.cc
  src/skybridge.cc src/gui_helpers.cc
  src/shaders/particle_sphere.cc
  src/devices/usb.cc ${DEVICES_SRC})

set(ZMQ_LIB libzmq-static)
if (WIN32)
  # have to link dynamically to zeromq on windows
  set(ZMQ_LIB libzmq)
  # include bcrypt from the Windows SDK
  set(PLATFORM_LIBS ${PLATFORM_LIBS} bcrypt)
endif()

add_executable(pointcaster ${SOURCE_FILES} ${RESOURCES})

target_include_directories(pointcaster PRIVATE
  ${LIBUSB_INCLUDE_DIRS})

target_link_libraries(pointcaster PRIVATE ${PLATFORM_LIBS}
  imgui::imgui #libremidi 
  ixwebsocket::ixwebsocket
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
  ${LIBUSB_LIBRARIES} ${ZMQ_LIB}
  ${DEVICES_LIBS} ${PLATFORM_LIBS})

# copy compile_commands.json for lsp integration 
# (assume we don't need this on Windows)
if (NOT WIN32)
  add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR})
endif()
