cmake_minimum_required(VERSION 3.22.0)
project(bob-pointcaster)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_COMPILER_IS_GNUCC)
  # use -pthread flag if we're compiling with gcc 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -fpermissive -Wall -fPIC")
endif()


find_package(spdlog CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
# find_package(libremidi CONFIG REQUIRED)

find_package(EBML CONFIG REQUIRED)
find_package(k4a CONFIG REQUIRED)
find_package(realsense2 CONFIG REQUIRED)

find_package(imgui CONFIG REQUIRED)
add_library(ImGui::ImGui ALIAS imgui::imgui)

find_package(libusb CONFIG REQUIRED)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
find_package(Corrade CONFIG REQUIRED Main)
find_package(Magnum CONFIG REQUIRED
  GL MeshTools Primitives SceneGraph
  Shaders Sdl2Application)
find_package(MagnumIntegration CONFIG REQUIRED ImGui)

corrade_add_resource(RESOURCES resources.conf)

set(SOURCE_FILES
  src/pointcaster.cc src/point_cloud.cc
  src/gui_helpers.cc 
  src/shaders/particle_sphere.cc
  src/devices/usb.cc
  src/devices/k4a/k4a_driver.cc
  src/devices/k4a/k4a_device.cc
  src/devices/rs2/rs2_driver.cc
  src/devices/rs2/rs2_device.cc)

set(PLATFORM_LIBS)
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
  imgui::imgui
  #libremidi
  Corrade::Main
  Magnum::Application
  Magnum::GL
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::Primitives
  Magnum::SceneGraph
  Magnum::Shaders
  MagnumIntegration::ImGui
  ${LIBUSB_LIBRARIES}
  spdlog::spdlog EBML::ebml k4a::k4a
  realsense2::realsense2 ${ZMQ_LIB})

# copy compile_commands.json for lsp integration 
# (assume we don't need this on Windows)
if (NOT WIN32)
  add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR})
endif()
