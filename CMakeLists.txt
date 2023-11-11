cmake_minimum_required(VERSION 3.22.0)
project(bob-pointcaster LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LINUX TRUE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(WITH_K4A "Enable Azure Kinect support" ON)
option(WITH_MQTT "Enable MQTT client support" ON)
option(WITH_THREAD_SANITIZER "Enable thread sanitizer on Debug configuration builds" OFF)
option(WITH_ADDRESS_SANITIZER "Enable address sanitizer on Debug configuration builds" OFF)
option(WITH_RECEIVER_LIB "Build pointreceiver client library" OFF)
option(WITH_TESTS "Build tests" OFF)

option(WITH_TRACY "Enable profiling using Tracy" OFF)
option(TRACY_ENABLE "Enable profiling using Tracy" OFF)
if (WITH_TRACY)
  set(TRACY_ENABLE ON CACHE BOOL "Enable profiling using Tracy" FORCE)
endif()

find_program(MOLD_LINKER_FOUND mold)
if (LINUX AND MOLD_LINKER_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=mold")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=mold")
endif()

if(LINUX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wfatal-errors")
endif()

# link time optimisation for release builds
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:TRUE>)

if (WIN32)
  set(Boost_INCLUDE_DIR ${_VCPKG_INSTALLED_DIR}/x64-windows-static/include)
endif()

# ----- Geometry -----

find_package(Eigen3 CONFIG REQUIRED)
list(APPEND LINK_LIBS Eigen3::Eigen)

find_path(EARCUT_HPP_INCLUDE_DIRS "mapbox/earcut.hpp")
list(APPEND INCLUDE_DIRS ${EARCUT_HPP_INCLUDE_DIRS})

find_path(CUDA_NOISE_INCLUDE_DIRS "cuda_noise.cuh")
list(APPEND INCLUDE_DIRS ${CUDA_NOISE_INCLUDE_DIRS})

# ----- Point clouds -----

find_package(bob-pointclouds CONFIG REQUIRED)
list(APPEND LINK_LIBS bob::pointclouds)

find_package(draco CONFIG REQUIRED)
list(APPEND LINK_LIBS draco::draco)

# ----- Networking -----

find_package(ZeroMQ CONFIG REQUIRED)
list(APPEND LINK_LIBS libzmq-static)
if (WIN32)
  # zmq windows dependency:
  list(APPEND PLATFORM_LIBS bcrypt)
endif()

if (WITH_MQTT)
  find_package(PahoMqttCpp CONFIG REQUIRED)
  list(APPEND LINK_LIBS PahoMqttCpp::paho-mqttpp3-static uuid)
  find_package(msgpack-cxx CONFIG REQUIRED)
  list(APPEND LINK_LIBS msgpack-cxx)
  list(APPEND OPTIONAL_SRC src/mqtt/mqtt_client.cc)
  add_compile_definitions(WITH_MQTT)
endif()

# ----- Control -----

find_package(libremidi CONFIG REQUIRED)
list(APPEND LINK_LIBS libremidi)
list(APPEND POINTCASTER_SRC src/midi/midi_client.cc)

# ----- Dear ImGui lib -----

find_package(imgui CONFIG REQUIRED)
add_library(ImGui::ImGui ALIAS imgui::imgui)
list(APPEND LINK_LIBS imgui::imgui)
add_definitions(-DIMGUI_DEFINE_MATH_OPERATORS)

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
    src/devices/k4a/k4a_device.cc
    src/devices/k4a/k4a_driver.cc
    src/devices/k4a/k4a_driver.cu)

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

# ----- CUDA -----

find_package(CUDAToolkit REQUIRED)
find_package(Thrust CONFIG REQUIRED)
thrust_create_target(Thrust)
list(APPEND LINK_LIBS Thrust)

# ----- Utility libs -----

find_package(fmt CONFIG REQUIRED)
list(APPEND LINK_LIBS fmt::fmt)

find_package(spdlog CONFIG REQUIRED)
list(APPEND LINK_LIBS spdlog::spdlog_header_only)

find_package(serdepp CONFIG REQUIRED)
list(APPEND LINK_LIBS serdepp::serdepp)

find_package(toml11 CONFIG REQUIRED)
list(APPEND LINK_LIBS toml11::toml11)

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
list(APPEND INCLUDE_DIRS ${ZPP_BITS_INCLUDE_DIRS})

find_package(Tweeny CONFIG REQUIRED)
list(APPEND LINK_LIBS tweeny)

find_package(unofficial-concurrentqueue CONFIG REQUIRED)
list(APPEND LINK_LIBS unofficial::concurrentqueue::concurrentqueue)

# ----- Profiling -----

find_package(Tracy CONFIG REQUIRED)
if (TRACY_ENABLE)
  list(APPEND LINK_LIBS Tracy::TracyClient)
  add_compile_definitions(TRACY_ENABLE)
  add_compile_definitions(TRACY_ON_DEMAND)
endif()

# ----- Pointcloud library -----

# ----- Pointcaster application -----

set(SOURCE_FILES
  src/pointcaster.cc
  src/devices/device.cc
  src/devices/usb.cc
  src/camera/camera_controller.cc
  src/analysis/analyser_2d.cc
  src/radio/radio.cc
  src/snapshots.cc
  src/point_cloud_renderer.cc
  src/sphere_renderer.cc
  src/shaders/particle_sphere.cc
  src/shaders/texture_display.cc
  src/gui/widgets.cc
  src/gui/range_slider.cc
  src/parameters.cc
  src/path.cc
  src/uuid.cc
  src/wtf/wtf.cc
  src/tween/tween_manager.cc
  src/publisher/publisher.cc
  src/publisher/publisher_utils.cc
  src/operators/session_operator_host.cc
  src/operators/session_operator_host.cu
  src/operators/noise_operator.cc
  src/operators/knn_filter_operator.cc
  ${POINTCASTER_SRC}
  ${OPTIONAL_SRC})

corrade_add_resource(RESOURCES resources.conf)

if (WIN32)
  list(APPEND SOURCE_FILES resources/windows-dpi.manifest)
endif()

add_executable(pointcaster ${SOURCE_FILES} ${RESOURCES})
set_target_properties(pointcaster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET pointcaster PROPERTY 
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

target_include_directories(pointcaster PRIVATE
    ${INCLUDE_DIRS} ${OPTIONAL_INCLUDE_DIRS} 
    ${PLATFORM_INCLUDE_DIRS})

target_link_libraries(pointcaster PRIVATE 
    ${LINK_LIBS} ${OPTIONAL_LIBS} ${PLATFORM_LIBS})

if (WITH_THREAD_SANITIZER)
  target_compile_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=thread>")
  target_link_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=thread>")
elseif(WITH_ADDRESS_SANITIZER)
  target_compile_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=address>")
  target_link_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=address>")
endif()

# ----- Pointreceiver library -----

if (WITH_RECEIVER_LIB)
  add_subdirectory(pointreceiver)
endif()

# ----- Post-build actions -----

if (WIN32 AND WITH_K4A)
  # shared k4abt binaries must be moved into executable directory manually
  # (this lib can't be static)
  file (GLOB K4ABT_BINS "${K4ABT_ROOT}/sdk/windows-desktop/amd64/release/bin/*")
  foreach(K4ABT_BIN ${K4ABT_BINS})
    add_custom_command(TARGET pointcaster POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${K4ABT_BIN} $<TARGET_FILE_DIR:pointcaster>)
  endforeach()
endif()

if (LINUX)
  # copy compile_commands.json for lsp integration on linux
  add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR})
endif()
