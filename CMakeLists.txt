cmake_minimum_required(VERSION 3.22.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LINUX TRUE)
endif()

get_filename_component(CMAKE_PRESET_NAME ${CMAKE_BINARY_DIR} NAME)

########################################################

project(pointcaster LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(WITH_K4A "Enable Azure Kinect support" ON)
option(WITH_MQTT "Enable MQTT client support" ON)
option(WITH_MIDI "Enable MIDI support" ON)
option(WITH_OSC "Enable OSC client/server support" ON)
option(WITH_THREAD_SANITIZER "Enable thread sanitizer on Debug configuration builds" OFF)
option(WITH_ADDRESS_SANITIZER "Enable address sanitizer on Debug configuration builds" OFF)
option(WITH_MARCH_NATIVE "Enable -march=native builds" ON)

option(WITH_POINTCASTER "Build main application" ON)
if (WITH_POINTCASTER)
  enable_language(CUDA)
endif()

option(WITH_RECEIVER_LIB "Build pointreceiver client library" OFF)
if(WIN32)
  option(RECEIVER_LIB_SHARED "Build pointreceiver client library with dynamic linkage" ON)
else()
  option(RECEIVER_LIB_SHARED "Build pointreceiver client library with dynamic linkage" OFF)
endif()

option(WITH_TESTS "Build tests" OFF)

option(GENERATE_REFLECTIONS ON)

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
  
find_package(Python REQUIRED COMPONENTS Interpreter)

if(LINUX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wfatal-errors")
endif()

if (WIN32)
    # stop win socket conflicts with includes from networking libraries
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:TRUE>)

if (WIN32 AND WITH_POINTCASTER)
  # set(Boost_INCLUDE_DIR ${_VCPKG_INSTALLED_DIR}/x64-windows-static-md/include)
  set(Boost_INCLUDE_DIR ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include)
endif()

if (GENERATE_REFLECTIONS)

  set(REFLECTED_TYPES
    src/session.h
    src/point_cloud_renderer_config.h
    src/analysis/analyser_2d_config.h
    src/camera/camera_config.h
    src/devices/device_config.h
    src/devices/usb_config.h
    src/midi/midi_device_config.h
    src/osc/osc_client_config.h
    src/osc/osc_server_config.h
    src/mqtt/mqtt_client_config.h
    src/radio/radio_config.h
    src/client_sync/sync_server.h
    src/tween/tween_config.h
    src/operators/operator_host_config.h
    src/operators/rake_operator.h
    src/operators/noise_operator.h
    src/operators/rotate_operator.h
    src/operators/sample_filter_operator.h
    src/operators/range_filter_operator.h
    src/operators/denoise/denoise_operator.h
  )

  foreach(HEADER IN LISTS REFLECTED_TYPES)
    cmake_path(GET HEADER STEM FILE_NAME)
    cmake_path(GET HEADER PARENT_PATH FILE_DIR)
    list(APPEND GENERATED_HEADERS "${FILE_DIR}/${FILE_NAME}.gen.h")
  endforeach()

  add_custom_command(
    OUTPUT ${GENERATED_HEADERS}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/generate_reflections.py ${REFLECTED_TYPES}
    DEPENDS ${REFLECTED_TYPES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating reflection information"
    VERBATIM
  )

  add_custom_target(generate_reflections ALL DEPENDS ${GENERATED_HEADERS})
  list(APPEND POINTCASTER_DEPENDENCIES generate_reflections)
  
endif()

find_package(ZeroMQ CONFIG REQUIRED)
list(APPEND LINK_LIBS libzmq libzmq-static)
if (WIN32)
  # zmq windows dependency:
  list(APPEND PLATFORM_LIBS bcrypt)
endif()


if (WITH_POINTCASTER)

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

  find_path(HAPPLY_INCLUDE_DIRS "happly.h")
  list(APPEND INCLUDE_DIRS ${HAPPLY_INCLUDE_DIRS})

  # ----- Networking -----

  if (WITH_MQTT)
    find_package(PahoMqttCpp CONFIG REQUIRED)
    list(APPEND LINK_LIBS PahoMqttCpp::paho-mqttpp3-static uuid)
    find_package(msgpack-cxx CONFIG REQUIRED)
    list(APPEND LINK_LIBS msgpack-cxx)
    list(APPEND OPTIONAL_SRC src/mqtt/mqtt_client.cc)
    add_compile_definitions(WITH_MQTT)
  endif()

  # ----- Control -----

  if (WITH_MIDI)
    find_package(libremidi CONFIG REQUIRED)
    list(APPEND LINK_LIBS libremidi)
    list(APPEND POINTCASTER_SRC src/midi/midi_device.cc)
    add_compile_definitions(WITH_MIDI)

  # look for installed Tobias Erichsen's libRtpMidiSDK
  find_library(RTPMIDI_LIB
    NAMES RtpMidi libRtpMidi
    HINTS 
      "/opt/libRtpMidiSDK/targets/Linux/x64" 
      "C:/Program Files (x86)/Tobias Erichsen/libRtpMidiSDK/targets/Microsoft/Windows/x64")

  if (RTPMIDI_LIB)
    message(STATUS "Found libRtpMidiSDK: ${RTPMIDI_LIB}")

    get_filename_component(RTPMIDI_LIB_DIR ${RTPMIDI_LIB} DIRECTORY)

    if (LINUX)
        set(RTPMIDI_INCLUDES "${RTPMIDI_LIB_DIR}/../../..")
    elseif(WIN32)
        set(RTPMIDI_INCLUDES "${RTPMIDI_LIB_DIR}/../../../..")
    endif()

    find_package(mdns_cpp CONFIG REQUIRED)
    list(APPEND LINK_LIBS mdns_cpp::mdns_cpp)

    list(APPEND POINTCASTER_SRC 
      src/midi/rtpmidi_device.cc src/midi/rtp_peer.cc)
    list(APPEND LINK_LIBS ${RTPMIDI_LIB})
    list(APPEND INCLUDE_DIRS ${RTPMIDI_INCLUDES})
    add_compile_definitions(WITH_RTP_MIDI)
  else()
    message(STATUS "libRtpMidiSDK was not found. Network MIDI will be disabled.")
  endif()
  
  endif()

  if (WITH_OSC)
    message(STATUS "Configuring with OSC client/server support")
    find_package(liblo CONFIG REQUIRED)
    list(APPEND LINK_LIBS liblo::liblo_static)
    list(APPEND POINTCASTER_SRC 
      src/osc/osc_client.cc src/osc/osc_server.cc)
    add_compile_definitions(WITH_OSC)
  endif()

  # ----- Dear ImGui lib -----

  find_package(imgui CONFIG REQUIRED)
  add_library(ImGui::ImGui ALIAS imgui::imgui)
  list(APPEND LINK_LIBS imgui::imgui)
  add_definitions(-DIMGUI_DEFINE_MATH_OPERATORS)

  list(APPEND THIRDPARTY_SRC thirdparty/imnodes/imnodes.cpp)
  list(APPEND INCLUDE_DIRS thirdparty/imnodes)

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

  if (NOT WIN32)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(libusb REQUIRED IMPORTED_TARGET libusb-1.0)
    list(APPEND LINK_LIBS PkgConfig::libusb)
  endif()

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

  # ----- Orbbec -----
  
  find_package(OrbbecSDK REQUIRED CONFIG)
  list(APPEND LINK_LIBS ${OrbbecSDK_LIBS})
  list(APPEND OPTIONAL_SRC 
    src/devices/orbbec/orbbec_device.cc
    src/devices/orbbec/orbbec_device.cu)

  # ----- OpenCV -----

  if (WIN32)
    find_package(OpenCV REQUIRED HINTS "/opt/opencv/build")
    get_target_property(OPENCV_DLL_RELEASE opencv_world IMPORTED_LOCATION_RELEASE)
    get_target_property(OPENCV_DLL_DEBUG opencv_world IMPORTED_LOCATION_DEBUG)
    list(APPEND INSTALL_DLLS_RELEASE "${OPENCV_DLL_RELEASE}")
    list(APPEND INSTALL_DLLS_DEBUG "${OPENCV_DLL_DEBUG}")
  else()
    find_package(OpenCV CONFIG REQUIRED)
  endif()
  list(APPEND LINK_LIBS ${OpenCV_LIBS})
  list(APPEND INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})

  # ----- PCL -----

  find_package(PCL CONFIG REQUIRED)
  list(APPEND LINK_LIBS ${PCL_LIBRARIES})

  # ----- TBB -----

  find_package(TBB CONFIG REQUIRED)
  list(APPEND LINK_LIBS TBB::tbb TBB::tbbmalloc)

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

  find_package(nfd CONFIG REQUIRED)
  list(APPEND LINK_LIBS nfd::nfd)

  # ----- Profiling -----

  find_package(Tracy CONFIG REQUIRED)
  if (TRACY_ENABLE)
    list(APPEND LINK_LIBS Tracy::TracyClient)
    add_compile_definitions(TRACY_ENABLE)
    add_compile_definitions(TRACY_ON_DEMAND)
  endif()

  # ----- Pointcaster application -----

  set(SOURCE_FILES
    src/pointcaster.cc
    src/devices/device.cc
    src/devices/usb.cc
    src/devices/sequence/ply_sequence_player.cc
    src/devices/sequence/ply_sequence_player.cu
    src/devices/sequence/ply_sequence_player_gui.cc
    src/camera/camera_controller.cc
    src/analysis/analyser_2d.cc
    src/radio/radio.cc
    src/client_sync/sync_server.cc
    src/snapshots.cc
    src/point_cloud_renderer.cc
    src/sphere_renderer.cc
    src/shaders/particle_sphere.cc
    src/shaders/texture_display.cc
    src/gui/widgets.cc
    src/gui/range_slider.cc
    src/gui/drag_slider.cc
    src/gui/windows.cc
    src/parameters.cc
    src/path.cc
    src/uuid.cc
    src/wtf/wtf.cc
    src/tween/tween_manager.cc
    src/publisher/publisher.cc
    src/publisher/publisher_utils.cc
    src/graph/operator_graph.cc
    src/operators/session_operator_host.cc
    src/operators/session_operator_host.cu
    src/operators/noise_operator.cc
    src/operators/rotate_operator.cc
    src/operators/rake_operator.cc
    src/operators/sample_filter_operator.cc
    src/operators/range_filter_operator.cc
    src/operators/denoise/kdtree.cu
    src/operators/denoise/denoise_operator.cc
    src/operators/pcl/cluster_extraction_operator.cc
    ${THIRDPARTY_SRC}
    ${POINTCASTER_SRC}
    ${OPTIONAL_SRC})

# ----- Resources -----

  corrade_add_resource(RESOURCES resources.conf)

  if (WIN32)
    list(APPEND SOURCE_FILES resources/windows-dpi.manifest)
  endif()

  add_executable(pointcaster ${SOURCE_FILES} ${RESOURCES})

# ----- Pointcaster build properties

  set_target_properties(pointcaster PROPERTIES 
    UNITY_BUILD OFF
    UNITY_BUILD_MODE BATCH
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_PROPAGATE_HOST_FLAGS OFF
    CUDA_ARCHITECTURES "61;75;86;89"
  )

  if (POINTCASTER_DEPENDENCIES)
    add_dependencies(pointcaster ${POINTCASTER_DEPENDENCIES})
  endif()

  target_include_directories(pointcaster PRIVATE
    ${INCLUDE_DIRS} ${OPTIONAL_INCLUDE_DIRS} 
    ${PLATFORM_INCLUDE_DIRS})

  target_link_libraries(pointcaster PRIVATE 
    ${LINK_LIBS} ${OPTIONAL_LIBS} ${PLATFORM_LIBS})

  target_precompile_headers(pointcaster PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/pch.h)

  # optimisations

  if (LINUX)

    if (WITH_MARCH_NATIVE)
      target_compile_options(pointcaster 
        PRIVATE "-march=native")
    endif()

    target_compile_options(pointcaster 
      PRIVATE "$<$<CONFIG:Release>:-O3>")

  endif()

  # platform specific

  if (WIN32)
    target_compile_options(pointcaster PRIVATE /bigobj /utf-8)
    target_compile_options(pointcaster PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	    -forward-slash-prefix-opts >)
  endif()

  # debugging related options

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG 
      "${CMAKE_CXX_FLAGS_DEBUG} /analyze /Z7" 
      CACHE STRING "Debug flags" FORCE)
  endif()

  if (WITH_THREAD_SANITIZER)
    target_compile_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=thread>")
    target_link_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=thread>")
  elseif(WITH_ADDRESS_SANITIZER)
    target_compile_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=address>")
    target_link_options(pointcaster PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=address>")
  endif()

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

if (WIN32)
  # if we have any extra DLLS that need to be copied into
  # the output directory
  add_custom_command(TARGET pointcaster POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<$<CONFIG:Release>:${INSTALL_DLLS_RELEASE}>
      $<$<CONFIG:Debug>:${INSTALL_DLLS_DEBUG}>
      $<TARGET_FILE_DIR:pointcaster>
    COMMAND_EXPAND_LISTS
  )
endif()

if (LINUX)
  # copy compile_commands.json for lsp integration on linux
  add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR})
endif()

# copy the required third-party notices into our output directory
if (WITH_POINTCASTER)
  add_custom_command(TARGET pointcaster POST_BUILD 
    COMMAND ${Python_EXECUTABLE} 
    ${CMAKE_SOURCE_DIR}/scripts/generate_license.py
    --target ${CMAKE_BINARY_DIR} 
    --triplet ${VCPKG_TARGET_TRIPLET} 
    --output $<TARGET_FILE_DIR:pointcaster>)
endif()

# ----- Upload/deploy build artefacts -----

if (WITH_POINTCASTER)

  list(APPEND DEPLOY_EXCLUDES
      "*cmake*" "*CMake*" "*vcpkg*"
      "*ninja*" "*.lib" "*.exp"
      "resource*.cpp" "Log/*"
      "data/*" "compile_commands.json")

  if(DEFINED ENV{POINTCASTER_DEPLOY_LOCATIONS})
    set(DEPLOY_LOCATIONS $ENV{POINTCASTER_DEPLOY_LOCATIONS})
  endif()

  add_custom_target(deploy DEPENDS pointcaster
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/deploy.py
          --exclude ${DEPLOY_EXCLUDES}
          --directory ${CMAKE_BINARY_DIR}
          -- ${DEPLOY_LOCATIONS})

  add_custom_target(deploy-archive DEPENDS pointcaster
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/deploy.py
          --exclude ${DEPLOY_EXCLUDES}
          --directory ${CMAKE_BINARY_DIR}
          --archive --sha256 
          --name ${ARCHIVE_NAME}
          -- "b2://pointcaster-builds")
endif()
