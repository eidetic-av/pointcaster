message("-- Configuring pointreceiver lib for ${VCPKG_TARGET_TRIPLET}")

# ----- Dependencies -----

find_package(spdlog CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS spdlog::spdlog)

find_package(ZeroMQ CONFIG REQUIRED)
list(APPEND RECEIVER_SHARED_LIBS libzmq)
list(APPEND RECEIVER_STATIC_LIBS libzmq-static)

find_package(draco CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS draco::draco)

find_package(bob-pointclouds CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS bob::pointclouds)

find_path(READERWRITERQUEUE_INCLUDE_DIRS "readerwriterqueue/atomicops.h")
list(APPEND RECEIVER_INCLUDE_DIRS ${READERWRITERQUEUE_INCLUDE_DIRS})

# ----- Pointreceiver library -----

set(RECEIVER_SOURCE_FILES src/pointreceiver.cc)

add_library(pointreceiver_obj OBJECT ${RECEIVER_SOURCE_FILES})
set_target_properties(pointreceiver_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(pointreceiver_obj
  PRIVATE ${RECEIVER_INCLUDE_DIRS})

add_library(pointreceiver SHARED $<TARGET_OBJECTS:pointreceiver_obj>)
add_library(pointreceiver::shared ALIAS pointreceiver)

add_library(pointreceiver_static STATIC $<TARGET_OBJECTS:pointreceiver_obj>)
add_library(pointreceiver::static ALIAS pointreceiver_static)

target_compile_features(pointreceiver_static PRIVATE cxx_std_20)
target_compile_features(pointreceiver PRIVATE cxx_std_20)

target_include_directories(pointreceiver_static
  PRIVATE ${RECEIVER_INCLUDE_DIRS})
target_include_directories(pointreceiver
  PRIVATE ${RECEIVER_INCLUDE_DIRS})

if (WIN32)
  target_compile_options(pointreceiver PRIVATE -fdeclspec)
endif()

set_target_properties(pointreceiver PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(pointreceiver_static PROPERTIES DEBUG_POSTFIX "d")

target_link_libraries(pointreceiver_static
  PRIVATE ${RECEIVER_LINK_LIBS} ${RECEIVER_STATIC_LIBS})
target_link_libraries(pointreceiver
  PRIVATE ${RECEIVER_LINK_LIBS} ${RECEIVER_SHARED_LIBS})

if (WITH_TESTS)
  add_executable(pointreceiver-test ${RECEIVER_SOURCE_FILES})
  target_link_libraries(pointreceiver-test
    PRIVATE ${RECEIVER_LINK_LIBS} ${RECEIVER_STATIC_LIBS})
endif()

# set msvc runtime
if(${VCPKG_TARGET_TRIPLET} STREQUAL "x64-windows")
  set_property(TARGET pointreceiver PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
