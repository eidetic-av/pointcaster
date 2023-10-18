if (VCPKG_TARGET_TRIPLET MATCHES "x64-linux|x64-windows-static")
  set(RECEIVER_LIB_LINKAGE STATIC)
else()
  set(RECEIVER_LIB_LINKAGE SHARED)
endif()

message("-- Configuring ${RECEIVER_LIB_LINKAGE} pointreceiver lib\
 for ${VCPKG_TARGET_TRIPLET}")

find_package(spdlog CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS spdlog::spdlog)

find_package(ZeroMQ CONFIG REQUIRED)
list(APPEND RECEIVER_PLATFORM_LIBS
  $<IF:$<STREQUAL:${RECEIVER_LIB_LINKAGE},STATIC>,libzmq-static,libzmq>)

find_package(draco CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS draco::draco)

find_path(READERWRITERQUEUE_INCLUDE_DIRS "readerwriterqueue/atomicops.h")
list(APPEND RECEIVER_INCLUDE_DIRS ${READERWRITERQUEUE_INCLUDE_DIRS})

find_package(bob-pointclouds CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS bob::pointclouds)

set(RECEIVER_SOURCE_FILES src/pointreceiver.cc)

add_library(pointreceiver ${RECEIVER_LIB_LINKAGE} ${RECEIVER_SOURCE_FILES})

set_target_properties(pointreceiver PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  DEBUG_POSTFIX "d")

if (WIN32)
  target_compile_options(pointreceiver PRIVATE -fdeclspec)
endif()

target_link_libraries(pointreceiver PRIVATE
  ${RECEIVER_PLATFORM_LIBS} ${RECEIVER_LINK_LIBS})

target_include_directories(pointreceiver PRIVATE ${RECEIVER_INCLUDE_DIRS})

if (WITH_TESTS)
  add_executable(pointreceiver-test ${RECEIVER_SOURCE_FILES})
  target_link_libraries(pointreceiver-test PRIVATE
  ${RECEIVER_PLATFORM_LIBS} ${RECEIVER_LINK_LIBS})
endif()


# set msvc runtime
if(WIN32 AND ${RECEIVER_LIB_LINKAGE} STREQUAL "STATIC")
  set_property(TARGET pointreceiver PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# copy compile_commands.json for lsp integration 
# (assume we don't need this on Windows)
# if (NOT WIN32)
#   add_custom_target(copy-compile-commands ALL
#     ${CMAKE_COMMAND} -E copy_if_different
#     ${CMAKE_BINARY_DIR}/compile_commands.json
#     ${CMAKE_CURRENT_LIST_DIR})
# endif()
