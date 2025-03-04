message("-- Configuring pointreceiver lib for ${VCPKG_TARGET_TRIPLET}")

# ----- Dependencies -----

find_package(spdlog CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS spdlog::spdlog)

find_package(ZeroMQ CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS libzmq-static)
list(APPEND RECEIVER_COMPILE_DEFS ZMQ_STATIC)

find_package(draco CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS draco::draco)

find_package(bob-pointclouds CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS bob::pointclouds)

find_path(READERWRITERQUEUE_INCLUDE_DIRS "readerwriterqueue/atomicops.h")
list(APPEND RECEIVER_INCLUDE_DIRS ${READERWRITERQUEUE_INCLUDE_DIRS})

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
list(APPEND RECEIVER_INCLUDE_DIRS ${ZPP_BITS_INCLUDE_DIRS})

# ----- Pointreceiver library -----

set(RECEIVER_SOURCE_FILES src/pointreceiver.cc)

add_library(pointreceiver_obj OBJECT ${RECEIVER_SOURCE_FILES})
set_target_properties(pointreceiver_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pointreceiver_obj PRIVATE ${RECEIVER_INCLUDE_DIRS})
target_compile_definitions(pointreceiver_obj PRIVATE ${RECEIVER_COMPILE_DEFS})

if(RECEIVER_LIB_SHARED)
  add_library(pointreceiver SHARED $<TARGET_OBJECTS:pointreceiver_obj>)
  if (WIN32)
    target_compile_options(pointreceiver PRIVATE -fdeclspec)
  endif()
  set_property(TARGET pointreceiver PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
else()
  add_library(pointreceiver STATIC $<TARGET_OBJECTS:pointreceiver_obj>)
  target_link_libraries(pointreceiver PRIVATE ${RECEIVER_LINK_LIBS} ${RECEIVER_STATIC_LIBS})
endif()

target_include_directories(pointreceiver 
  PRIVATE ${RECEIVER_INCLUDE_DIRS}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(pointreceiver PRIVATE cxx_std_20)
target_link_libraries(pointreceiver PRIVATE ${RECEIVER_LINK_LIBS})

set_target_properties(pointreceiver PROPERTIES 
  DEBUG_POSTFIX "d"
)

add_custom_command(TARGET pointreceiver POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pointreceiver.h
    $<TARGET_FILE_DIR:pointreceiver>
)

if (WITH_TESTS)
  add_executable(pointreceiver-test ${RECEIVER_SOURCE_FILES})
  target_link_libraries(pointreceiver-test PRIVATE ${RECEIVER_LINK_LIBS} 
    $<$<BOOL:${RECEIVER_LIB_SHARED}>:${RECEIVER_SHARED_LIBS}>
    $<$<NOT:$<BOOL:${RECEIVER_LIB_SHARED}>>:${RECEIVER_STATIC_LIBS}>)
endif()
