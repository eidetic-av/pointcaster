message("-- Configuring pointreceiver lib for ${VCPKG_TARGET_TRIPLET}")

# ----- Dependencies -----

find_package(ZeroMQ CONFIG REQUIRED)
if(RECEIVER_LIB_SHARED AND NOT WIN32) 
  list(APPEND RECEIVER_LINK_LIBS libzmq)
else()
  list(APPEND RECEIVER_LINK_LIBS libzmq-static)
  list(APPEND RECEIVER_COMPILE_DEFS ZMQ_STATIC)
endif()

find_package(draco CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS draco::draco)

find_package(bob-pointclouds CONFIG REQUIRED)
list(APPEND RECEIVER_LINK_LIBS bob::pointclouds)

find_path(READERWRITERQUEUE_INCLUDE_DIRS "readerwriterqueue/atomicops.h")
list(APPEND RECEIVER_INCLUDE_DIRS ${READERWRITERQUEUE_INCLUDE_DIRS})

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
list(APPEND RECEIVER_INCLUDE_DIRS ${ZPP_BITS_INCLUDE_DIRS})

if (WIN32)
  find_package(spdlog CONFIG REQUIRED)
  list(APPEND RECEIVER_LINK_LIBS spdlog::spdlog)
endif()

# ----- Pointreceiver library -----

set(RECEIVER_SOURCE_FILES src/pointreceiver.cc)

add_library(pointreceiver_obj OBJECT ${RECEIVER_SOURCE_FILES})
set_target_properties(pointreceiver_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pointreceiver_obj PRIVATE ${RECEIVER_INCLUDE_DIRS})
target_compile_definitions(pointreceiver_obj PRIVATE ${RECEIVER_COMPILE_DEFS} LIBRARY_BUILD)

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

# copy our API header into the output directory
add_custom_command(TARGET pointreceiver POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pointreceiver.h
    $<TARGET_FILE_DIR:pointreceiver>
)

# copy the required third-party notices into the output directory
add_custom_command(TARGET pointreceiver POST_BUILD 
  COMMAND ${Python_EXECUTABLE} 
  ${CMAKE_SOURCE_DIR}/scripts/generate_license.py
  --target ${CMAKE_BINARY_DIR} 
  --triplet ${VCPKG_TARGET_TRIPLET} 
  --output $<TARGET_FILE_DIR:pointreceiver>)

if (RECEIVER_LIB_SHARED AND NOT WIN32)
  # make sure that each dependency is copied to the build output directory too
  # not in win32 builds since all deps are static there
  foreach(lib_target ${RECEIVER_LINK_LIBS})
    if (NOT "${lib_target}" STREQUAL "")
      get_target_property(lib_type ${lib_target} TYPE)
      if(NOT lib_type STREQUAL "STATIC_LIBRARY")
        add_custom_command(TARGET pointreceiver POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
	        "$<TARGET_FILE:${lib_target}>"
          "$<TARGET_FILE_DIR:pointreceiver>")
      endif()
    endif()
  endforeach()
endif()

if (WITH_TESTS)
  add_executable(pointreceiver-test ${RECEIVER_SOURCE_FILES})
  target_link_libraries(pointreceiver-test PRIVATE ${RECEIVER_LINK_LIBS} 
    $<$<BOOL:${RECEIVER_LIB_SHARED}>:${RECEIVER_SHARED_LIBS}>
    $<$<NOT:$<BOOL:${RECEIVER_LIB_SHARED}>>:${RECEIVER_STATIC_LIBS}>)
endif()

# ----- Upload/deploy build artefacts -----

list(APPEND RECEIVER_DEPLOY_EXCLUDES
    "*cmake*" "*CMake*" "*vcpkg*" "*cache*"
    "*ninja*" "*.exp" "pointreceiver-test.*"
    "compile_commands.json")

  if(DEFINED ENV{POINTRECEIVER_DEPLOY_LOCATIONS})
    set(RECEIVER_DEPLOY_LOCATIONS $ENV{POINTRECEIVER_DEPLOY_LOCATIONS})
  endif()

  add_custom_target(deploy DEPENDS pointreceiver
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/deploy.py
    	  --name ${ARCHIVE_NAME}
          --exclude ${RECEIVER_DEPLOY_EXCLUDES}
          --directory ${CMAKE_CURRENT_BINARY_DIR}
          -- ${RECEIVER_DEPLOY_LOCATIONS})

  add_custom_target(deploy-archive DEPENDS pointreceiver
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/deploy.py
          --exclude ${RECEIVER_DEPLOY_EXCLUDES}
          --directory ${CMAKE_CURRENT_BINARY_DIR}
          --archive --sha256 
          --name ${ARCHIVE_NAME}
          -- "b2://pointcaster-builds")
