cmake_minimum_required(VERSION 3.22.3)
project(bob-pointcaster)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_COMPILER_IS_GNUCC)
  # use -pthread flag if we're compiling with gcc 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -fpermissive -pthread -Wno-deprecated-declarations -fPIC")
endif()

find_package(spdlog CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(EBML CONFIG REQUIRED)
find_package(k4a CONFIG REQUIRED)

include_directories(include)

set(SOURCE_FILES src/pointcaster.cc src/k4a_driver.cc)

set(PLATFORM_LIBS)
if (WIN32)
  # include bcrypt from the Windows SDK
  set(PLATFORM_LIBS ${PLATFORM_LIBS} bcrypt)
endif()

add_executable(pointcaster ${SOURCE_FILES})

# set msvc runtime to static if compiling with a *-windows-static triplet
if (WIN32 AND (VCPKG_TARGET_TRIPLET MATCHES static))
set_property(TARGET ssp_server PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(pointcaster PRIVATE ${PLATFORM_LIBS}
  spdlog::spdlog EBML::ebml k4a::k4a libzmq-static)

# copy compile_commands.json for lsp integration 
# (assume we don't need this on Windows)
if (NOT WIN32)
  add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR})
endif()
