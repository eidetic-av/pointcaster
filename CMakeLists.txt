cmake_minimum_required(VERSION 3.30.5)

project(pointcaster VERSION 0.2.6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Qt)
include(QmlAdapterGenerator)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_custom_target(link_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json)

# flatten windows installs: exe and dlls at the install root with plugins/ and qml/ beside them
if(WIN32)
    set(CMAKE_INSTALL_BINDIR ".")
endif()

add_subdirectory(src)

include(Packaging)
