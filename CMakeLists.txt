cmake_minimum_required(VERSION 3.31.6)

project(pointcaster VERSION 0.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Qt)

add_subdirectory(src)

include(Packaging)
