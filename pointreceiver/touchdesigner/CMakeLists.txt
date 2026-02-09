option(BUILD_TOUCHDESIGNER_LIB "Build TouchDesigner POP/CHOP plugins" ON)

if(NOT BUILD_TOUCHDESIGNER_LIB)
    return()
endif()

find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

set(TOUCHDESIGNER_LIB_OUTPUT_DIR "$<TARGET_FILE_DIR:pointreceiver>")

# Core shared DLL (owns singletons, registries, threads, etc.)
add_library(PointreceiverTDCore SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PointreceiverPlugin.cpp"
)

# export all symbols so POP/CHOP can link without declspecs
set_target_properties(PointreceiverTDCore PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_compile_features(PointreceiverTDCore PRIVATE cxx_std_23)

target_include_directories(PointreceiverTDCore PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/logger"
)

target_compile_options(PointreceiverTDCore PRIVATE
    /EHsc
    /utf-8
)

target_link_libraries(PointreceiverTDCore
    PRIVATE
    pointreceiver
    bob::pointclouds
# TODO i think PUBLIC is only necessary here because the logger isnt a library like it is in the qt branch
    PUBLIC 
    spdlog::spdlog
    fmt::fmt
)

set_target_properties(PointreceiverTDCore PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
)

# POP plugin DLL
add_library(PointreceiverCloudInPOP MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PointreceiverCloudInPOP.cpp"
)

target_compile_features(PointreceiverCloudInPOP PRIVATE cxx_std_23)

target_include_directories(PointreceiverCloudInPOP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/logger"
)

target_compile_options(PointreceiverCloudInPOP PRIVATE
    /EHsc
    /utf-8
)

target_link_libraries(PointreceiverCloudInPOP PRIVATE
    PointreceiverTDCore
)

set_target_properties(PointreceiverCloudInPOP PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
)

# CHOP plugin DLL
add_library(PointreceiverMessageInCHOP MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PointreceiverMessageInCHOP.cpp"
)

target_compile_features(PointreceiverMessageInCHOP PRIVATE cxx_std_23)

target_include_directories(PointreceiverMessageInCHOP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/logger"
)

target_compile_options(PointreceiverMessageInCHOP PRIVATE
    /EHsc
    /utf-8
)

target_link_libraries(PointreceiverMessageInCHOP PRIVATE
    PointreceiverTDCore
)

set_target_properties(PointreceiverMessageInCHOP PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
)
