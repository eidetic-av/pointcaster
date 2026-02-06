option(BUILD_TOUCHDESIGNER_LIB "Build dll for Pointreceiver lib integration in Derivative TouchDesigner" ON)

if(NOT BUILD_TOUCHDESIGNER_LIB)
    return()
endif()

add_library(PointreceiverPOP MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PointreceiverPOP.cpp"
)

target_compile_features(PointreceiverPOP PRIVATE cxx_std_23)
target_include_directories(PointreceiverPOP PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include"
    # TODO move logger to public include directory
    "${CMAKE_SOURCE_DIR}/src/logger"
)

# no dlls of any config should not have a lib prefix
set_target_properties(PointreceiverPOP PROPERTIES PREFIX "")

set_property(TARGET PointreceiverPOP PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)
target_compile_options(PointreceiverPOP PRIVATE 
# TODO what are these flags? it was from the POP sample vcxproject
    /EHsc
    # utf-8 needed for pc::logger logging
    /utf-8
)

# TODO wont be necessary when logger comes from a lib
# when its merged from qt branch
find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

target_link_libraries(PointreceiverPOP PRIVATE
    pointreceiver
    bob::pointclouds
    # TODO this should link to pointcaster::logger 
    # lib here instead of spdlog
    spdlog::spdlog fmt::fmt
)

# place the touchdesigner DLL next to pointreceiver.dll
set(TOUCHDESIGNER_LIB_OUTPUT_DIR "$<TARGET_FILE_DIR:pointreceiver>")

set_target_properties(PointreceiverPOP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
)
