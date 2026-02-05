option(BUILD_TOUCHDESIGNER_LIB "Build dll for Pointreceiver lib integration in Derivative TouchDesigner" ON)

if (NOT BUILD_TOUCHDESIGNER_LIB)
    return()
endif()

add_library(PointreceiverPOP MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PointreceiverPOP.cpp"
)

target_compile_features(PointreceiverPOP PRIVATE cxx_std_23)
target_include_directories(PointreceiverPOP PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# no dlls of any config should not have a lib prefix
set_target_properties(PointreceiverPOP PROPERTIES PREFIX "")

set_property(TARGET PointreceiverPOP PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)
target_compile_options(PointreceiverPOP PRIVATE /EHsc)

target_link_libraries(PointreceiverPOP PRIVATE pointreceiver)

# place the touchdesigner DLL next to pointreceiver.dll
set(TOUCHDESIGNER_LIB_OUTPUT_DIR "$<TARGET_FILE_DIR:pointreceiver>")

set_target_properties(PointreceiverPOP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TOUCHDESIGNER_LIB_OUTPUT_DIR}"
)
