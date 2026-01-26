include(GenerateExportHeader)

add_library(pointcaster_core SHARED
    point_cloud.cc
    codec/draco.cc
    uuid/uuid.cc
    uuid/wtf/wtf.cc
    logger/logger.cc
)

add_library(pointcaster::core ALIAS pointcaster_core)

find_path(ZPP_BITS_INCLUDE_DIRS "zpp_bits.h")
target_include_directories(pointcaster_core PRIVATE ${ZPP_BITS_INCLUDE_DIRS})

find_package(draco CONFIG REQUIRED)
target_link_libraries(pointcaster_core PRIVATE draco::draco)

find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
target_link_libraries(pointcaster_core PUBLIC spdlog::spdlog fmt::fmt)

target_include_directories(pointcaster_core
    ${WARNING_GUARD}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(pointcaster_core
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
)

generate_export_header(pointcaster_core
    EXPORT_FILE_NAME
        ${PROJECT_BINARY_DIR}/include/pointcaster/core.h
)