include(GenerateExportHeader)

# the runtime config registry and app-side config conversions/helpers

# shared because the ConfigRegistry is owned by the app and
# referenced from dynamically loaded plugins

add_library(pointcaster_config SHARED
    config_registry.cc
    transform_config.cc
)

add_library(pointcaster::config ALIAS pointcaster_config)

target_link_libraries(pointcaster_config PUBLIC pointcaster::core)

# shared configs need reflectcpp
find_package(reflectcpp CONFIG REQUIRED)
target_link_libraries(pointcaster_config PRIVATE reflectcpp::reflectcpp)

set_target_properties(pointcaster_config
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        AUTOMOC OFF
)

generate_export_header(pointcaster_config
    EXPORT_FILE_NAME
        ${PROJECT_BINARY_DIR}/include/pointcaster/config.h
)

install(TARGETS pointcaster_config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION lib
  # import libs are link-time only, keep them out of the runtime install
  ARCHIVE DESTINATION lib EXCLUDE_FROM_ALL
)
