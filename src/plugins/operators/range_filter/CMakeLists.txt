set(RANGE_FILTER_PLUGIN_DIR ${OPERATORS_PLUGIN_DIR}/range_filter)

corrade_add_plugin(
  RangeFilterOperator
  ${RANGE_FILTER_PLUGIN_DIR}
  ${RANGE_FILTER_PLUGIN_DIR}
  RangeFilterOperator.conf
  range_filter_operator.cc
)

set_target_properties(RangeFilterOperator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${RANGE_FILTER_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${RANGE_FILTER_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${RANGE_FILTER_PLUGIN_DIR}
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/range_filter"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/range_filter"
  AUTOMOC OFF
)

target_link_libraries(RangeFilterOperator PRIVATE
  pointcaster
  pointcaster::core
  pointcaster::pipeline
  Corrade::PluginManager
)

target_include_directories(RangeFilterOperator PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
)

add_custom_command(TARGET RangeFilterOperator POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${RANGE_FILTER_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/RangeFilterOperator.conf"
  "${RANGE_FILTER_PLUGIN_DIR}/RangeFilterOperator.conf"
)

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} RangeFilterOperator PARENT_SCOPE)
