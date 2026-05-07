set(FRINGE_REMOVAL_PLUGIN_DIR ${OPERATORS_PLUGIN_DIR}/fringe_removal)

corrade_add_plugin(
  FringeRemovalOperator
  ${FRINGE_REMOVAL_PLUGIN_DIR}
  ${FRINGE_REMOVAL_PLUGIN_DIR}
  FringeRemovalOperator.conf
  fringe_removal_operator.cc
)

set_target_properties(FringeRemovalOperator PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY ${FRINGE_REMOVAL_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${FRINGE_REMOVAL_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${FRINGE_REMOVAL_PLUGIN_DIR}
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/fringe_removal"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/fringe_removal"
  AUTOMOC OFF
)

find_package(PCL CONFIG REQUIRED)

target_link_libraries(FringeRemovalOperator PRIVATE
  pointcaster::core
  Corrade::PluginManager
  ${PCL_LIBRARIES}
)

target_include_directories(FringeRemovalOperator 
  PRIVATE 
    "${CMAKE_SOURCE_DIR}/src" 
    "${CMAKE_SOURCE_DIR}/include"
)

add_custom_command(TARGET FringeRemovalOperator POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${FRINGE_REMOVAL_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/FringeRemovalOperator.conf"
  "${FRINGE_REMOVAL_PLUGIN_DIR}/FringeRemovalOperator.conf"
)

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} FringeRemovalOperator PARENT_SCOPE)

# target_precompile_headers(FringeRemovalOperator PRIVATE
#   <string>
#   <string_view>
#   <vector>
#   <memory>
#   <mutex>
#   <thread>
#   <chrono>
#   <atomic>
#   <optional>
# )
