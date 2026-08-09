set(CLUSTER_EXTRACTION_PLUGIN_DIR ${OPERATORS_PLUGIN_DIR}/cluster_extraction)

corrade_add_plugin(
  ClusterExtractionOperator
  ${CLUSTER_EXTRACTION_PLUGIN_DIR}
  ${CLUSTER_EXTRACTION_PLUGIN_DIR}
  ClusterExtractionOperator.conf
  cluster_extraction_operator.cc
)

set_target_properties(ClusterExtractionOperator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CLUSTER_EXTRACTION_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CLUSTER_EXTRACTION_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CLUSTER_EXTRACTION_PLUGIN_DIR}
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/cluster_extraction"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/cluster_extraction"
  AUTOMOC OFF
)

find_package(PCL CONFIG REQUIRED)

target_link_libraries(ClusterExtractionOperator PRIVATE
  pointcaster
  pointcaster::core
  pointcaster::pipeline
  Corrade::PluginManager
  ${PCL_LIBRARIES}
)

target_include_directories(ClusterExtractionOperator PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
)

add_custom_command(TARGET ClusterExtractionOperator POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CLUSTER_EXTRACTION_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/ClusterExtractionOperator.conf"
  "${CLUSTER_EXTRACTION_PLUGIN_DIR}/ClusterExtractionOperator.conf"
)

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} ClusterExtractionOperator PARENT_SCOPE)

# target_precompile_headers(ClusterExtractionOperator PRIVATE
# <algorithm>
# <cmath>
# <cstdint>
# <limits>
# <numbers>
# <vector>
# <string>
# <opencv2/core.hpp>
# <opencv2/imgproc.hpp>
# <pcl/features/organized_edge_detection.h>
# <pcl/point_cloud.h>
# <pcl/point_types.h>
# )
