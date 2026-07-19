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

if (NOT WIN32)
  # this quiets some very noisy opencv warnings
  set_source_files_properties(fringe_removal_operator.cc PROPERTIES
    COMPILE_OPTIONS "-Wno-old-style-cast;-Wno-zero-as-null-pointer-constant"
  )
endif()

find_package(PCL CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED
  COMPONENTS
  opencv_core
  opencv_imgproc
)

target_link_libraries(FringeRemovalOperator PRIVATE
  pointcaster
  pointcaster::core
  pointcaster::pipeline
  Corrade::PluginManager
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

target_include_directories(FringeRemovalOperator PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
  ${OpenCV_INCLUDE_DIRS}
)

add_custom_command(TARGET FringeRemovalOperator POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${FRINGE_REMOVAL_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/FringeRemovalOperator.conf"
  "${FRINGE_REMOVAL_PLUGIN_DIR}/FringeRemovalOperator.conf"
)

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} FringeRemovalOperator PARENT_SCOPE)

# target_precompile_headers(FringeRemovalOperator PRIVATE
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
