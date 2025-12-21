# ----- OpenCV CUDA libs -----

set(OPENCV_CUDA_PLUGIN_DIR "$<TARGET_FILE_DIR:pointcaster>/plugins/opencv_cuda")

corrade_add_plugin(CudaAnalyser2D
  ${OPENCV_CUDA_PLUGIN_DIR}
  ${OPENCV_CUDA_PLUGIN_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/opencv_cuda/CudaAnalyser2D.conf
  ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/opencv_cuda/cuda_analyser_2d.cc
)
set_target_properties(CudaAnalyser2D PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${OPENCV_CUDA_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${OPENCV_CUDA_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${OPENCV_CUDA_PLUGIN_DIR}
)
target_link_libraries(CudaAnalyser2D PRIVATE
  Corrade::PluginManager opencv_world
  opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc
  opencv_cudaoptflow opencv_cudawarping
)
add_custom_command(TARGET CudaAnalyser2D POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OPENCV_CUDA_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/opencv_cuda/CudaAnalyser2D.conf"
  "${OPENCV_CUDA_PLUGIN_DIR}/CudaAnalyser2D.conf"
)
add_custom_command(TARGET CudaAnalyser2D POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OPENCV_CUDA_PLUGIN_DIR}/deps"
  COMMAND ${CMAKE_COMMAND}
  -DINPUT_EXECUTABLE="$<TARGET_FILE:CudaAnalyser2D>"
  -DOUTPUT_DIRECTORY="${OPENCV_CUDA_PLUGIN_DIR}/deps"
  -DVCPKG_INSTALLED_DIR="${VCPKG_INSTALLED_DIR}"
  -DVCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET}"
  -DCUDA_BIN_DIR="${CUDAToolkit_BIN_DIR}"
  -DMOVE_EXISTING_DEPS=ON
  -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeDependencies.cmake"
)

list(APPEND POINTCASTER_DEPENDENCIES CudaAnalyser2D)

# ----- Azure Kinect Body Tracking plugin -----

if(WITH_K4A)
  set(K4ABT_PLUGIN_DIR "$<TARGET_FILE_DIR:pointcaster>/plugins/k4abt")
  set(K4ABT_PLUGIN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/k4abt")

  find_package(k4abt REQUIRED)

  corrade_add_plugin(AzureKinectBodyTracking
    ${K4ABT_PLUGIN_DIR}
    ${K4ABT_PLUGIN_DIR}
    ${K4ABT_PLUGIN_SOURCE_DIR}/k4abt.conf
    ${K4ABT_PLUGIN_SOURCE_DIR}/k4a_body_tracking.cc
  )
  set_target_properties(AzureKinectBodyTracking PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${K4ABT_PLUGIN_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${K4ABT_PLUGIN_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${K4ABT_PLUGIN_DIR}
  )
  target_link_libraries(AzureKinectBodyTracking PRIVATE
    Corrade::PluginManager k4a::k4a k4a::k4abt
  )
  add_custom_command(TARGET AzureKinectBodyTracking POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${K4ABT_PLUGIN_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${K4ABT_PLUGIN_SOURCE_DIR}/k4abt.conf"
    "${K4ABT_PLUGIN_DIR}/k4abt.conf"
  )
  add_custom_command(TARGET AzureKinectBodyTracking POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${K4ABT_PLUGIN_DIR}/deps"
    COMMAND ${CMAKE_COMMAND}
    -DINPUT_EXECUTABLE="$<TARGET_FILE:AzureKinectBodyTracking>"
    -DOUTPUT_DIRECTORY="${K4ABT_PLUGIN_DIR}/deps"
    -DVCPKG_INSTALLED_DIR="${VCPKG_INSTALLED_DIR}"
    -DVCPKG_TARGET_TRIPLET="${VCPKG_TARGET_TRIPLET}"
    -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeDependencies.cmake"
  )

  foreach(K4ABT_BIN ${k4abt_RUNTIME_DEPENDENCIES})
    add_custom_command(TARGET AzureKinectBodyTracking POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${K4ABT_BIN} "${K4ABT_PLUGIN_DIR}/deps")
  endforeach()

  list(APPEND POINTCASTER_DEPENDENCIES AzureKinectBodyTracking)
endif()
