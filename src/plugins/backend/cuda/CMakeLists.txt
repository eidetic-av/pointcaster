set(CUDA_PLUGIN_DIR ${BACKEND_PLUGIN_DIR}/cuda)

corrade_add_plugin(CudaBackend
  ${BACKEND_PLUGIN_DIR}
  ${BACKEND_PLUGIN_DIR}
  CudaBackend.conf
  cuda_backend.cc
)

set_target_properties(CudaBackend PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${BACKEND_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${BACKEND_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${BACKEND_PLUGIN_DIR}
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/cuda"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/cuda"
  AUTOMOC OFF
)

find_package(CUDAToolkit REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

target_link_libraries(CudaBackend PRIVATE
  pointcaster::core
  pointcaster::metrics
  Corrade::PluginManager
  CUDA::cudart_static 
  CUDA::cuda_driver
  Thrust::Thrust
)

target_include_directories(CudaBackend 
  PRIVATE 
    "${CMAKE_SOURCE_DIR}/src" 
    "${CMAKE_SOURCE_DIR}/include"
)

add_custom_command(TARGET CudaBackend POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CUDA_PLUGIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_CURRENT_SOURCE_DIR}/CudaBackend.conf"
  "${CUDA_PLUGIN_DIR}/CudaBackend.conf"
)

set(CUDA_INSTALLED_LIB_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(CUDA_INSTALLED_LIB_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
endif()

file(GLOB CUDA_RUNTIME_DEPENDENCIES "${CUDA_INSTALLED_LIB_DIR}/*cuda*")

install(FILES ${CUDA_RUNTIME_DEPENDENCIES} DESTINATION ${CUDA_PLUGIN_DIR})

install(DIRECTORY "${CUDA_INSTALLED_LIB_DIR}/extensions" DESTINATION ${CUDA_PLUGIN_DIR})

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} CudaBackend PARENT_SCOPE)

target_precompile_headers(CudaBackend PRIVATE
  <Corrade/Containers/Array.h>
  <Corrade/Containers/GrowableArray.h>
  <Corrade/Containers/String.h>
  <Corrade/Containers/StringView.h>
  <Corrade/PluginManager/AbstractPlugin.h>
)
