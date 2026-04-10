# Corrade doesn't play nice with nvcc so we have an nvcc-compiled 
# library that contains the actual CUDA translation units, and also
# the corrade plugin that links to that CUDA lib and is loaded at
# runtime

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)

set(CUDA_PLUGIN_DIR ${BACKEND_PLUGIN_DIR}/cuda)

add_library(cuda_kernels
  cuda_kernels.cu
)

set_target_properties(cuda_kernels PROPERTIES
  AUTOMOC OFF
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_PROPAGATE_HOST_FLAGS OFF
  # CUDA_ARCHITECTURES "75-real;86-real;89-real;89-virtual"
  CUDA_ARCHITECTURES "75-virtual"
)

if (WIN32)
  target_compile_options(cuda_kernels PRIVATE 
    /bigobj /utf-8 -forward-slash-prefix-opts)
endif()

# allow extra lambda features
target_compile_options(cuda_kernels PRIVATE --extended-lambda)

# these turn off noisy compiler output from dependencies
target_compile_options(cuda_kernels PRIVATE 
  --diag-suppress 20012,20013,20014,20015,27,1394,221)

corrade_add_plugin(CudaBackend
  ${CUDA_PLUGIN_DIR}
  ${CUDA_PLUGIN_DIR}
  CudaBackend.conf
  cuda_backend.cc
)

set_target_properties(CudaBackend PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CUDA_PLUGIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CUDA_PLUGIN_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CUDA_PLUGIN_DIR}
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/cuda"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/cuda"
  AUTOMOC OFF
)

find_package(Corrade CONFIG REQUIRED Main PluginManager)

find_package(CUDAToolkit REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

target_link_libraries(cuda_kernels PRIVATE
  Corrade::PluginManager
  CUDA::cudart_static 
  CUDA::cuda_driver
  Thrust::Thrust
)

target_link_libraries(CudaBackend PRIVATE
  cuda_kernels
  pointcaster::core
  pointcaster::metrics
  Corrade::PluginManager
)

target_include_directories(cuda_kernels 
  PRIVATE 
    "${CMAKE_SOURCE_DIR}/src" 
    "${CMAKE_SOURCE_DIR}/include"
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

# file(GLOB CUDA_RUNTIME_DEPENDENCIES "${CUDA_INSTALLED_LIB_DIR}/*cuda*")

# install(FILES ${CUDA_RUNTIME_DEPENDENCIES} DESTINATION ${CUDA_PLUGIN_DIR})

# install(DIRECTORY "${CUDA_INSTALLED_LIB_DIR}/extensions" DESTINATION ${CUDA_PLUGIN_DIR})

set(CORRADE_PLUGINS ${CORRADE_PLUGINS} CudaBackend PARENT_SCOPE)

target_precompile_headers(CudaBackend PRIVATE
  <Corrade/Containers/Array.h>
  <Corrade/Containers/GrowableArray.h>
  <Corrade/Containers/String.h>
  <Corrade/Containers/StringView.h>
  <Corrade/PluginManager/AbstractPlugin.h>
  <config/transform_config.h>
  <cpplocate/cpplocate.h>
  <filesystem>
  <functional>
  <tuple>
)
