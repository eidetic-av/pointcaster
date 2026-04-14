corrade_add_static_plugin(
  CpuBackend
  CpuBackend
  ""
  CpuBackend.conf
  cpu_backend.cc
)

set_target_properties(CpuBackend PROPERTIES AUTOMOC OFF)

find_package(TBB REQUIRED)

target_link_libraries(CpuBackend PRIVATE
  pointcaster::core
  pointcaster::metrics
  Corrade::PluginManager
  TBB::tbb
)

find_path(BSHOSHANY_THREAD_POOL_INCLUDE_DIRS "BS_thread_pool.hpp")

target_include_directories(CpuBackend 
  PRIVATE 
    "${CMAKE_SOURCE_DIR}/src" 
    "${CMAKE_SOURCE_DIR}/include"
    ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS}
)

# remove the installed directory because its static, we don't need it
install(CODE [===[
  file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}/CpuBackend/")
]===])

set(CORRADE_STATIC_PLUGINS ${CORRADE_STATIC_PLUGINS} CpuBackend PARENT_SCOPE)
set(CORRADE_PLUGINS ${CORRADE_PLUGINS} CpuBackend PARENT_SCOPE)

# target_precompile_headers(CpuBackend PRIVATE
# )
