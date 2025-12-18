if(NOT DEFINED INPUT_EXECUTABLE OR NOT EXISTS "${INPUT_EXECUTABLE}")
  message(FATAL_ERROR "INPUT_EXECUTABLE not set: '${INPUT_EXECUTABLE}'")
endif()

if(NOT DEFINED OUTPUT_DIRECTORY)
  message(FATAL_ERROR "OUTPUT_DIRECTORY not set")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")

set(search_dirs
  "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
  "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin"
)

if(DEFINED CUDA_BIN_DIR AND EXISTS "${CUDA_BIN_DIR}")
  list(APPEND search_dirs "${CUDA_BIN_DIR}")
endif()

file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${INPUT_EXECUTABLE}"
  RESOLVED_DEPENDENCIES_VAR resolved_deps
  UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
  DIRECTORIES ${search_dirs}
  # don't copy in windows system libs
  PRE_EXCLUDE_REGEXES
    "api-ms-win-.*"
    "ext-ms-.*"
  POST_EXCLUDE_REGEXES
    ".*[\\\\/]Windows[\\\\/].*"
    ".*[\\\\/]System32[\\\\/].*"
)

list(SORT resolved_deps)

message(STATUS "Resolved runtime dependencies for: ${INPUT_EXECUTABLE}")
foreach(dll IN LISTS resolved_deps)
  message(STATUS "  ${dll}")

  get_filename_component(dll_name "${dll}" NAME)
  file(COPY_FILE
    "${dll}"
    "${OUTPUT_DIRECTORY}/${dll_name}"
    ONLY_IF_DIFFERENT
  )
endforeach()

if(unresolved_deps)
  list(SORT unresolved_deps)
  message(WARNING "Unresolved runtime dependencies:")
  foreach(dep IN LISTS unresolved_deps)
    message(WARNING "  ${dep}")
  endforeach()
endif()
