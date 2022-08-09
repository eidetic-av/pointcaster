vcpkg_from_git(
  OUT_SOURCE_PATH SOURCE_PATH
  URL https://git.sr.ht/~eidetic/bob-pointclouds
  REF d1b91c24bf3247340ad43b4ae62baeba0d558a67
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()

file(INSTALL "${SOURCE_PATH}/include/pointclouds.h"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include/bob")

configure_file("${CURRENT_PORT_DIR}/vcpkg-cmake-wrapper.cmake"
  "${CURRENT_PACKAGES_DIR}/share/${PORT}/vcpkg-cmake-wrapper.cmake" @ONLY)
configure_file("${CURRENT_PORT_DIR}/usage"
  "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage" @ONLY)
