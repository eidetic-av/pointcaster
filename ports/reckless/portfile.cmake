vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO mattiasflodin/reckless
  REF v3.0.3
  SHA512 63122af0d6dc6e8097c711cb119d818cf19936caba510cec4bd1f1414d204cddc595e0dd9eeb872dadd2d113c1a2cb573182228200cbdaa603341e2252a93ab7
  HEAD_REF master
  PATCHES add-install-targets.patch
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/${PORT}")

file(
  INSTALL "${SOURCE_PATH}/LICENSE.txt"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
)
