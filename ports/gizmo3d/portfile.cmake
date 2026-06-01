vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO DScudeler/gizmo-3d
  REF "${VERSION}"
  SHA512 daa0dd424d4e07439cd14fe1575d32531fdb63db003d9fca3c5293acd9777ecdf8f52101855b9f91c7933a0c4af5f35943f6b1940ed308e291961cfa85730ec8
  HEAD_REF main
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
    PACKAGE_NAME gizmo3d 
    CONFIG_PATH "lib/cmake/gizmo3d")

file(
  INSTALL "${SOURCE_PATH}/LICENSE.md"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")