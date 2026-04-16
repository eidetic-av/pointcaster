vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO DScudeler/gizmo-3d
  REF "${VERSION}"
  SHA512 38dd2fb5f03215bc0fb95388306eaac3488f64a044dd4bc54f71aa067c68055bf07ca283500323e3f96696c7b96ac710246f8b4f18a1989478c2264feaf2e72f
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