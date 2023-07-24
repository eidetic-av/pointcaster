vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO eidetic-av/pointclouds
  REF 8f6074e4957403372741a4a343ba14a4b10ede86
  SHA512 7bde355a11517f6f306c8ec35cade102fd6d959bc923988d3a7d75e8d6693ca3d37790e819089abc90d75df90d0b38a17d41d81697c73645e70cd67663781aa2
  HEAD_REF main
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS -DBUILD_TESTS=off)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "share/${PORT}/cmake/")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
