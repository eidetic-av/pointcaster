vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO eidetic-av/pointclouds
  REF b0efd691d9219738dcde756c7bd1fdab3096af4a
  SHA512 8e8e84ff364625063d2b6a035197a838404e93f7942ae27eb9a2cb5c845994379ca8aead136b52cf310e0925ae0374e1dc73c2b0ebacd41bd281331010ca556d
  HEAD_REF main
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS -DBUILD_TESTS=off)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "share/${PORT}/cmake/")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
