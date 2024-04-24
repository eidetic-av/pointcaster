vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO gocarlos/mdns_cpp
    REF 05b181ca2b3920b787287a291fae326ecd0ef019
    SHA512 645e28d76848209abdf31326f903e28774bcbc818b093f63dda7303dcd8d5813b536d2f0c84f1d6415323f94eecf92e5e285cbd4b1d1e6676a5c84caca90b221
    HEAD_REF master
)


vcpkg_cmake_configure( SOURCE_PATH "${SOURCE_PATH}")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
  PACKAGE_NAME "mdns_cpp"
  CONFIG_PATH "share/mdns_cpp/cmake"
)

# file(REMOVE_RECURSE
#     "${CURRENT_PACKAGES_DIR}/debug/cmake"
#     "${CURRENT_PACKAGES_DIR}/debug/include"
#     "${CURRENT_PACKAGES_DIR}/lib/cmake")

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
