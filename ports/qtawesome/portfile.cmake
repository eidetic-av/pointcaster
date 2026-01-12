vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO gamecreature/QtAwesome
    REF ${VERSION}
    SHA512 334e36c5729d2433195109bf1d8696b947eb173de32791d9557b9de9dcef4f253fe3458daabf7bfc4007a2156479bd11aeb911702ad656749e17437a01dcdcef
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME QtAwesome
    CONFIG_PATH "lib/cmake/QtAwesome"
)

file(INSTALL
    "${SOURCE_PATH}/LICENSE.md"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME copyright
)

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/share/licenses"
    "${CURRENT_PACKAGES_DIR}/debug/share/licenses"
)

vcpkg_copy_pdbs()