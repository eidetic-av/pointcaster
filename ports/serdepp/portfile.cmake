vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO injae/serdepp
    REF 5c01ca3a36a1e41918f87d8b52e3a138c3ed9644
    SHA512 1b5a30e58ce78f302678666dee0b363adfd60072a0ad9c8113a28fd5f8f0b26e4e0a90154002ac0f8e653ce8f51d44486511a4070fb01d46e256fbe72273aa28
    HEAD_REF main
)


vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DSERDEPP_BUILD_TESTING=OFF
        -DENABLE_INLINE_CPPM_TOOLS=ON
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/serdepp)

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/cmake"
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/lib/cmake"
)

# # Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
