vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO covexp/cuda-noise
    REF master
    SHA512 6987f225c49b8da4e2b0a73512d52b44abb48e750ab40479ba48e22fda232bc470f3bf3a4e0406117609e31569133eeb823e7c2972f806783269809f971c4a76
    HEAD_REF master
)

# This is a header only library
file(INSTALL "${SOURCE_PATH}/include/cuda_noise.cuh" DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME "copyright")
