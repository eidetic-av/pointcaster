set(VCPKG_BUILD_TYPE release)

include(${CMAKE_CURRENT_LIST_DIR}/x64-linux-custom.cmake)

set(VCPKG_CXX_FLAGS_RELEASE "-O3 -march=skylake-avx512 -ffast-math")
set(VCPKG_C_FLAGS_RELEASE "-O3 -march=skylake-avx512 -ffast-math")

if(PORT MATCHES "opencv4")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS
        -DENABLE_FAST_MATH=ON
        -DCUDA_FAST_MATH=ON
    )
endif()