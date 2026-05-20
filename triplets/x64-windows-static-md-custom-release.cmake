set(VCPKG_BUILD_TYPE release)

include(${CMAKE_CURRENT_LIST_DIR}/x64-windows-static-md-custom.cmake)

set(VCPKG_CXX_FLAGS_RELEASE "/O2 /arch:AVX512 /fp:fast")
set(VCPKG_C_FLAGS_RELEASE "/O2 /arch:AVX512 /fp:fast")
set(VCPKG_LINKER_FLAGS_RELEASE "/LTCG")

if(PORT MATCHES "opencv4")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS
        -DENABLE_FAST_MATH=ON
        -DCUDA_FAST_MATH=ON
    )
endif()