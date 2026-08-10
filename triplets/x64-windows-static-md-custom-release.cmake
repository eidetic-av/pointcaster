set(VCPKG_BUILD_TYPE release)

include(${CMAKE_CURRENT_LIST_DIR}/x64-windows-static-md-custom.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/x64-all-platforms-release.cmake)

set(VCPKG_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /fp:fast")
set(VCPKG_C_FLAGS_RELEASE "/O2 /arch:AVX2 /fp:fast")
set(VCPKG_LINKER_FLAGS_RELEASE "/LTCG")