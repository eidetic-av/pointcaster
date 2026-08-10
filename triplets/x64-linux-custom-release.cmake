set(VCPKG_BUILD_TYPE release)

include(${CMAKE_CURRENT_LIST_DIR}/x64-linux-custom.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/x64-all-platforms-release.cmake)

set(VCPKG_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v3 -ffast-math")
set(VCPKG_C_FLAGS_RELEASE "-O3 -march=x86-64-v3 -ffast-math")