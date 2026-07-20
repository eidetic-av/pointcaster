# some vendored third-party patches prevent debug builds of the orbbec sdk here
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO orbbec/OrbbecSDK_v2
    REF "v${VERSION}"
    SHA512 3458a21bbdd3d9c7b011f8535a4f0ec7eddb82df9f6bc8887501597063326c1dda9f99f48e08ad559726bb56cb9b795d98adf5c1dfb1775e5b3d07a2f1e4c4f9
    HEAD_REF main
    PATCHES
        cmake-package-install.patch
        ros-initialisation-for-gcc.patch
        use-vcpkg-spdlog.patch
)

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    message(FATAL_ERROR "orbbecsdk-v2 only supports shared builds")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DOB_BUILD_MAIN_PROJECT=ON
        -DOB_BUILD_EXAMPLES=OFF
        -DOB_BUILD_TESTS=OFF
        -DOB_BUILD_DOCS=OFF
        -DOB_BUILD_TOOLS=OFF
        -DOB_INSTALL_EXAMPLES_SOURCE=OFF
        # vcpkg_install_copyright() below already collects licenses:
        -DOB_INSTALL_LICENSES=OFF
)

vcpkg_cmake_build()
vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# remove uneeded file not needed for runtime use
file(REMOVE
    "${CURRENT_PACKAGES_DIR}/setup.sh"
    "${CURRENT_PACKAGES_DIR}/debug/setup.sh"
)

vcpkg_cmake_config_fixup(
    CONFIG_PATH lib/cmake/OrbbecSDK
    PACKAGE_NAME OrbbecSDK
)

# runtime 'extensions' (filters, frameprocessor, depthengine, firmwareupdater) are
# loaded by libobsensor at runtime... keep them alongside other runtime libs
# (different directories for windows and linux though)

if(VCPKG_TARGET_IS_WINDOWS)
  set(bin_dir "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/win_x64/bin")
  set(runtime_extensions_dir "${bin_dir}/extensions")

  if(EXISTS "${runtime_extensions_dir}")
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/bin")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${runtime_extensions_dir}"
              "${CURRENT_PACKAGES_DIR}/bin/extensions"
    )

    # copy all DLLs adjacent to OrbbecSDK.dll
    file(GLOB runtime_dlls "${bin_dir}/*.dll")
    foreach(dll_path IN LISTS runtime_dlls)
      get_filename_component(dll_filename "${dll_path}" NAME)

      if(NOT dll_filename MATCHES "^OrbbecSDK.*\\.dll$")
        file(COPY "${dll_path}" DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
      endif()
    endforeach()
  endif()
else()

  if(EXISTS "${CURRENT_PACKAGES_DIR}/lib/extensions" AND NOT EXISTS "${CURRENT_PACKAGES_DIR}/debug/lib/extensions")
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/lib")
    file(COPY "${CURRENT_PACKAGES_DIR}/lib/extensions" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
  endif()

endif()

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
