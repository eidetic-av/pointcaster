vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO orbbec/OrbbecSDK_v2
    REF v${VERSION}
    SHA512 e8927b211d2d6d01483568d745b21e044f3486e7c1da9bcec1715651cfa8f26d671f2a30e2adfeddc5d54541a0d5c39ee8e01fc5b8c84fe25ff2a90985bf8ee4
    PATCHES
        cmake-package-install.patch
)

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    message(FATAL_ERROR "orbbecsdk_v2 only supports shared builds")
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
)

vcpkg_cmake_build()
vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_cmake_config_fixup(
    CONFIG_PATH lib/cmake/OrbbecSDK
    PACKAGE_NAME OrbbecSDK
)

if(VCPKG_TARGET_IS_WINDOWS)
  set(bin_dir "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/win_x64/bin")
  set(runtime_extensions_dir "${bin_dir}/extensions")

  if(EXISTS "${runtime_extensions_dir}")
    # copy 'extensions' that orbbec loads at runtime
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
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
