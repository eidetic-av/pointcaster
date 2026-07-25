# prebuilt binary sdk

set(VCPKG_BUILD_TYPE release)
set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)
set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO orbbec/OrbbecSDK
    REF "v${VERSION}"
    SHA512 9e166d277910f4f811b094e1cf8b76553eee16d1b90815233c5eb90c50ccd03eef185fba51b8c029dbd6c9aa0fb97c8560ecc5c092dfad4305e10cf8750d53e7
    HEAD_REF main
)

file(INSTALL "${SOURCE_PATH}/include/"
    DESTINATION "${CURRENT_PACKAGES_DIR}/include/orbbecsdk-v1")

if(VCPKG_TARGET_IS_LINUX)
    set(lib_dir "linux_x64")
elseif(VCPKG_TARGET_IS_WINDOWS)
    set(lib_dir "win_x64")
else()
    message(FATAL_ERROR "Platform not supported.")
endif()

# runtime libs beside the sdk lib (depthengine, ob_usb, live555) are dlopened
# by libOrbbecSDK relative to itself, so keep the whole directory together
file(GLOB_RECURSE sdk_binaries "${SOURCE_PATH}/lib/${lib_dir}/*")
foreach(sdk_binary IN LISTS sdk_binaries)
    get_filename_component(sdk_binary_name "${sdk_binary}" NAME)
    if(sdk_binary_name MATCHES "\\.dll$")
        file(COPY "${sdk_binary}" DESTINATION "${CURRENT_PACKAGES_DIR}/bin/orbbecsdk-v1")
    elseif(sdk_binary_name MATCHES "\\.lib$" OR sdk_binary_name MATCHES "\\.so")
        file(COPY "${sdk_binary}" DESTINATION "${CURRENT_PACKAGES_DIR}/lib/orbbecsdk-v1")
    endif()
endforeach()

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/OrbbecSDKv1Config.cmake"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/orbbecsdkv1")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
