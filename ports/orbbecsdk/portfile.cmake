if("${VCPKG_LIBRARY_LINKAGE}" STREQUAL "static")
  message(STATUS "Note: ${PORT} only supports dynamic library linkage. Building dynamic library.")
  set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(VCPKG_TARGET_IS_LINUX)

  set(LINUX_ARCHIVE_FILE_NAME "OrbbecSDK_v2.6.3_202512231427_4e448f9_linux_x86_64.zip")

  vcpkg_download_distfile(ARCHIVE_FILE
    URLS "https://github.com/orbbec/OrbbecSDK_v2/releases/download/v${VERSION}/${LINUX_ARCHIVE_FILE_NAME}"
    FILENAME "${LINUX_ARCHIVE_FILE_NAME}"
    SHA512 3919f234c373d8ea5cb30e69bc007264f78ece702f3a6b6f648edf727e25e5f1802a8bcebb52ff52be07af9911a438ede1f7eb7379fcf871e14a5d0000a4a1d2
  )

  vcpkg_extract_source_archive(
    SOURCE_PATH
    ARCHIVE "${ARCHIVE_FILE}"
    SOURCE_BASE ${VERSION}
  )

elseif(VCPKG_TARGET_IS_WINDOWS)

  set(WINDOWS_ARCHIVE_FILE_NAME "OrbbecSDK_v2.6.3_202512232226_4e448f9_win_x64.zip")

  vcpkg_download_distfile(ARCHIVE_FILE
    URLS "https://github.com/orbbec/OrbbecSDK_v2/releases/download/v${VERSION}/${WINDOWS_ARCHIVE_FILE_NAME}"
    FILENAME "${WINDOWS_ARCHIVE_FILE_NAME}"
    SHA512 2372a22cedc40252babd8f66f0fa84c08218bfa630cde2a39f3d567edfb24877caa775aa855b8d1e98f267c03068a3d3637b1b9e0ac6861b061dda5ae46db357
  )

  vcpkg_extract_source_archive(
    SOURCE_PATH
    ARCHIVE "${ARCHIVE_FILE}"
    SOURCE_BASE ${VERSION}
    NO_REMOVE_ONE_LEVEL
  )

endif()

file(INSTALL "${SOURCE_PATH}/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/lib/" DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${SOURCE_PATH}/lib/" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
file(INSTALL "${SOURCE_PATH}/bin/" DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
file(INSTALL "${SOURCE_PATH}/bin/" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin")
file(INSTALL "${SOURCE_PATH}/shared/" DESTINATION "${CURRENT_PACKAGES_DIR}/shared")

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/OrbbecSDKConfig.cmake.in"
  "${CURRENT_PACKAGES_DIR}/share/${PORT}/OrbbecSDKConfig.cmake"
  @ONLY
)
file(COPY "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
