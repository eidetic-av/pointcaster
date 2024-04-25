vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO orbbec/OrbbecSDK
  REF "v${VERSION}"
  SHA512 72874b8bea22812aeb67fff8de01c0daaceb56112f348db522e31be7c7dc1bda2a067f166d40ceb2352076b4b7c8eeddf13f166e7d8f6eda49d432fb7069dc3c
  HEAD_REF main
  PATCHES fix-lib-paths.patch
)

file(INSTALL
  ${SOURCE_PATH}/include/
  DESTINATION ${CURRENT_PACKAGES_DIR}/include)

file(INSTALL
    ${SOURCE_PATH}/OrbbecSDKConfig.cmake
    DESTINATION ${CURRENT_PACKAGES_DIR}/share/OrbbecSDK)

file(INSTALL
  ${SOURCE_PATH}/cmake/
  DESTINATION ${CURRENT_PACKAGES_DIR}/share/OrbbecSDK/cmake)

# platform specific var setup

if (VCPKG_TARGET_IS_LINUX)
  set(LIB_DIR "linux_x64")
elseif(VCPKG_TARGET_IS_WINDOWS)
  set(LIB_DIR "win_x64")
else()
  message(FATAL_ERROR "Platform not supported.")
endif()

# the depth engine library will come from the k4a repo instead,
# which works with both k4a and orbbec devices. So skip it for this port

file(GLOB_RECURSE LIB_FILES "${SOURCE_PATH}/lib/${LIB_DIR}/*")

foreach(LIB_FILE ${LIB_FILES})
  get_filename_component(FILE_NAME ${LIB_FILE} NAME)
  if (NOT FILE_NAME MATCHES "depthengine")
    if(FILE_NAME MATCHES "dll")
      file(INSTALL ${LIB_FILE}
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    else()
      file(INSTALL ${LIB_FILE}
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    endif()
  endif()
endforeach()

file(INSTALL ${SOURCE_PATH}/LICENSE.txt 
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME "copyright")
