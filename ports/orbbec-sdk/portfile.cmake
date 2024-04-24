vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO orbbec/OrbbecSDK
  REF "v${VERSION}"
  SHA512 72874b8bea22812aeb67fff8de01c0daaceb56112f348db522e31be7c7dc1bda2a067f166d40ceb2352076b4b7c8eeddf13f166e7d8f6eda49d432fb7069dc3c
  HEAD_REF main
)

file(INSTALL
  ${SOURCE_PATH}/lib/linux_x64/
  DESTINATION ${CURRENT_PACKAGES_DIR}/lib
  FILES_MATCHING PATTERN "*.so*")

file(INSTALL
  ${SOURCE_PATH}/include/
  DESTINATION ${CURRENT_PACKAGES_DIR}/include)

file(INSTALL
  ${SOURCE_PATH}/cmake/
  DESTINATION ${CURRENT_PACKAGES_DIR}/share/OrbbecSDK)

configure_file(
    ${CURRENT_PACKAGES_DIR}/share/OrbbecSDK/OrbbecSDKConfig.cmake.in
    ${CURRENT_PACKAGES_DIR}/share/OrbbecSDK/OrbbecSDKConfig.cmake
    @ONLY)

file(INSTALL ${SOURCE_PATH}/LICENSE.txt 
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME "copyright")
