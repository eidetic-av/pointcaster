vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO orbbec/OrbbecSDK
  REF "v${VERSION}"
  SHA512 ae3866246641c1786d51fafc6cd4ae60c8946e5b2c881c1122830623051e21782f74d46cd9cf08c55e487af6d8961bca769ce0e5bcc1377683ad5e24945e15c0
  HEAD_REF main
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

# TODO
# some predistributed dlls come even when using static builds?
set(VCPKG_POLICY_DLLS_IN_STATIC_LIBRARY enabled)

# the depth engine library will come from the k4a repo instead,
# which works with both k4a and orbbec devices. So skip it for this port

file(GLOB_RECURSE LIB_FILES "${SOURCE_PATH}/lib/${LIB_DIR}/*")

file(GLOB_RECURSE LIB_FILES
     "${SOURCE_PATH}/lib/${LIB_DIR}/*")

foreach(LIB_FILE IN LISTS LIB_FILES)
  get_filename_component(FILE_NAME ${LIB_FILE} NAME)
  if (NOT FILE_NAME MATCHES "depthengine")
    # orbbec sdk doesnt ship debug dlls, just use release in both configs
    if (FILE_NAME MATCHES "\\.dll$")
      file(INSTALL ${LIB_FILE}
                   DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
      file(INSTALL ${LIB_FILE}
                   DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)
    elseif (FILE_NAME MATCHES "\\.lib$")
      file(INSTALL ${LIB_FILE}
                   DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
      file(INSTALL ${LIB_FILE}
                   DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)
      file(INSTALL ${LIB_FILE}
                   DESTINATION
                     "${CURRENT_PACKAGES_DIR}/share/OrbbecSDK/lib/${LIB_DIR}")
    endif()
  endif()
endforeach()

file(INSTALL ${SOURCE_PATH}/LICENSE.txt 
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME "copyright")
