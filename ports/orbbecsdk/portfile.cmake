vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO orbbec/OrbbecSDK
  REF "v${VERSION}"
  SHA512 9e166d277910f4f811b094e1cf8b76553eee16d1b90815233c5eb90c50ccd03eef185fba51b8c029dbd6c9aa0fb97c8560ecc5c092dfad4305e10cf8750d53e7
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

# ***MOVING DEPTH ENGINE FOR NOW***  the depth engine library will come from the k4a repo instead,
# which works with both k4a and orbbec devices. So skip it for this port

file(GLOB_RECURSE LIB_FILES "${SOURCE_PATH}/lib/${LIB_DIR}/*")

file(GLOB_RECURSE LIB_FILES
     "${SOURCE_PATH}/lib/${LIB_DIR}/*")

foreach(LIB_FILE IN LISTS LIB_FILES)
  get_filename_component(FILE_NAME ${LIB_FILE} NAME)
#   if (NOT FILE_NAME MATCHES "depthengine")
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
#   endif()
endforeach()

file(INSTALL ${SOURCE_PATH}/LICENSE.txt 
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME "copyright")