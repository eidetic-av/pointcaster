vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO jcelerier/libremidi
  REF 7edfe056e5ab7c7a5e0df443156f2fe83095ed4f
  SHA512 50242a9828ae6cfa1b86a735c07a351da58dced5e3578391b461f7537a9ba3942e92f78a7c8b56dd85c55cfca52584ef5562c99fabcc75017e3dc464c52d8016
  HEAD_REF master
  PATCHES fix-cmake-exports.patch
)

vcpkg_configure_cmake(
  SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_install_cmake()
vcpkg_fixup_cmake_targets(
  CONFIG_PATH "lib/cmake/${PORT}"
)

# if dynamic linkage, remove static directories
if (${VCPKG_LIBRARY_LINKAGE} EQUAL dynamic)
  SET(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)
  # TODO for release folder too
  file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}\debug\lib\static" "${CURRENT_PACKAGES_DIR}\lib\static")
endif()

file(GLOB LIBREMIDI_HEADERS
  "${SOURCE_PATH}/include/libremidi/*.hpp")
foreach(header ${LIBREMIDI_HEADERS})
file(
  COPY "${header}"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include/libremidi/"
)
endforeach(header)

file(GLOB LIBREMIDI_DETAIL_HEADERS
  "${SOURCE_PATH}/include/libremidi/detail/*.hpp")
foreach(header ${LIBREMIDI_DETAIL_HEADERS})
file(
  COPY "${header}"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include/libremidi/detail/"
)
endforeach(header)

file(
  INSTALL "${SOURCE_PATH}/LICENSE.ModernMidi"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
)
file(
  INSTALL "${SOURCE_PATH}/LICENSE.RtMidi"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
)
