vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO jcelerier/libremidi
  REF e935d45e1e5a4dc5d5e076996737f5998b3ed341
  SHA512 fee0a26adb5753eb6d7939dec7e19325b74a392d68c5b14c3a7a2c4a2635ea392303fd54d1b2cb0f0a56c5dc81b504b9fe8c70f5a46aaf850c761d44e819a04a
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
  RENAME copyright.modernmidi
)
file(
  INSTALL "${SOURCE_PATH}/LICENSE.RtMidi"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright.rtmidi
)
