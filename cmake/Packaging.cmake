include(GNUInstallDirs)

# desktop entry + icon
install(FILES
    "${CMAKE_SOURCE_DIR}/packaging/pointcaster.desktop"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/applications"
)
install(FILES
    "${CMAKE_SOURCE_DIR}/packaging/pointcaster-icon.png"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/256x256/apps"
)

# Qt/QML deploy script

qt_generate_deploy_qml_app_script(
    TARGET pointcaster
    OUTPUT_SCRIPT pointcaster_deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
    NO_TRANSLATIONS
)

install(SCRIPT "${pointcaster_deploy_script}")

# non-Qt runtime deps

install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES $<TARGET_FILE:pointcaster>
    RESOLVED_DEPENDENCIES_VAR resolved_deps
    POST_EXCLUDE_REGEXES
      "^/lib/x86_64-linux-gnu/libc\\.so\\."
      "^/lib/x86_64-linux-gnu/libm\\.so\\."
      "^/lib/x86_64-linux-gnu/libdl\\.so\\."
      "^/lib/x86_64-linux-gnu/libpthread\\.so\\."
      "^/lib/x86_64-linux-gnu/librt\\.so\\."
      "^/lib/x86_64-linux-gnu/libgcc_s\\.so\\."
      "^/lib/x86_64-linux-gnu/libresolv\\.so\\."
      "^/lib64/ld-linux-x86-64\\.so\\."
  )

  foreach(dep ${resolved_deps})
    file(COPY "${dep}" DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
  endforeach()
]])

# TODO: this is super hacky but this x11 dep is required
# to run the appimage
install(
    FILES "/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0.0.0"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RENAME "libxcb-cursor.so.0"
)
install(
    FILES "/usr/lib/x86_64-linux-gnu/libOpenGL.so.0.0.0"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RENAME "libOpenGL.so.0"
)

# set up for the AppImage runtime environment
# install(FILES
# "${CMAKE_SOURCE_DIR}/packaging/run-env.sh"
# DESTINATION apprun-hooks
# )
# install(
#    PROGRAMS "${CMAKE_SOURCE_DIR}/packaging/AppImageRuntime.sh"
#    DESTINATION "."
#    RENAME "AppRun"
# )

# move the real pointcaster executable to appdir/usr/libexec
# and place a shell script at appdir/usr/bin instead that
# sets env variables before launching the real pointcaster exe

# install(TARGETS pointcaster RUNTIME DESTINATION libexec)
# install(
# PROGRAMS "${CMAKE_SOURCE_DIR}/packaging/launch-pointcaster.sh"
# DESTINATION bin RENAME pointcaster 
# )
install(CODE "set(POINTCASTER_SOURCE_DIR \"${CMAKE_SOURCE_DIR}\")")
install(SCRIPT "${CMAKE_SOURCE_DIR}/cmake/InstallAppImageWrapper.cmake")

# CPack metadata
set(CPACK_PACKAGE_NAME "pointcaster")
set(CPACK_PACKAGE_VENDOR "Matt Hughes")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pointcaster")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")

set(CPACK_GENERATOR "AppImage")

set(CPACK_APPIMAGE_DESKTOP_FILE "pointcaster.desktop")
set(CPACK_PACKAGE_ICON "pointcaster-icon.png")

#set(CPACK_SET_DESTDIR ON)
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

include(CPack)
