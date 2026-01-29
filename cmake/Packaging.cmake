include(GNUInstallDirs)

qt_generate_deploy_qml_app_script(
    TARGET pointcaster
    OUTPUT_SCRIPT pointcaster_deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
    NO_TRANSLATIONS
)
install(SCRIPT "${pointcaster_deploy_script}")

set(CPACK_PACKAGE_NAME "pointcaster")
set(CPACK_PACKAGE_VENDOR "Matt Hughes")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pointcaster")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")

if(NOT WIN32)

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

    # and also some other platform libs it seems:

    # TODO: super hacky absolute paths for deps
    install(
        FILES "/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0.0.0"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RENAME "libxcb-cursor.so.0"
    )

    # TODO this isn't copying on pop-os build???
    install(
        FILES "/usr/lib/x86_64-linux-gnu/libOpenGL.so.0.0.0"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RENAME "libOpenGL.so.0"
    )

    install(CODE "set(POINTCASTER_SOURCE_DIR \"${CMAKE_SOURCE_DIR}\")")
    install(SCRIPT "${CMAKE_SOURCE_DIR}/cmake/InstallAppImageWrapper.cmake")

    # desktop entry + icon
    install(FILES
        "${CMAKE_SOURCE_DIR}/packaging/pointcaster.desktop"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/applications"
    )
    install(FILES
        "${CMAKE_SOURCE_DIR}/packaging/pointcaster-icon.png"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/256x256/apps"
    )

    set(CPACK_GENERATOR "AppImage")

    set(CPACK_APPIMAGE_DESKTOP_FILE "pointcaster.desktop")
    set(CPACK_PACKAGE_ICON "pointcaster-icon.png")

    #set(CPACK_SET_DESTDIR ON)
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

    include(CPack)

endif()
