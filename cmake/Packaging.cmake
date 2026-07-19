include(GNUInstallDirs)

# trim windeployqt output: no software gl, no d3d12 shader compilers, no qml debug plugins, no vc_redist (crt dlls installed manually below)
qt_generate_deploy_qml_app_script(
    TARGET pointcaster
    OUTPUT_SCRIPT pointcaster_deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
    NO_TRANSLATIONS
    NO_COMPILER_RUNTIME
    DEPLOY_TOOL_OPTIONS
        --no-opengl-sw
        --no-system-dxc-compiler
        --skip-plugin-types qmltooling
)

# inject NO_OVERWRITE into the deploy script, otherwise windeployqt --force fails replacing qml plugin dlls it still has open
install(CODE "
    file(READ \"${pointcaster_deploy_script}\" _pointcaster_deploy_script_contents)
    string(REPLACE
        \"qt6_deploy_runtime_dependencies(\"
        \"qt6_deploy_runtime_dependencies(\n    NO_OVERWRITE\"
        _pointcaster_deploy_script_contents
        \"\${_pointcaster_deploy_script_contents}\"
    )
    file(WRITE \"${pointcaster_deploy_script}\" \"\${_pointcaster_deploy_script_contents}\")
")
# qt's deploy support hardcodes its bin dir to "bin", override at install time to match our flat windows layout
if(WIN32)
    install(CODE [[set(QT_DEPLOY_BIN_DIR ".")]])
endif()
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
        "${CMAKE_SOURCE_DIR}/packaging/pointcaster.png"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/256x256/apps"
    )

    set(CPACK_GENERATOR "AppImage")

    set(CPACK_APPIMAGE_DESKTOP_FILE "pointcaster.desktop")
    set(CPACK_PACKAGE_ICON "pointcaster.png")

    #set(CPACK_SET_DESTDIR ON)
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

    include(CPack)

endif()

if(WIN32)
    # on windows we have trouble deploying tbb so just do it manually
    # (TO_CMAKE_PATH stops TBB_DIR backslashes becoming escape sequences in cmake_install.cmake)
    file(TO_CMAKE_PATH "$ENV{TBB_DIR}" _pointcaster_tbb_dir)
    install(FILES "${_pointcaster_tbb_dir}/redist/intel64/vc14/tbb12.dll"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # app-local msvc crt dlls instead of shipping the vc_redist installer
    if(NOT DEFINED ENV{VCToolsRedistDir})
        message(FATAL_ERROR "VCToolsRedistDir not set - windows builds must run in a VS developer shell")
    endif()
    file(TO_CMAKE_PATH "$ENV{VCToolsRedistDir}" _pointcaster_vc_redist_dir)
    set(_pointcaster_crt_dir "${_pointcaster_vc_redist_dir}/x64/Microsoft.VC143.CRT")
    install(FILES
        "${_pointcaster_crt_dir}/msvcp140.dll"
        "${_pointcaster_crt_dir}/msvcp140_1.dll"
        "${_pointcaster_crt_dir}/msvcp140_2.dll"
        "${_pointcaster_crt_dir}/msvcp140_atomic_wait.dll"
        "${_pointcaster_crt_dir}/msvcp140_codecvt_ids.dll"
        "${_pointcaster_crt_dir}/vcruntime140.dll"
        "${_pointcaster_crt_dir}/vcruntime140_1.dll"
        "${_pointcaster_crt_dir}/vcruntime140_threads.dll"
        "${_pointcaster_crt_dir}/concrt140.dll"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # post-deploy trim of dev-only artifacts and duplicated plugin dlls
    # note: qtquick controls styles must all ship, qml imports Fusion and the platform default style differs per machine
    install(CODE [[
        set(_prefix "${CMAKE_INSTALL_PREFIX}")

        # link-time and tooling artifacts nothing reads at runtime
        file(GLOB_RECURSE _dev_artifacts
            "${_prefix}/plugins/*.lib"
            "${_prefix}/plugins/*.exp"
            "${_prefix}/qml/*.qmltypes"
        )
        if(_dev_artifacts)
            file(REMOVE ${_dev_artifacts})
        endif()

        # only touch the corrade plugin trees, qt's own plugin dirs use no .conf metadata
        set(_corrade_plugin_globs
            "${_prefix}/plugins/backend/*.dll"
            "${_prefix}/plugins/devices/*.dll"
            "${_prefix}/plugins/operators/*.dll"
        )

        # vcpkg applocal duplicates dlls beside each plugin, copies that also ship beside the exe are never loaded
        file(GLOB_RECURSE _plugin_dlls ${_corrade_plugin_globs})
        foreach(_plugin_dll ${_plugin_dlls})
            get_filename_component(_dll_name "${_plugin_dll}" NAME)
            if(EXISTS "${_prefix}/${_dll_name}")
                file(REMOVE "${_plugin_dll}")
            endif()
        endforeach()
    ]])
endif()
