include(GNUInstallDirs)

# windeployqt-only flags; linux uses qt's generic deploy tool which ignores
# DEPLOY_TOOL_OPTIONS (qmltooling is removed post-install there instead)
if(WIN32)
    set(_pointcaster_deploy_tool_options
        DEPLOY_TOOL_OPTIONS
            --no-opengl-sw
            --no-system-dxc-compiler
            --skip-plugin-types qmltooling
    )
endif()

qt_generate_deploy_qml_app_script(
    TARGET pointcaster
    OUTPUT_SCRIPT pointcaster_deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
    NO_TRANSLATIONS
    NO_COMPILER_RUNTIME
    ${_pointcaster_deploy_tool_options}
)

# inject NO_OVERWRITE into the deploy script,
# otherwise windeployqt --force fails replacing qml plugin dlls it still has open
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
# qt's deploy support hardcodes its bin dir to "bin",
# override at install time to match our flat windows layout
if(WIN32)
    install(CODE [[set(QT_DEPLOY_BIN_DIR ".")]])
endif()
install(SCRIPT "${pointcaster_deploy_script}")

set(CPACK_PACKAGE_NAME "pointcaster")
set(CPACK_PACKAGE_VENDOR "Matt Hughes")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pointcaster")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")

# name packages after the configure preset, e.g. pointcaster-0.2.1-windows-release
get_filename_component(_configure_preset "${CMAKE_BINARY_DIR}" NAME)
set(CPACK_PACKAGE_FILE_NAME "pointcaster-${PROJECT_VERSION}-${_configure_preset}")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/dist")

# captured here because include(CPack) overwrites CPACK_PACKAGE_FILE_NAME
# with the source-package name while generating CPackSourceConfig.cmake
set(_pointcaster_package_basename "${CPACK_PACKAGE_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}")

if(NOT WIN32) # Linux

    # non-Qt runtime deps: bundle everything except glibc, the gcc runtime and the
    # driver/host-integration libs appimages must take from the host system
    # https://github.com/AppImageCommunity/pkg2appimage/blob/master/excludelist
    install(CODE [[
        file(GET_RUNTIME_DEPENDENCIES
            EXECUTABLES $<TARGET_FILE:pointcaster>
            RESOLVED_DEPENDENCIES_VAR resolved_deps
            POST_EXCLUDE_REGEXES
            "/ld-linux-x86-64\\.so\\."
            "/libc\\.so\\."
            "/libm\\.so\\."
            "/libmvec\\.so\\."
            "/libdl\\.so\\."
            "/libpthread\\.so\\."
            "/librt\\.so\\."
            "/libresolv\\.so\\."
            "/libgcc_s\\.so\\."
            "/libstdc\\+\\+\\.so\\."
            "/libGL\\.so\\."
            "/libEGL\\.so\\."
            "/libGLX\\.so\\."
            "/libGLdispatch\\.so\\."
            "/libOpenGL\\.so\\."
            "/libX11\\.so\\."
            "/libxcb\\.so\\."
            "/libfontconfig\\.so\\."
            "/libfreetype\\.so\\."
            "/libexpat\\.so\\."
            "/libz\\.so\\."
            "/libcom_err\\.so\\."
            "/libgpg-error\\.so\\."
        )

        # follow symlink chains so the real libs land in the package instead of dangling links
        foreach(dep ${resolved_deps})
            file(COPY "${dep}" DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" FOLLOW_SYMLINK_CHAIN)
        endforeach()
    ]])

    # our gcc runtime is newer than most host distros provide, so bundle it aside:
    # pointcaster.sh only prefers it over the host copy when actually newer (checkrt pattern)
    foreach(_gcc_runtime_lib libstdc++.so.6 libgcc_s.so.1)
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=${_gcc_runtime_lib}
            OUTPUT_VARIABLE _gcc_runtime_lib_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
        file(REAL_PATH "${_gcc_runtime_lib_path}" _gcc_runtime_lib_path)
        if(NOT EXISTS "${_gcc_runtime_lib_path}")
            message(FATAL_ERROR "${_gcc_runtime_lib} not found via ${CMAKE_CXX_COMPILER}")
        endif()
        install(FILES "${_gcc_runtime_lib_path}"
            DESTINATION "optional/gcc"
            RENAME "${_gcc_runtime_lib}"
        )
    endforeach()

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

    # remove dev-only plugins and unused qml styles
    install(CODE [[
        set(_prefix "${CMAKE_INSTALL_PREFIX}")

        # qml debugger/profiler plugins are dev-only
        file(REMOVE_RECURSE "${_prefix}/plugins/qmltooling")

        # remove unused qt quick styles
        foreach(_style FluentWinUI3 Imagine Material Universal)
            file(REMOVE_RECURSE "${_prefix}/qml/QtQuick/Controls/${_style}")
            file(GLOB _style_libs "${_prefix}/lib/libQt6QuickControls2${_style}*.so*")
            if(_style_libs)
                file(REMOVE ${_style_libs})
            endif()
        endforeach()
    ]])

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
            # the orbbec plugin manages its own directory layout for loading
            # its own dependencies, so leave everything under it in place
            if(_plugin_dll MATCHES "/orbbec/")
                continue()
            endif()
            get_filename_component(_dll_name "${_plugin_dll}" NAME)
            if(EXISTS "${_prefix}/${_dll_name}")
                file(REMOVE "${_plugin_dll}")
            endif()
        endforeach()

        # tuck dependency dlls without .conf metadata into deps/ so corrade's plugin scan doesn't warn about them (they stay on the dll search path)
        file(GLOB_RECURSE _plugin_dlls ${_corrade_plugin_globs})
        foreach(_plugin_dll ${_plugin_dlls})
            # /orbbec/ is left intact
            if(_plugin_dll MATCHES "/deps/" OR _plugin_dll MATCHES "/orbbec/")
                continue()
            endif()
            get_filename_component(_dll_dir "${_plugin_dll}" DIRECTORY)
            get_filename_component(_dll_base "${_plugin_dll}" NAME_WE)
            get_filename_component(_dll_name "${_plugin_dll}" NAME)
            if(NOT EXISTS "${_dll_dir}/${_dll_base}.conf")
                file(MAKE_DIRECTORY "${_dll_dir}/deps")
                file(RENAME "${_plugin_dll}" "${_dll_dir}/deps/${_dll_name}")
            endif()
        endforeach()
    ]])

    # portable zip with pointcaster.exe at the archive root (extractors already create a folder named after the zip)
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)

    include(CPack)
endif()

# `deploy` builds the dist/ package then uploads it and a sha256 checksum to
# b2://<bucket>/<branch>/pointcaster/<package-file-name>
# other modules (pointreceiver, touchdesigner, ...) can later add their own
# deploy targets reusing scripts/deploy.py with a different module name
set(_pointcaster_deploy_bucket "pointcaster-builds")
if(WIN32)
    set(_pointcaster_package_file "${_pointcaster_package_basename}.zip")
else()
    set(_pointcaster_package_file "${_pointcaster_package_basename}.AppImage")
endif()

add_custom_target(deploy
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/deploy.py"
        "${_pointcaster_deploy_bucket}" pointcaster "${_pointcaster_package_file}"
    USES_TERMINAL
    VERBATIM
)
add_dependencies(deploy package)
