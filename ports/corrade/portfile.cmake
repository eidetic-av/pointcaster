vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO mosra/corrade
    REF 295bbba1f49887da060465f88b8501965f6acd7d
    SHA512 58af843018979bb7a7ea9a41f2655a81e6c64e51dbc08c64cc3bc194c38bd23e6d4dc5169773d8312579954b70cc01b05558e0f0adb69ef874408a6a0cb3b694
    HEAD_REF master
    PATCHES
    # fix-vs2019.patch
      build-corrade-rc-always.patch
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" BUILD_STATIC)

# Handle features
set(_COMPONENTS "")
foreach(_feature IN LISTS ALL_FEATURES)
    # Uppercase the feature name and replace "-" with "_"
    string(TOUPPER "${_feature}" _FEATURE)
    string(REPLACE "-" "_" _FEATURE "${_FEATURE}")

    # Final feature is empty, ignore it
    if(_feature)
        list(APPEND _COMPONENTS ${_feature} WITH_${_FEATURE})
    endif()
endforeach()

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS FEATURES ${_COMPONENTS})

set(corrade_rc_param "")
if(VCPKG_CROSSCOMPILING)
    set(corrade_rc_param
        "-DCORRADE_RC_EXECUTABLE=${CURRENT_HOST_INSTALLED_DIR}/tools/corrade/corrade-rc${VCPKG_HOST_EXECUTABLE_SUFFIX}"
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        "${corrade_rc_param}"
        -DUTILITY_USE_ANSI_COLORS=ON
        -DBUILD_STATIC=${BUILD_STATIC}
        -DCORRADE_CPU_USE_IFUNC=OFF
    MAYBE_UNUSED_VARIABLES
        CORRADE_RC_EXECUTABLE
)

vcpkg_cmake_install()

# Debug includes and share are the same as release
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# corrade-rc is not built when CMAKE_CROSSCOMPILING
if("utility" IN_LIST FEATURES)
    vcpkg_copy_tools(TOOL_NAMES "corrade-rc" AUTO_CLEAN)
endif()

# Ensure no empty folders are left behind
if(NOT FEATURES)
    # No features, no binaries (only Corrade.h).
    file(REMOVE_RECURSE
        "${CURRENT_PACKAGES_DIR}/bin"
        "${CURRENT_PACKAGES_DIR}/lib"
        "${CURRENT_PACKAGES_DIR}/debug")
    # debug is completely empty, as include and share
    # have already been removed.

elseif(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    # No dlls
    file(REMOVE_RECURSE
        "${CURRENT_PACKAGES_DIR}/bin"
        "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

# Handle copyright
file(INSTALL "${SOURCE_PATH}/COPYING"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME copyright)

vcpkg_copy_pdbs()
