vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO mosra/corrade
    REF f37591439479a5169134f8f2eba551b95054057a
    SHA512 268c569c6290e4bed7a5fff38de0bd30b013dab090e08f2d9df75c192e0403eaa5f99cc8288a0eaab9995bce58181d24d86cebf3800de2281b55dfb3f80cb336
    HEAD_REF master
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" BUILD_STATIC)

# Handle features
set(_COMPONENTS "")
foreach(_feature IN LISTS ALL_FEATURES)
    # Uppercase the feature name and replace "-" with "_"
    string(TOUPPER "${_feature}" _FEATURE)
    string(REPLACE "-" "_" _FEATURE "${_FEATURE}")

    # Final feature is empty, ignore it
    if(_feature AND NOT "${_feature}" STREQUAL "dynamic-pluginmanager")
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
    MAYBE_UNUSED_VARIABLES
        CORRADE_RC_EXECUTABLE
        UTILITY_USE_ANSI_COLORS
)

vcpkg_cmake_install()

# Debug includes and share are the same as release
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# corrade-rc is not built when CMAKE_CROSSCOMPILING
vcpkg_copy_tools(TOOL_NAMES "corrade-rc" AUTO_CLEAN)

# Ensure no empty folders are left behind
if(FEATURES STREQUAL "core")
    # No features, no libs (only Corrade.h).
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
