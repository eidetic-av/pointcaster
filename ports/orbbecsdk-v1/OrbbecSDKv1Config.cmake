if(TARGET ob::OrbbecSDKv1)
    return()
endif()

get_filename_component(_orbbecsdkv1_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_library(ob::OrbbecSDKv1 SHARED IMPORTED)
set_target_properties(ob::OrbbecSDKv1 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_orbbecsdkv1_root}/include/orbbecsdk-v1"
)

# the sdk ships release binaries only, both configurations use them
if(WIN32)
    set_target_properties(ob::OrbbecSDKv1 PROPERTIES
        IMPORTED_LOCATION "${_orbbecsdkv1_root}/bin/orbbecsdk-v1/OrbbecSDK.dll"
        IMPORTED_IMPLIB "${_orbbecsdkv1_root}/lib/orbbecsdk-v1/OrbbecSDK.lib"
    )
else()
    set_target_properties(ob::OrbbecSDKv1 PROPERTIES
        IMPORTED_LOCATION "${_orbbecsdkv1_root}/lib/orbbecsdk-v1/libOrbbecSDK.so.1.10.35"
        IMPORTED_SONAME "libOrbbecSDK.so.1.10"
    )
endif()

set(OrbbecSDKv1_FOUND TRUE)

unset(_orbbecsdkv1_root)
