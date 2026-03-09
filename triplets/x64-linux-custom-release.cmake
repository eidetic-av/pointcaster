set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

# orbbec camera sdk expects to be dynamically linked
# (it has a whole runtime 'extensions' thing with its own bundled libs)
if(PORT MATCHES "orbbecsdk")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

# spdlog and fmt requires dynamic linkage because of the orbbec sdk
# being loaded as a dynamic plugin, and that expects these as .so
if(PORT MATCHES "spdlog")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if (PORT MATCHES "fmt")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if (PORT MATCHES "tracy")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS 
        -DTRACY_ON_DEMAND=ON
        -DTRACY_DELAYED_INIT=ON
        -DTRACY_MANUAL_LIFETIME=ON
    )
endif()