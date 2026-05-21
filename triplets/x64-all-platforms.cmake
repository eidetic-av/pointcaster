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
        ${VCPKG_CMAKE_CONFIGURE_OPTIONS}
        -DTRACY_ON_DEMAND=ON
        -DTRACY_DELAYED_INIT=ON
        -DTRACY_MANUAL_LIFETIME=ON
    )
endif()

if(PORT MATCHES "opencv4")
    # opencv will only work with the tbb backend if it's dynamic
    set(VCPKG_LIBRARY_LINKAGE dynamic)
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS
        ${VCPKG_CMAKE_CONFIGURE_OPTIONS}
        # ensure nvcc builds using all cores
        "-DCUDA_NVCC_FLAGS=--split-compile=0"
        "-DCMAKE_CUDA_FLAGS=--split-compile=0"
        "-DCMAKE_CUDA_RELEASE_FLAGS=--threads=0"
        # target nvidia turing, ampere and ada
        "-DCUDA_ARCH_BIN=7.5;8.6;8.9"
        "-DCUDA_ARCH_PTX=8.9"
    )
endif()

if(PORT MATCHES "pcl")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS
        ${VCPKG_CMAKE_CONFIGURE_OPTIONS}
        # target nvidia turing, ampere and ada
        "-DCUDA_ARCH_BIN=75-real;86-real;89-real;89-virtual"
        "-DCMAKE_CUDA_FLAGS=--split-compile=0"
        "-DCMAKE_CUDA_RELEASE_FLAGS=--threads=0"
    )
endif()