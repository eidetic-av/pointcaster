# Pointcaster

Pointcaster is an engine for capturing, transforming, analysing and
broadcasting 3D scenes captured using depth sensors. It enables
room-scale mixed reality installations, performances and applications.
Currently it supports Orbbec Femto Mega and Azure Kinect sensors.

Depth data captured by multiple sensors can be synthesized into a single
point-cloud in real-time, and streamed into client applications through
the Pointreceiver library over local networks or the internet. Scene
analysis such as clustering and presence detection can be configured and
streamed over multiple protocols including MIDI, RTP-MIDI, OSC and MQTT
to enable integration into game engines, digital audio workstations and
any other application you can think of.

![Pointcaster screenshot](https://b2.matth.cc/pointcaster-scrot.png)

## Download

Automated builds for Pointcaster executables and the Pointreceiver library are available for download at [https://b2.matth.cc/pointcaster](https://b2.matth.cc/pointcaster).

## Developing

-   `pointcaster` requires [CMake](https://cmake.org/) and
    [vcpkg](https://vcpkg.io) to build and manage dependencies. A
    working CUDA toolkit installation and `nvcc` on the path is also
    required.
-   Linux builds are using GCC 12
-   Windows builds use Visual Studio 2022, MSVC v143
-   CUDA compiler version 12.6

### Configure

Windows builds are configured using CMake presets. You may need to update the path to your vcpkg toolchain file location inside `CMakePresets.json`, then configure for your desired preset:

```
cmake --preset windows-release
```

Linux builds will be adapted to use presets soon, but for now use classic command line configuration:

``` fish
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DVCPKG_OVERLAY_PORTS=$(pwd)/ports
```

#### On Linux systems with multiple versions of GCC

At least Arch is shipping with GCC 13 as default, so the
`CMAKE_CXX_COMPILER` CMake option needs to be set:

``` fish
cmake -B build -DCMAKE_CXX_COMPILER=/usr/bin/gcc-12 ...
```

### Build

From the project root:

``` sh
cmake --build .\build\windows-release
```

## Notes

### Sensor Drivers

-   To run multiple Azure Kinect sensors on Linux, you must boot with
    the following kernel parameter that adjusts memory provided to USB
    hosts. Set it to a number appropriate for your system and camera
    count. 256MB is probably overkill:

    usbcore.usbfs_memory_mb=256
