rm build
mkdir .\build\win_x64
echo "Configuring for Windows DLL"
cmake -B .\build\win_x64 -S . -G "Ninja Multi-Config" -DCMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
echo "Building Windows Debug"
cmake --build .\build\win_x64 --config Debug
echo "Building Windows Release"
cmake --build .\build\win_x64 --config Release
