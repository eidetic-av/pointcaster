mkdir build
cd build
echo "Configuring for Windows Static Lib"
cmake -B . -S .. -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows-static -DVCPKG_OVERLAY_PORTS="/bob/bob-pointreceiver/native/ports"
echo "Building Windows Debug"
cmake --build . --config Debug
echo "Building Windows Release"
cmake --build . --config Release
cd ..
