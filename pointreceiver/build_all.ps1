mkdir build
cd build
echo "Configuring for Windows DLL"
cmake -B . -S .. -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_OVERLAY_PORTS="/bob/bob-pointreceiver/native/ports"
echo "Building Windows Debug"
cmake --build . --config Debug
echo "Building Windows Release"
cmake --build . --config Release
cd ..

# echo "Configuring for Android shared lib"
# mkdir build_android
# cd build_android
# cmake -B . -S .. -G "Ninja Multi-Config" -DCMAKE_SYSTEM_NAME=Android -DCMAKE_ANDROID_ARCH_ABI=armeabi-v7a -DCMAKE_ANDROID_NDK="/Users/iwbrb/scoop/apps/android-sdk/current/ndk/24.0.8215888" -DCMAKE_ANDROID_API=29 -DCMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=arm-android-dynamic
# echo "Building Android Debug"
# cmake --build . --config Debug
# echo "Building Android Release"
# cmake --build . --config Release
# cd ..
