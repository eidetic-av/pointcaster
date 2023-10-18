echo "Configuring for Android shared lib"
mkdir build/armeabi-v7a
cmake -B build/armeabi-v7a -S . -G "Ninja Multi-Config" -DCMAKE_SYSTEM_NAME=Android -DCMAKE_ANDROID_ARCH_ABI=armeabi-v7a -DCMAKE_ANDROID_NDK="/Users/iwbrb/scoop/apps/android-sdk/current/ndk/24.0.8215888" -DCMAKE_ANDROID_API=29 -DCMAKE_TOOLCHAIN_FILE="/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=arm-android-dynamic
echo "Building Android Debug"
cmake --build build/armeabi-v7a --config Debug
echo "Building Android Release"
cmake --build build/armeabi-v7a --config Release
