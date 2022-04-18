vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO eyalz800/zpp_bits
    REF v4.3.3
    SHA512 6534711a75d62e4244125329db436ca27bf6b8c31aa9ecb92473f4e5687aa390d9d197d1cb91ed3452363af1537ffdef4a13fe6a1c3049616576a475ba69a548
    HEAD_REF master
    PATCHES 
    	no-xor-keyword.patch
)

file(
    COPY "${SOURCE_PATH}/zpp_bits.h"
    DESTINATION "${CURRENT_PACKAGES_DIR}/include"
)

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
