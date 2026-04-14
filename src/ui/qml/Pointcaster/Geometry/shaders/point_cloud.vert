VARYING flat vec4 vColor;

void MAIN()
{
    // read ALL packed data before overwriting VERTEX
    int xy   = floatBitsToInt(VERTEX.x);   // bytes 0-3: [x:i16, y:i16]
    int zp   = floatBitsToInt(VERTEX.y);   // bytes 4-7: [z:i16, pad:i16]
    uint rgba = floatBitsToUint(VERTEX.z); // bytes 8-11: [r:u8, g:u8, b:u8, a:u8]

    // sign-extend each int16 from the packed int32 (little-endian)
    float px = float(int(xy << 16) >> 16); // lower 16 bits = x
    float py = float(xy >> 16);            // upper 16 bits = y
    float pz = float(int(zp << 16) >> 16); // lower 16 bits = z

    VERTEX = vec3(px, py, pz) * 0.1; // mm to cm

    // extract uint8 colour components, sRGB gamma decode to linear
    vec3 srgb = vec3(
        float(rgba & 0xFFu),
        float((rgba >> 8u) & 0xFFu),
        float((rgba >> 16u) & 0xFFu)
    ) / 255.0;
    vColor = vec4(pow(srgb, vec3(2.2)), 1.0);

    POINT_SIZE = 1.0;
}