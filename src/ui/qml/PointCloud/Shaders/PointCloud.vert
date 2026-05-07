VARYING flat vec4 vColor;

void MAIN()
{
    int xy   = floatBitsToInt(VERTEX.x);
    int zp   = floatBitsToInt(VERTEX.y);
    uint rgba = floatBitsToUint(VERTEX.z);

    float px = float(int(xy << 16) >> 16);
    float py = float(xy >> 16);
    float pz = float(int(zp << 16) >> 16);

    vec3 pos = vec3(px, py, pz) * 0.1;

    vec3 srgb = vec3(
        float(rgba & 0xFFu),
        float((rgba >> 8u) & 0xFFu),
        float((rgba >> 16u) & 0xFFu)
    ) / 255.0;
    vColor = vec4(srgb, 1.0);

    POSITION = MODELVIEWPROJECTION_MATRIX * vec4(pos, 1.0);
    POINT_SIZE = uPointSize;
}