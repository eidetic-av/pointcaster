VARYING flat vec4 vColor;

void MAIN()
{
    int xy   = floatBitsToInt(VERTEX.x);
    int zp   = floatBitsToInt(VERTEX.y);
    float px = float(int(xy << 16) >> 16);
    float py = float(xy >> 16);
    float pz = float(int(zp << 16) >> 16);
    vec3 pos = vec3(px, py, pz) * 0.1;

    int i = int(UV0.x);
    vColor = vec4(
        float((i >> 24) & 0xFF) / 255.0,
        float((i >> 16) & 0xFF) / 255.0,
        float((i >>  8) & 0xFF) / 255.0,
        float(i & 0xFF) / 255.0
    );

    POSITION = MODELVIEWPROJECTION_MATRIX * vec4(pos, 1.0);

    float viewportHeight = uViewportHeight > 1.0 ? uViewportHeight : 1080.0;
    float pixelsPerUnit = abs(PROJECTION_MATRIX[1][1]) * viewportHeight * 0.5;
    float diameter = uPointSize * pixelsPerUnit / max(POSITION.w, 0.001);
    POINT_SIZE = clamp(diameter, uMinPointPixels, uMaxPointPixels);
}
