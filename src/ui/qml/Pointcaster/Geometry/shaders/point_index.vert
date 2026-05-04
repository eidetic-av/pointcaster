VARYING flat vec4 vColor;

void MAIN()
{
    int xy   = floatBitsToInt(VERTEX.x);
    int zp   = floatBitsToInt(VERTEX.y);

    float px = float(int(xy << 16) >> 16);
    float py = float(xy >> 16);
    float pz = float(int(zp << 16) >> 16);

    vec3 pos = vec3(px, py, pz) * 0.1;

    // we split our point index integer into four bytes
    // in order to pack it into our color buffer.

    int i = gl_VertexIndex;
    unsigned char bytes[4];
    bytes[0] = (i >> 24) & 0xFF;
    bytes[1] = (i >> 16) & 0xFF;
    bytes[2] = (i >> 8) & 0xFF;
    bytes[3] = i & 0xFF;

    vColor = bytes;

    POSITION = MODELVIEWPROJECTION_MATRIX * vec4(pos, 1.0);
    POINT_SIZE = uPointSize;
}