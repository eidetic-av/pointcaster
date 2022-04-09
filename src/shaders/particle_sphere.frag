uniform highp mat4 projectionMatrix;

uniform float particleRadius;

flat in vec3 viewCenter;
flat in vec3 color;
out lowp vec4 fragmentColor;

void main() {
    vec3 viewDir = normalize(viewCenter);
    vec3 normal;
    vec3 fragPos;

    normal.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(normal.xy, normal.xy);
    if(mag > 1.0) discard; /* outside the sphere */

    normal.z = sqrt(1.0 - mag);
    fragPos  = viewCenter + normal*particleRadius; /* correct fragment position */

    mat4 prjMatTransposed = transpose(projectionMatrix);
    float z = dot(vec4(fragPos, 1.0), prjMatTransposed[2]);
    float w = dot(vec4(fragPos, 1.0), prjMatTransposed[3]);
    gl_FragDepth = 0.5*(z/w + 1.0); /* correct fragment depth */

    fragmentColor = vec4(color, 1.0);
}
