uniform highp mat4 projectionMatrix;
uniform highp vec3 lightDir;

uniform vec3 ambientColor;
uniform vec3 specularColor;
uniform float shininess;

uniform float particleRadius;

flat in vec3 viewCenter;
flat in vec3 color;
out lowp vec4 fragmentColor;

vec3 shadeLight(vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 halfDir  = normalize(lightDir - viewDir);
    vec3 diffuse  = max(dot(normal, lightDir), 0.0)*color;
    vec3 specular = pow(max(dot(halfDir, normal), 0.0), shininess)*specularColor;

    return ambientColor + diffuse * 0.5 + specular; /* scale diffuse to 0.5 */
}

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

    fragmentColor = vec4(shadeLight(normal, fragPos, viewDir), 1.0);
}