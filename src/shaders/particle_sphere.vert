uniform highp mat4 viewMatrix;
uniform highp mat4 projectionMatrix;

uniform int numParticles;
uniform int colorMode;
uniform float particleRadius;
uniform float pointSizeScale;

uniform vec3 diffuseColor;

layout(location = 0) in highp vec3 position;

flat out vec3 viewCenter;
flat out vec3 color;

const vec3 colorRamp[] = vec3[] (
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 0.5, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 0.0, 1.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 1.0, 1.0),
    vec3(0.0, 0.0, 1.0)
);

float rand(vec2 co) {
    const float a  = 12.9898f;
    const float b  = 78.233f;
    const float c  = 43758.5453f;
    float dt = dot(co.xy, vec2(a, b));
    float sn = mod(dt, 3.14);
    return fract(sin(sn) * c);
}

vec3 generateVertexColor() {
    if(colorMode == 2) { /* consistent random color */
        return vec3(rand(vec2(gl_VertexID, gl_VertexID)),
                    rand(vec2(gl_VertexID + 1, gl_VertexID)),
                    rand(vec2(gl_VertexID, gl_VertexID + 1)));
    } else if(colorMode == 1 ) { /* ramp color by particle id */
        float segmentSize = float(numParticles)/6.0f;
        float segment = floor(float(gl_VertexID)/segmentSize);
        float t = (float(gl_VertexID) - segmentSize*segment)/segmentSize;
        vec3 startVal = colorRamp[int(segment)];
        vec3 endVal = colorRamp[int(segment) + 1];
        return mix(startVal, endVal, t);
    } else { /* uniform diffuse color */
        return diffuseColor;
    }
}

void main() {
    vec4 eyeCoord = viewMatrix*vec4(position, 1.0);
    vec3 posEye = vec3(eyeCoord);

    /* output */
    viewCenter = posEye;
    color = generateVertexColor();

    gl_PointSize = particleRadius*pointSizeScale/length(posEye);
    gl_Position = projectionMatrix*eyeCoord;
}