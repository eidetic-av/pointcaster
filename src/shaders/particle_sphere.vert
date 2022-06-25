uniform highp mat4 viewMatrix;
uniform highp mat4 projectionMatrix;

uniform float particleRadius;
uniform float pointSizeScale;

layout(location = 0) in vec3 position;
layout(location = 2) in float in_color;

flat out vec3 viewCenter;
flat out vec3 color;

vec3 decodeColor(float raw) {
  int bgra = floatBitsToInt(raw);
  float a = float(bgra >> 24) / 255.0;
  float r = float((bgra & 0x00ff0000) >> 16) / 255.0;
  float g = float((bgra & 0x0000ff00) >> 8) / 255.0;
  float b = float(bgra & 0x000000ff) / 255.0;
  return vec3(r, g, b);
}

void main() {
  // convert short millimetres back to float metres
  vec3 pos_m = position / 1000.0;

  vec4 eye_coord = viewMatrix*vec4(pos_m.x, pos_m.y, pos_m.z, 1.0);
  vec3 pos_eye = vec3(eye_coord);

  /* output */
  viewCenter = pos_eye;
       
  color = decodeColor(in_color);

  gl_PointSize = particleRadius*pointSizeScale/length(pos_eye);
  gl_Position = projectionMatrix*eye_coord;
}
