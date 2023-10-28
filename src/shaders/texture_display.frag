in vec2 fragmentTextureCoordinates;
out vec4 fragmentColor;
uniform sampler2D tex;
void main() {
    fragmentColor = texture(tex, fragmentTextureCoordinates);
}
