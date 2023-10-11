#pragma once

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

namespace pc::shaders {

class TextureDisplayShader : public Magnum::GL::AbstractShaderProgram {
public:
  typedef Magnum::GL::Attribute<0, Magnum::Vector2> Position;
  typedef Magnum::GL::Attribute<1, Magnum::Vector2> TextureCoordinates;

  explicit TextureDisplayShader();

  TextureDisplayShader &bind_texture(Magnum::GL::Texture2D &texture) {
    texture.bind(TextureUnit);
    return *this;
  }

private:
  enum : Magnum::Int { TextureUnit = 0 };
};

} // namespace pc::shaders
