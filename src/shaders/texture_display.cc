#include "texture_display.h"

#include <Corrade/Utility/Resource.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Version.h>

namespace pc::shaders {

using namespace Magnum;

TextureDisplayShader::TextureDisplayShader() {
  Utility::Resource rs("data");

  GL::Shader vertShader{GL::Version::GL330, GL::Shader::Type::Vertex};
  GL::Shader fragShader{GL::Version::GL330, GL::Shader::Type::Fragment};
  vertShader.addSource(rs.getString("texture_display.vert")).submitCompile();
  fragShader.addSource(rs.getString("texture_display.frag")).submitCompile();

  attachShaders({vertShader, fragShader});
  CORRADE_INTERNAL_ASSERT(link());
}

} // namespace pc::shaders
