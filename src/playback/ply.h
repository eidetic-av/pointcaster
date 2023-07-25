#pragma once

#include <string>

namespace pc::playback {

class PlySequencePlayer {
public:
  PlySequencePlayer();

  void addSequence(const std::string &directory);

  void drawImGuiControls();
};

} // namespace pc::playback
