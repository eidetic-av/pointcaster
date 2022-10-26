#pragma once

namespace bob::playback {

  class PlySequencePlayer {
  public: 
    PlySequencePlayer();

    void addSequence(const std::string& directory);

    void drawImGuiControls();

  }

}
