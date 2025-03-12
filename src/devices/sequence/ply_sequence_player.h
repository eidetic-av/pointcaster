#pragma once

#include "../device.h"
#include "ply_sequence_player_configuration.gen.h"

namespace pc::devices {

class PlySequencePlayer {
public:
  explicit PlySequencePlayer(PlySequencePlayerConfiguration &config);
};

} // namespace pc::devices