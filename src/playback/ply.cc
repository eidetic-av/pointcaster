#include "ply.h"
#include <exception>
#include <imgui.h>
#include <nfd.h>
#include "logger.h"

namespace pc::playback {

PlySequencePlayer::PlySequencePlayer() {}

void PlySequencePlayer::addSequence(const std::string &directory) try {
  pc::logger->info("Loading ply sequence from: {}", directory);
} catch (std::exception e) {
  pc::logger->error("Exception thrown attempting to load sequence.\n{}", e.what());
}

void PlySequencePlayer::drawImGuiControls() {
  if (ImGui::Button("Add Sequence")) {
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog(NULL, NULL, &outPath);
    addSequence(outPath);
    free(outPath);
  }
}

} // namespace pc::playback
