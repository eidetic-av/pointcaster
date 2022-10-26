#include "ply.h"
#include <spdlog/spdlog.h>
#include <imgui.h>
#include <nfd.h>

bob::playback::PlySequencePlayer() {

}

bob::playback::addSequence(const std::string &directory) try {
  spdlog::info("Loading ply sequence from: {}", directory);
} catch (std::exception e) {
  spdlog::error("Exception thrown attempting to load sequence.\n{}", e.what());
}

bob::playback::drawImGuiControls() {
  if (ImGui::Button("Add Sequence")) {
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath );
    addSequence(outPath);
    free(outPath);
  }
}
