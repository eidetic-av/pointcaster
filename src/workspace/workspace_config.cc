#include "workspace_config.h"
#include <algorithm>
#include <exception>
#include <iostream>
#include <logger/logger.h>
#include <rfl/yaml.hpp>

namespace pc {

// we wrap our workspace configuration structure in a file struct
// so it gets serialized with a top-level "[workspace]" root element
struct WorkspaceFile {
  WorkspaceConfiguration workspace;
};

bool load_workspace_from_file(WorkspaceConfiguration &config,
                              const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    pc::logger()->error("Could not open '{}'", file_path);
    config = WorkspaceConfiguration{};
    return false;
  }
  const std::string yaml_string{std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>()};
  try {
    config = rfl::yaml::read<WorkspaceFile, rfl::AddTagsToVariants,
                             rfl::SnakeCaseToCamelCase>(yaml_string)
                 .value()
                 .workspace;
    pc::logger()->info("Loaded configuration from '{}'", file_path);
    pc::logger()->trace(
        "Loaded Workspace:\n{}",
        rfl::yaml::write<rfl::AddTagsToVariants, rfl::SnakeCaseToCamelCase>(
            WorkspaceFile{.workspace = config}));
  } catch (const std::exception &e) {
    pc::logger()->error("Failed to parse '{}': {}", file_path, e.what());
    config = WorkspaceConfiguration{};
    return false;
  }
  return true;
}

void save_workspace_to_file(const WorkspaceConfiguration &config,
                            const std::string &file_path) {
  try {
    const auto yaml_string =
        rfl::yaml::write<rfl::AddTagsToVariants, rfl::SnakeCaseToCamelCase>(
            WorkspaceFile{.workspace = config});
    std::ofstream(file_path) << yaml_string;
    pc::logger()->info("Saved workspace file to '{}'", file_path);
  } catch (const std::exception &e) {
    pc::logger()->error("Failed to save '{}': {}", file_path, e.what());
  }
}

std::optional<std::reference_wrapper<SessionConfiguration>>
session_config_from_workspace(WorkspaceConfiguration &config,
                              std::string_view session_id) {
  auto it = std::find_if(
      config.sessions.begin(), config.sessions.end(),
      [session_id](auto &session) { return session.id == session_id; });
  if (it == config.sessions.end()) return std::nullopt;
  return std::ref(*it);
}

} // namespace pc