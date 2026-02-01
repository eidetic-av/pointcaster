#include "workspace_config.h"
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
    config = rfl::yaml::read<WorkspaceFile, rfl::AddTagsToVariants>(yaml_string)
                 .value()
                 .workspace;

    pc::logger()->info("Loaded configuration from '{}'", file_path);
    pc::logger()->trace("Loaded Workspace:\n{}",
                        rfl::yaml::write<rfl::AddTagsToVariants>(
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
    const auto yaml_string = rfl::yaml::write<rfl::AddTagsToVariants>(
        WorkspaceFile{.workspace = config});
    std::ofstream(file_path) << yaml_string;
    pc::logger()->info("Saved workspace file to '{}'", file_path);
  } catch (const std::exception &e) {
    pc::logger()->error("Failed to save '{}': {}", file_path, e.what());
  }
}

} // namespace pc