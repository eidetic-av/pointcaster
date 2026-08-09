#pragma once

#include <thread>

namespace pc {
class Workspace;
}

namespace pc::publishers {

class WorkspacePublisher {
public:
  explicit WorkspacePublisher(Workspace &workspace);

private:
  std::jthread _worker;
};

} // namespace pc::publishers
