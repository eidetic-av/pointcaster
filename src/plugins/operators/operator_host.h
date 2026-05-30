#pragma once

namespace pc::operators {

class OperatorHost {
public:
  virtual ~OperatorHost() = default;

  // Ask the host to re-run its operator pipeline on the current input.
  virtual void reprocess() = 0;
};

} // namespace pc::operators