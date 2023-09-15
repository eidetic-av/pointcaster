#pragma once

#include "../device.h"
#include "k4w2_driver.h"

namespace pc::devices {

class K4W2Device : public Device {
public:
  K4W2Device();
  ~K4W2Device();

  std::string getBroadcastId() override;
};
} // namespace pc::devices
