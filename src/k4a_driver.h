#include <k4a/k4a.h>
#include "driver.h"

namespace bob::sensors {
  class K4ADriver : Driver {
  public:
    int device_index;

    K4ADriver(int _device_index = 0);

    bool Open() override;
    bool Close() override;

    bool IsOpen() override {
      return open_;
    };

    k4a_device_t device_ = nullptr;
    k4a_device_configuration_t _config;
  private:
    bool open_ = false;
  };
} //namespace bob::sensors
