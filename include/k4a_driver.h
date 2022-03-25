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

  private:
    k4a_device_t device_ = nullptr;
    bool open_ = false;
  };
} //namespace bob::sensors
