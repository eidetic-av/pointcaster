#pragma once

#include "../device.h"
#include "k4a_driver.h"
#include <array>
#include <k4abt.hpp>
#include <utility>
#include <vector>
#include <atomic>

namespace pc::devices {

class K4ADevice : public Device {
public:
  K4ADevice(DeviceConfiguration config);
  ~K4ADevice();

  std::string id() override;

  void draw_device_controls() override;
  void update_device_control(int *target, int value,
			     std::function<void(int)> set_func);

  static inline std::atomic<std::size_t> count;
  static std::string get_serial_number(const std::size_t device_index);

  static inline std::size_t active_driver_count() {
    return K4ADriver::active_count;
  };
};

} // namespace pc::devices
