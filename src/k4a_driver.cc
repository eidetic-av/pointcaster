#include "k4a_driver.h"
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>

namespace bob::sensors {

K4ADriver::K4ADriver(int _device_index) {
  device_index = _device_index;
}

bool K4ADriver::Open() {
  if (k4a_device_open(device_index, &device_) != K4A_RESULT_SUCCEEDED)
    return false;

  _config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  _config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  // _config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  _config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
  _config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED ;
  _config.camera_fps = K4A_FRAMES_PER_SECOND_30;

  k4a_device_start_cameras(device_, &_config);

  open_ = true;
  return true;
}

bool K4ADriver::Close() {
  k4a_device_close(device_);
  return true;
}

} // namespace bob::sensors
