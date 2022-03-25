#ifdef _WIN32
#include <WinSock2.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>
#include <spdlog/spdlog.h>

#include "../include/k4a_driver.h"

using namespace bob::sensors;

int main(int argc, char *argv[]) {
  spdlog::info("This is Box of Birds PointCaster v0.0.1");

  K4ADriver kinect;
  spdlog::debug(kinect.device_index);
  bool result = kinect.Open();
  spdlog::debug(result);
  sleep(2);
  result = kinect.Close();
  spdlog::debug(result);

  return 0;
}
