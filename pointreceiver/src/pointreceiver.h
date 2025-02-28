#pragma once

#include <pointclouds.h>
#include <spdlog/spdlog.h>

#ifdef __ANDROID__
#include <android/log.h>
#include <jni.h>
#define APPNAME "PointReceiverNative"
#elif _WIN32
#define JNIEXPORT __declspec(dllexport)
#else
#define JNIEXPORT
#endif

#ifdef __ANDROID__
void log(std::string text) {
  auto text_cstr = text.c_str();
  int (*alias)(int, const char *, const char *, ...) = __android_log_print;
  alias(ANDROID_LOG_VERBOSE, APPNAME, text_cstr);
}
#else
void log(std::string text) { spdlog::info("{}", text); }
#endif

extern "C" {
JNIEXPORT int
startPointcloudReceiver(const char *point_caster_address = "127.0.0.1:9992",
                        int timeout_ms = 0);
JNIEXPORT int stopPointcloudReceiver();
JNIEXPORT bool dequeue(int timeout_ms = 5);
JNIEXPORT int pointCount();
JNIEXPORT bob::types::color *pointColors();
JNIEXPORT bob::types::position *pointPositions();

enum class OrientationType { unchanged = 0, flip_x, flip_x_z, flip_z };

JNIEXPORT float *shaderFriendlyPointPositions(
    bool parallel_transform = false,
    OrientationType orientation_type = OrientationType::unchanged);
}
