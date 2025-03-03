#pragma once

#include <pointclouds.h>
#include <spdlog/spdlog.h>

#ifdef __ANDROID__
#include <android/log.h>
#include <jni.h>
#define APPNAME "PointReceiverNative"
#elif _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __ANDROID__
void log(std::string text) {
  auto text_cstr = text.c_str();
  int (*alias)(int, const char *, const char *, ...) = __android_log_print;
  alias(ANDROID_LOG_VERBOSE, APPNAME, text_cstr);
}
#else
template <typename... Args>
void log(fmt::format_string<Args...> fmt, Args &&...args) {
  spdlog::info(fmt, std::forward<Args>(args)...);
}
#endif

extern "C" {

typedef struct PointReceiverContext PointReceiverContext;

typedef enum {
  MSG_TYPE_CONNECTED = 0,
  MSG_TYPE_CLIENT_HEARTBEAT,
  MSG_TYPE_CLIENT_HEARTBEAT_RESPONSE,
  MSG_TYPE_PARAMETER_UPDATE,
  MSG_TYPE_PARAMETER_REQUEST,
  MSG_TYPE_UNKNOWN
} PointReceiverMessageType;

typedef enum {
  PARAM_VALUE_FLOAT = 0,
  PARAM_VALUE_INT,
  PARAM_VALUE_FLOAT3,
  PARAM_VALUE_ARRAY3F,
  PARAM_VALUE_UNKNOWN
} PointReceiverParameterValueType;

typedef struct {
  PointReceiverMessageType message_type;
  char id[256];
  PointReceiverParameterValueType value_type; 
  union {
    float float_val;
    int int_val;
    struct { float x, y, z; } float3_val;
  } value;
} PointReceiverSyncMessage;

EXPORT PointReceiverContext *createPointReceiverContext();

EXPORT int startMessageReceiver(PointReceiverContext *ctx,
                                const char *pointcaster_address);
EXPORT int stopMessageReceiver(PointReceiverContext *ctx);
EXPORT bool dequeueMessage(PointReceiverContext *ctx,
                           PointReceiverSyncMessage *out_message,
                           int timeout_ms);

// TODO convert everything below here to a C API

EXPORT int
startPointcloudReceiver(const char *pointcaster_address = "127.0.0.1",
                        int timeout_ms = 0);
EXPORT int stopPointcloudReceiver();
EXPORT bool dequeuePointcloud(int timeout_ms = 5);
EXPORT int pointCount();
EXPORT bob::types::color *pointColors();
EXPORT bob::types::position *pointPositions();

enum class OrientationType { unchanged = 0, flip_x, flip_x_z, flip_z };

EXPORT float *shaderFriendlyPointPositions(
    bool parallel_transform = false,
    OrientationType orientation_type = OrientationType::unchanged);
}
