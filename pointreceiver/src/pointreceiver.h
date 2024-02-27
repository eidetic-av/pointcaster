#pragma once

#include <spdlog/spdlog.h>
#include <pointclouds.h>

#ifdef __ANDROID__
	#include <jni.h>
	#include <android/log.h>
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
void log(std::string text) {
	spdlog::info("{}", text);
}
#endif

extern "C" {
	JNIEXPORT int startNetworkThread(const char* point_caster_address = "127.0.0.1:9999", int timeout_ms = 0);
	JNIEXPORT int stopNetworkThread();
	JNIEXPORT bool dequeue();
	JNIEXPORT int pointCount();
	JNIEXPORT bob::types::position* pointPositions();
	JNIEXPORT bob::types::color* pointColors();
}
