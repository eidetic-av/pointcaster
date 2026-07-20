#pragma once

#if defined(_WIN32)
  #ifdef POINTCASTER_BUILD
    #define POINTCASTER_API __declspec(dllexport)
  #else
    #define POINTCASTER_API __declspec(dllimport)
  #endif
#else
  #ifdef POINTCASTER_BUILD
    #define POINTCASTER_API __attribute__((visibility("default")))
  #else
    #define POINTCASTER_API
  #endif
#endif