#pragma once

#if defined(_WIN32)
  #ifdef POINTCASTER_BUILD
    #define POINTCASTER_API __declspec(dllexport)
  #else
    #define POINTCASTER_API __declspec(dllimport)
  #endif
  // dllexport/dllimport on a whole class turns its inline-in-header members into
  // strong symbols, which collide (LNK2005) when the same header is ODR-used from
  // more than one POINTCASTER_BUILD binary. so on windows, export member-by-member
  // instead of tagging the class itself.
  #define POINTCASTER_CLASS_API
#else
  #ifdef POINTCASTER_BUILD
    #define POINTCASTER_API __attribute__((visibility("default")))
  #else
    #define POINTCASTER_API
  #endif
  // gcc/clang inline functions stay weak regardless of visibility, so tagging the
  // whole class is safe here and needed for dlopen'd plugins to see its vtable/typeinfo
  #define POINTCASTER_CLASS_API POINTCASTER_API
#endif