#pragma once

#include "uuid.h"
#include <string>

struct Identifiable {
  std::string _id;
  Identifiable() : _id{pc::uuid::word()} {};
  std::string_view id() const { return _id; };
};
