// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3
// as published by the Free Software Foundation.

// This random word generator was adapted from
// https://github.com/Wicpar/WTF-the-random-word-generator

#include "wtf.h"
#include <algorithm>
#include <random>

namespace wtf {

static std::random_device rd;
static std::mt19937 gen(rd());

std::string generator::generateWord(int32_t min_letters, int32_t max_letters,
                                    std::string prefix,
                                    std::string suffix) {
  // Bounds checking
  if (min_letters < 0) min_letters = 0;
  if (max_letters < min_letters) max_letters = min_letters;
  
  int32_t l = min_letters;
  if (min_letters != max_letters) {
    std::uniform_int_distribution<> dis(min_letters, max_letters);
    l = dis(gen);
  }
  
  while (prefix.length() < l) {
    prefix += getNextRandomAuthorizedLetter(prefix);
  }
  if (suffix.length() <= prefix.length()) {
    prefix.erase(prefix.length() - suffix.length(), suffix.length());
  }
  prefix += suffix;
  return prefix;
}

char generator::getNextRandomAuthorizedLetter(std::string_view word) {
  if (word.empty()) {
    std::uniform_int_distribution<> dis(0, 2);
    switch (dis(gen)) {
      case 0:
        return randomVocal(' ');
      case 1:
        return randomShort(' ');
      case 2:
        return randomLong(' ');
    }
  } else {
    char lastchar = word[word.size() - 1];
    if (lastchar == 'q') {
      return randomVocal(lastchar);
    } else if (word.size() < 2) {
      if (isShort(lastchar)) {
        std::uniform_int_distribution<> dis(0, 1);
        switch (dis(gen)) {
          case 0:
            return randomVocal(lastchar);
          case 1:
            return randomLong(lastchar);
        }
      } else {
        std::uniform_int_distribution<> dis(0, 2);
        switch (dis(gen)) {
          case 0:
            return randomVocal(lastchar);
          case 1:
            return randomShort(lastchar);
          case 2:
            return randomLong(lastchar);
        }
      }
    } else {
      char prelastchar = word[word.size() - 2];
      if (isVocal(prelastchar)) {
        if (isShort(lastchar)) {
          std::uniform_int_distribution<> dis(0, 1);
          switch (dis(gen)) {
            case 0:
              return randomVocal(lastchar);
            case 1:
              return randomLong(lastchar);
          }
        } else if (isLong(lastchar)) {
          std::uniform_int_distribution<> dis(0, 2);
          switch (dis(gen)) {
            case 0:
              return randomVocal(lastchar);
            case 1:
              return randomShort(lastchar);
            case 2:
              return randomLong(lastchar);
          }
        } else {
          std::uniform_int_distribution<> dis(0, 1);
          switch (dis(gen)) {
            case 0:
              return randomShort(lastchar);
            case 1:
              return randomLong(lastchar);
          }
        }
      } else {
        if (isShort(lastchar) || isLong(lastchar)) {
          return randomVocal(lastchar);
        } else {
          std::uniform_int_distribution<> dis(0, 2);
          switch (dis(gen)) {
            case 0:
              return randomVocal(lastchar);
            case 1:
              return randomShort(lastchar);
            case 2:
              return randomLong(lastchar);
          }
        }
      }
    }
  }
  return '?';
}

bool generator::isLong(char c) {
  return std::find(longs.begin(), longs.end(), c) != longs.end();
}

bool generator::isShort(char c) {
  return std::find(shorts.begin(), shorts.end(), c) != shorts.end();
}

bool generator::isVocal(char c) {
  return std::find(vocals.begin(), vocals.end(), c) != vocals.end();
}

bool generator::isRepeatable(char c) {
  return std::find(repeatables.begin(), repeatables.end(), c) !=
	 repeatables.end();
}

char generator::randomChar(std::span<const char> arr, char lastchar) {
  std::uniform_int_distribution<> dis(0, arr.size() - 1);
  char c = arr[dis(gen)];
  if (lastchar == ' ' || !isVocal(lastchar) || (c != lastchar) ||
      isRepeatable(c)) {
    return c;
  }
  return randomChar(arr, lastchar);
}

char generator::randomVocal(char lastchar) {
  return randomChar(vocals, lastchar);
}

char generator::randomShort(char lastchar) {
  return randomChar(shorts, lastchar);
}

char generator::randomLong(char lastchar) {
  return randomChar(longs, lastchar);
}

} // namespace wtf
