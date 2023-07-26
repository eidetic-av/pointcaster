// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3
// as published by the Free Software Foundation.

// This random word generator was adapted from
// https://github.com/Wicpar/WTF-the-random-word-generator

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <span>

namespace wtf {

class generator {
public:
  static std::string generateWord(int32_t min_letters, int32_t max_letters,
				  std::string prefix = "",
				  std::string suffix = "");

private:
  static char getNextRandomAuthorizedLetter(std::string_view word);

  static bool isLong(char c);
  static bool isShort(char c);
  static bool isVocal(char c);
  static bool isRepeatable(char c);

  static char randomChar(std::span<const char> arr, char last_char);
  static char randomVocal(char last_char);
  static char randomShort(char last_char);
  static char randomLong(char last_char);

  static constexpr std::array<char, 6> vocals = {'a', 'e', 'i', 'o', 'u', 'y'};
  static constexpr std::array<char, 9> shorts = {'t', 'p', 'q', 'd', 'g',
						 'k', 'x', 'c', 'b'};
  static constexpr std::array<char, 11> longs = {'z', 'r', 's', 'f', 'h', 'j',
						 'l', 'm', 'w', 'v', 'n'};
  static constexpr std::array<char, 15> repeatables = {'a', 'z', 'e', 'r', 't',
						       'o', 'p', 's', 'd', 'f',
						       'g', 'l', 'm', 'c', 'n'};
};

} // namespace wtf
