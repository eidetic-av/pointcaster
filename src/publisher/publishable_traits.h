#pragma once

#include <array>
#include <list>
#include <vector>

template <typename> struct is_publishable_container : std::false_type {};

template <typename T, typename Alloc>
struct is_publishable_container<std::vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc>
struct is_publishable_container<std::list<T, Alloc>> : std::true_type {};

template <typename T, std::size_t N>
struct is_publishable_container<std::array<T, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_publishable_container_v =
    is_publishable_container<T>::value;

