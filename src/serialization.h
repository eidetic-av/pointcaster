#pragma once

#ifndef __CUDACC__
#include <serdepp/serde.hpp>
#else
#define DERIVE_SERDE(...) // No-op for CUDA
#endif
#include <tuple>

namespace pc::reflect {

// define a struct that can hold an arbitrary list of types as template args
template <typename... Types> struct type_list {};

// forward declare the type_at struct for getting type at a given index in a type list
template <typename TypeList, size_t Index> struct type_at;

// specialization of type_at for index 0, i.e., getting the first type.
template <typename Head, typename... Tail>
struct type_at<type_list<Head, Tail...>, 0> {
  using type = Head;
};

// specialization of type_at for non-zero indices
template <typename Head, typename... Tail, size_t Index>
struct type_at<type_list<Head, Tail...>, Index> {
  using type = typename type_at<type_list<Tail...>, Index - 1>::type;
};

// alias template for easier use of the type_at struct
template <typename TypeList, size_t Index>
using type_at_t = typename type_at<TypeList, Index>::type;

#ifndef __CUDACC__
// concept to check if a type is serializable... it must contain a MemberTypes
// declaration, and a MemberCount member
template <typename T>
concept IsSerializable = requires {
  typename T::MemberTypes;
  { T::MemberCount } -> std::convertible_to<std::size_t>;
};
#endif

} // namespace pc::reflect
