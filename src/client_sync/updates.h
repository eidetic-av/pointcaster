#pragma once

#include "../structs.h"

#include <string>
#include <variant>

namespace pc::client_sync {

using array3f = std::array<float, 3>;
using array4f = std::array<float, 4>;
using pc::types::Float3;
using pc::types::Float4;
using AABBList = std::vector<std::array<array3f, 2>>;
using Float3List = std::vector<array3f>;
using Float4List = std::vector<array4f>;

enum class MessageType : uint8_t {
  Connected = 0x00,
  ClientHeartbeat = 0x01,
  ClientHeartbeatResponse = 0x02,
  ParameterUpdate = 0x10,
  ParameterRequest = 0x11
};

using ParameterVariant = std::variant<float, int, Float3, Float4, array3f,
                                      AABBList, Float3List, Float4List>;

struct ParameterUpdate {
  std::string id;
  ParameterVariant value;
};

struct EndpointUpdate {
  std::string id;
  size_t port;
  bool active;
};

using SyncMessage = std::variant<MessageType, ParameterUpdate, EndpointUpdate>;

template <typename T, typename Variant> struct is_variant_member;

template <typename T, typename... Types>
struct is_variant_member<T, std::variant<Types...>>
    : std::disjunction<std::is_same<T, Types>...> {};

template <typename T>
constexpr bool IsConvertableToParameterUpdate =
    is_variant_member<T, ParameterVariant>::value;

} // namespace pc::client_sync