#pragma once

#include "../structs.h"

#include <string>
#include <variant>

namespace pc::client_sync {

using array3f = std::array<float, 3>;
using pc::types::Float3;
using AABBList = std::vector< std::array<array3f, 2> >;
using Float3List = std::vector<array3f>;

enum class MessageType : uint8_t {
  Connected = 0x00,
  ClientHeartbeat = 0x01,
  ClientHeartbeatResponse = 0x02,
  ParameterUpdate = 0x10,
  ParameterRequest = 0x11
};

using ParameterVariant =
    std::variant<float, int, Float3, array3f, AABBList, Float3List>;

struct ParameterUpdate {
  std::string id;
  ParameterVariant value;
};

using SyncMessage = std::variant<MessageType, ParameterUpdate>;

template<typename T, typename Variant>
struct is_variant_member;

template<typename T, typename... Types>
struct is_variant_member<T, std::variant<Types...>>
    : std::disjunction<std::is_same<T, Types>...> {};

template <typename T>
constexpr bool IsConvertableToParameterUpdate =
    is_variant_member<T, ParameterVariant>::value;

} // namespace pc::client_sync