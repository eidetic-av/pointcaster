#pragma once
#include <string>

namespace pc::publisher {

std::string
construct_topic_string(const std::string_view topic_name,
                       std::initializer_list<std::string_view> topic_nodes);

}
