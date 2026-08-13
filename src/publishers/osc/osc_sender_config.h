#pragma once
#include <rfl/DefaultVal.hpp>
#include <string>

namespace pc::publishers {

struct OscSenderConfiguration {
  rfl::DefaultVal<bool> enabled = false;
  rfl::DefaultVal<std::string> host = "127.0.0.1";
  rfl::DefaultVal<int> port = 9000; // @minmax(1024, 49151)

  // positions can either be sent as raw millimeter shorts, or encoded by this
  // sender as metre floats
  enum class PositionEncoding { Millimetres, Metres };
  rfl::DefaultVal<PositionEncoding> position_encoding =
      PositionEncoding::Millimetres;

  // a point cloud fans out into one message per point at '<path>/<index>',
  // so this caps how many of those go out for any single cloud...
  // (which shouldn't really be many when we're talking about OSC here)
  rfl::DefaultVal<int> max_cloud_points = 16; // @minmax(1, 256)
};

} // namespace pc::publishers
