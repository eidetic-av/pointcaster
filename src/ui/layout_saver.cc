#include "layout_saver.h"

#include <QByteArray>
#include <QString>
#include <fstream>
#include <kddockwidgets/LayoutSaver.h>
#include <logger/logger.h>
#include <nlohmann/json.hpp>

#include <iostream>

namespace pc::ui {

using nlohmann::json;

// This is the base KDDW :: LayoutSaver::saveToFile, but after it serializes
// we can use nlohmann to add additional fields that kddw lib doesnt serialize
// itself (like whether the main window outside of the docking panel is
// maximized/fullscreen etc.)

bool LayoutSaver::save_file(const QString &json_filename) {
  if (!saveToFile(json_filename)) {
    pc::logger()->error("Failed to save layout file");
    return false;
  }
  {
    std::ifstream in(json_filename.toStdString());
    json save_data = json::parse(in);

    auto foldedPropertyPaths = json::object();
    for (const auto &[key, value] :
         _workspace_model->foldedPropertyPaths().toStdMap()) {
      foldedPropertyPaths[key.toStdString()] = value == "true";
    }
    save_data["foldedPropertyPaths"] = foldedPropertyPaths;

    std::ofstream out(json_filename.toStdString());
    out << save_data.dump(4) << '\n';
  }
  return true;
}

// Same idea with file loading... load the file, read our own fields we added
// on top of the base KDDW serialization, then perform the base deserialization
// (it will ignore our custom fields)

bool LayoutSaver::load_file(const QString &json_filename) {
  std::ifstream in(json_filename.toStdString());
  json save_data = json::parse(in);

  for (auto &[path, value] : save_data["foldedPropertyPaths"].items()) {
    _workspace_model->setFoldedProperty(path.c_str(), value);
  }

  return restoreFromFile(json_filename);
}

} // namespace pc::ui