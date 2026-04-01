#pragma once

#include <QString>
#include <kddockwidgets/LayoutSaver.h>
#include <ui/models/workspace_model.h>

namespace pc::ui {

class LayoutSaver : public KDDockWidgets::LayoutSaver {
public:
  explicit LayoutSaver(WorkspaceModel *workspace_model)
      : KDDockWidgets::LayoutSaver(), _workspace_model(workspace_model) {}

  bool save_file(const QString &json_filename);
  bool load_file(const QString &json_filename);

private:
  WorkspaceModel *_workspace_model;
};

} // namespace pc::ui