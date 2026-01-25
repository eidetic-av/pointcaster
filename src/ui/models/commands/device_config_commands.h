#pragma once

#include <QString>
#include <QUndoCommand>
#include <plugins/devices/device_variants.h>

namespace pc {
struct Workspace;
}

namespace pc::ui {

class SetDeviceConfigCommand final : public QUndoCommand {
public:
  static constexpr int CommandId = 1;

  SetDeviceConfigCommand(
      pc::Workspace &workspace, int device_index, int field_index,
      pc::devices::DeviceConfigurationVariant before,
      pc::devices::DeviceConfigurationVariant after, QString command_text,
      std::function<void(int deviceIndex)> ui_refresh_callback);

  void undo() override;
  void redo() override;

  int id() const override { return CommandId; }
  bool mergeWith(const QUndoCommand *other) override;

private:
  void apply(const pc::devices::DeviceConfigurationVariant &value);

  pc::Workspace &_workspace;
  int _device_index = -1;
  int _field_index = -1;
  pc::devices::DeviceConfigurationVariant _before;
  pc::devices::DeviceConfigurationVariant _after;

  std::function<void(int deviceIndex)> _ui_refresh_callback;
};

} // namespace pc::ui
