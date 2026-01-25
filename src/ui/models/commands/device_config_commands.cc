#include "device_config_commands.h"
#include "../../../workspace.h"
#include <mutex>

namespace pc::ui {

SetDeviceConfigCommand::SetDeviceConfigCommand(
    pc::Workspace &workspace, int device_index, int field_index,
    pc::devices::DeviceConfigurationVariant before,
    pc::devices::DeviceConfigurationVariant after, QString command_text,
    std::function<void(int deviceIndex)> ui_refresh_callback)
    : QUndoCommand(std::move(command_text)), _workspace(workspace),
      _device_index(device_index), _field_index(field_index),
      _before(std::move(before)), _after(std::move(after)),
      _ui_refresh_callback(std::move(ui_refresh_callback)) {}

void SetDeviceConfigCommand::undo() {
  apply(_before);
}
void SetDeviceConfigCommand::redo() {
  apply(_after);
}

bool SetDeviceConfigCommand::mergeWith(const QUndoCommand *other) {
  if (!other || other->id() != CommandId) return false;

  auto *o = static_cast<const SetDeviceConfigCommand *>(other);
  if (&o->_workspace != &_workspace) return false;
  if (o->_device_index != _device_index) return false;
  if (o->_field_index != _field_index) return false;

  _after = o->_after;
  setText(o->text());

  return true;
}

void SetDeviceConfigCommand::apply(
    const pc::devices::DeviceConfigurationVariant &value) {
  std::scoped_lock lock(_workspace.config_access);

  if (_device_index < 0) return;

  const std::size_t device_index_u = static_cast<std::size_t>(_device_index);
  if (device_index_u >= _workspace.config.devices.size()) return;

  _workspace.config.devices[device_index_u] = value;

  if (device_index_u < _workspace.devices.size()) {
    if (auto *plugin = _workspace.devices[device_index_u].get()) {
      plugin->update_config(value);
      plugin->on_config_field_changed(_device_index, _field_index);
    }
  }

  if (_ui_refresh_callback) _ui_refresh_callback(_device_index);
}

} // namespace pc::ui