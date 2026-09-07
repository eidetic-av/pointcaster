#pragma once

#include <config/config_registry.h>
#include <config/config_variant.h>
#include <format>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <workspace/workspace_config.h>
#include <workspace/workspace_socket.h>

namespace pc {
class Workspace;
} // namespace pc

namespace pc::publishers {

// A WorkspaceListener runs its own thread that listens for
// "published" and "pushed" workspace config values.
//
// The owner provides 'config(workspace)' (locking workspace.config_access if
// it reads workspace.config directly), 'handle_update(path, value, snapshot)',
// and optionally 'handle_config_change(path, snapshot)' and 'tick()'.
//
// Both optional callbacks run on the listener thread, so they're free to
// block. 'handle_config_change' is called once for the initial config and
// then for every change beneath this listener's prefix; 'tick()' is called
// at least every 100ms for work that isn't driven by config.
//
// The template arg ConfigT determines which configuration struct
// should be associated with this listener type.
//

template <typename OwnerT, typename ConfigT> class WorkspaceListener {
public:
  WorkspaceListener(OwnerT &owner, Workspace &workspace,
                    ConfigRegistry &config_registry)
      : _owner(owner), _workspace(workspace), _config_registry(config_registry),
        _config_subscription(_config_registry.on_change(
            config_prefix(),
            [this](std::string_view path) {
              std::scoped_lock lock(_changed_paths_access);
              _changed_paths.emplace_back(path);
            })),
        _listener_thread(listen, this) {
    // seed the initial config so owners don't need their own startup path
    std::scoped_lock lock(_changed_paths_access);
    _changed_paths.emplace_back(config_prefix());
  };

  ~WorkspaceListener() {
    _listener_thread.request_stop();
    if (_listener_thread.joinable()) _listener_thread.join();
    _config_registry.remove_subscription(_config_subscription);
  }

  WorkspaceListener(const WorkspaceListener &) = delete;
  WorkspaceListener &operator=(const WorkspaceListener &) = delete;
  WorkspaceListener(WorkspaceListener &&) = delete;
  WorkspaceListener &operator=(WorkspaceListener &&) = delete;

private:
  OwnerT &_owner;
  Workspace &_workspace;
  ConfigRegistry &_config_registry;
  ConfigRegistry::SubscriptionId _config_subscription;

  std::mutex _changed_paths_access;
  std::vector<std::string> _changed_paths;

  // must stay last
  std::jthread _listener_thread;

  // TODO publishers/ here assumes that workspace listeners are only used for
  // that purpose, and their config always lives underneath that prefix
  static std::string config_prefix() {
    return std::format("publishers/{}", ConfigT::Tag::strings().at(0));
  }

  ConfigT dispatch_config_changes() {
    static_assert(
        !requires { &OwnerT::handle_config_change; } ||
            requires(std::string_view path, const ConfigT &config) {
              _owner.handle_config_change(path, config);
            },
        "handle_config_change must take (std::string_view, const "
        "ConfigT&) to be dispatched");

    auto config_snapshot = _owner.config(_workspace);
    std::vector<std::string> changed_paths;
    {
      std::scoped_lock lock(_changed_paths_access);
      if (_changed_paths.empty()) return config_snapshot;
      changed_paths.swap(_changed_paths);
    }
    if constexpr (requires(std::string_view path, const ConfigT &config) {
                    _owner.handle_config_change(path, config);
                  }) {
      for (const auto &path : changed_paths) {
        _owner.handle_config_change(path, config_snapshot);
      }
    }
    return config_snapshot;
  }

  static void listen(std::stop_token stop_token, WorkspaceListener *self) {
    auto socket = WorkspaceSocket::create_subscriber();
    while (!stop_token.stop_requested()) {
      auto config_snapshot = self->dispatch_config_changes();
      if constexpr (requires { self->_owner.tick(config_snapshot); }) {
        self->_owner.tick(config_snapshot);
      } else if constexpr (requires { self->_owner.tick(); }) {
        self->_owner.tick();
      }
      // receive blocks until we get an update (or timeout)
      auto update = socket.receive();
      if (update == std::nullopt) continue;
      self->_owner.handle_update(update->first, update->second,
                                 config_snapshot);
      // after we've handled the update that woke the thread,
      // drain any more available updates without blocking
      for (update = socket.try_receive(); update.has_value();
           update = socket.try_receive()) {
        self->_owner.handle_update(update->first, update->second,
                                   config_snapshot);
      }
    }
  }
};

} // namespace pc::publishers
