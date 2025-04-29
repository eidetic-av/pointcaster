namespace pc::gui {

struct PointcasterSessionLayout {
  bool fullscreen = false;                         // @optional
  bool hide_ui = false;                            // @optional
  bool show_log = true;                            // @optional
  bool show_devices_window = true;                 // @optional
  bool show_camera_window = false;                 // @optional
  bool show_radio_window = false;                  // @optional
  bool show_snapshots_window = false;              // @optional
  bool show_global_transform_window = false;       // @optional
  bool show_stats = false;                         // @optional
  bool show_session_operator_graph_window = false; // @optional
};

} // namespace pc::gui