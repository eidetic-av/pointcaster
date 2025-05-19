#pragma once

#include "../pointcaster.h"
#include <atomic>

namespace pc::gui {

void draw_devices_window(PointCaster &app);

void show_app_menu();
void draw_app_menu(PointCaster &app);

} // namespace pc::gui