#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <thread>
#include <variant>
#include <vector>

#include <Corrade/Utility/StlMath.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Version.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/ImGuiIntegration/Widgets.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <serdepp/adaptor/toml11.hpp>
#include <serdepp/serde.hpp>
#include <toml.hpp>

#include <tracy/Tracy.hpp>
#include <zmq.hpp>
#include <zpp_bits.h>