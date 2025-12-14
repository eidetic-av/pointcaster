target_precompile_headers(pointcaster PRIVATE 
    <algorithm>
    <chrono>
    <cstring>
    <filesystem>
    <fstream>
    <functional>
    <iostream>
    <limits>
    <mutex>
    <queue>
    <random>
    <ranges>
    <set>
    <span>
    <thread>
    <variant>
    <vector>

    <Corrade/Utility/StlMath.h>
    <Magnum/GL/Context.h>
    <Magnum/GL/DefaultFramebuffer.h>
    <Magnum/GL/PixelFormat.h>
    <Magnum/GL/Renderer.h>
    <Magnum/GL/Version.h>
    <Magnum/ImGuiIntegration/Context.hpp>
    <Magnum/ImGuiIntegration/Widgets.h>
    <Magnum/Image.h>
    <Magnum/ImageView.h>
    <Magnum/Magnum.h>
    <Magnum/Math/Color.h>
    <Magnum/Math/FunctionsBatch.h>
    <Magnum/PixelFormat.h>
    <Magnum/Platform/Sdl2Application.h>
    <Magnum/Primitives/Icosphere.h>
    <Magnum/SceneGraph/Camera.h>
    <Magnum/SceneGraph/Drawable.h>
    <Magnum/SceneGraph/MatrixTransformation3D.h>
    <Magnum/SceneGraph/Scene.h>

    <imgui.h>
    <imgui_internal.h>

    <serdepp/adaptor/toml11.hpp>
    <serdepp/serde.hpp>
    <toml.hpp>

    <tracy/Tracy.hpp>
    <zmq.hpp>
    <zpp_bits.h>
  )
