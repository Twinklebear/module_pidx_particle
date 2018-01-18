#include <random>
#include <algorithm>
#include <vector>
#include <array>
#include <chrono>
#include <functional>

#include <turbojpeg.h>
#include <GLFW/glfw3.h>

#include "common/imgui/imgui.h"
#include "widgets/imgui_impl_glfw_gl3.h"

#include "arcball.h"
#include "util.h"
#include "image_util.h"
#include "client_server.h"

using namespace ospcommon;

std::vector<unsigned char> jpgBuf;

// Extra stuff we need in GLFW callbacks
struct WindowState {
  Arcball &camera;
  vec2f prev_mouse;
  bool camera_changed;
  AppState &app;

  WindowState(AppState &app, Arcball &camera)
    : camera(camera), prev_mouse(-1), camera_changed(true), app(app)
  {}
};

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
  if (ImGui::GetIO().WantCaptureKeyboard) {
    return;
  }

  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, true);
        break;
      case 'P':
      case 'p':
        if (!jpgBuf.empty()) {
          std::ofstream fout("screenshot.jpg", std::ios::binary);
          fout.write(reinterpret_cast<const char*>(jpgBuf.data()), jpgBuf.size());
          std::cout << "Screenshot saved to 'screenshot.jpg'\n";
        }
        break;
      default:
        break;
    }
  }
  // Forward on to ImGui
  ImGui_ImplGlfwGL3_KeyCallback(window, key, scancode, action, mods);
}

void cursorPosCallback(GLFWwindow *window, double x, double y) {
  if (ImGui::GetIO().WantCaptureMouse) {
    return;
  }

  WindowState *state = static_cast<WindowState*>(glfwGetWindowUserPointer(window));

  const vec2f mouse(x, y);
  if (state->prev_mouse != vec2f(-1)) {
    const bool leftDown =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool rightDown =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool middleDown =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    const vec2f prev = state->prev_mouse;

    if (leftDown) {
      const vec2f mouseFrom(clamp(prev.x * 2.f / state->app.fbSize.x - 1.f,  -1.f, 1.f),
                            clamp(1.f - 2.f * prev.y / state->app.fbSize.y, -1.f, 1.f));
      const vec2f mouseTo  (clamp(mouse.x * 2.f / state->app.fbSize.x - 1.f,  -1.f, 1.f),
          clamp(1.f - 2.f * mouse.y / state->app.fbSize.y, -1.f, 1.f));
      state->camera.rotate(mouseFrom, mouseTo);
      state->camera_changed = true;
    } else if (rightDown) {
      state->camera.zoom(mouse.y - prev.y);
      state->camera_changed = true;
    } else if (middleDown) {
      const vec2f mouseFrom(clamp(prev.x * 2.f / state->app.fbSize.x - 1.f,  -1.f, 1.f),
                            clamp(1.f - 2.f * prev.y / state->app.fbSize.y, -1.f, 1.f));
      const vec2f mouseTo   (clamp(mouse.x * 2.f / state->app.fbSize.x - 1.f,  -1.f, 1.f),
          clamp(1.f - 2.f * mouse.y / state->app.fbSize.y, -1.f, 1.f));
      const vec2f mouseDelta = mouseTo - mouseFrom;
      state->camera.pan(mouseDelta);
      state->camera_changed = true;
    }
  }
  state->prev_mouse = mouse;
}

void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
  WindowState *state = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
  state->app.fbSize = vec2i(width, height);
  state->app.fbSizeChanged = true;
}

void charCallback(GLFWwindow *window, unsigned int c) {
  ImGuiIO& io = ImGui::GetIO();
  if (c > 0 && c < 0x10000) {
    io.AddInputCharacter((unsigned short)c);
  }
}

int main(int argc, const char **argv)
{
  std::string serverhost;
  int port = -1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp("-server", argv[i]) == 0) {
      serverhost = argv[++i];
    } else if (std::strcmp("-port", argv[i]) == 0) {
      port = std::atoi(argv[++i]);
    }
  }
  if (serverhost.empty() || port < 0) {
    throw std::runtime_error("Usage: ./pidx_viewer -server <server host> -port <port>");
  }

  AppState app;
  AppData appdata;
  bool got_world_bounds = false;
  box3f world_bounds(vec3f(-1), vec3f(1));
  Arcball arcball_camera(world_bounds);

  if (!glfwInit()) {
    return 1;
  }
  GLFWwindow *window = glfwCreateWindow(app.fbSize.x, app.fbSize.y,
      "PIDX Particle OSPRay Viewer", nullptr, nullptr);

  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);

  auto window_state = std::make_shared<WindowState>(app, arcball_camera);

  ImGui_ImplGlfwGL3_Init(window, false);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, cursorPosCallback);
  glfwSetWindowUserPointer(window, window_state.get());
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSetMouseButtonCallback(window, ImGui_ImplGlfwGL3_MouseButtonCallback);
  glfwSetScrollCallback(window, ImGui_ImplGlfwGL3_ScrollCallback);
  glfwSetCharCallback(window, charCallback);

  JPGDecompressor decompressor;
  ServerConnection server(serverhost, port, app);

  std::vector<uint32_t> imgBuf;
  int frameTime = 0;

  while (!app.quit)  {
    imgBuf.resize(app.fbSize.x * app.fbSize.y, 0);
    if (server.get_new_frame(jpgBuf, frameTime)) {
      decompressor.decompress(jpgBuf.data(), jpgBuf.size(), app.fbSize.x,
          app.fbSize.y, imgBuf);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(app.fbSize.x, app.fbSize.y, GL_RGBA, GL_UNSIGNED_BYTE, imgBuf.data());
    
    ImGui_ImplGlfwGL3_NewFrame();

    if (!got_world_bounds) {
      ImGui::Text("Waiting for server to load");
    }
    ImGui::Text("Last frame took %dms", frameTime);

    ImGui::Render();
    
    glfwSwapBuffers(window);

    glfwPollEvents();
    if (glfwWindowShouldClose(window)) {
      app.quit = true;
    }

    if (!got_world_bounds && server.get_world_bounds(world_bounds)) {
      std::cout << "Got world bounds = " << world_bounds << std::endl;
      got_world_bounds = true;
      arcball_camera = Arcball(world_bounds);
      window_state->camera_changed = true;
    }

    const vec3f eye = window_state->camera.eyePos();
    const vec3f look = window_state->camera.lookDir();
    const vec3f up = window_state->camera.upDir();
    app.v[0] = vec3f(eye.x, eye.y, eye.z);
    app.v[1] = vec3f(look.x, look.y, look.z);
    app.v[2] = vec3f(up.x, up.y, up.z);
    app.cameraChanged = window_state->camera_changed;
    window_state->camera_changed = false;

    server.update_app_state(app, appdata);

    if (app.fbSizeChanged) {
      app.fbSizeChanged = false;
      glViewport(0, 0, app.fbSize.x, app.fbSize.y);
    }
  }

  ImGui_ImplGlfwGL3_Shutdown();
  glfwDestroyWindow(window);

  return 0;
}

