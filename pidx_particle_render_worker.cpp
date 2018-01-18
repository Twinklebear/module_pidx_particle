#include <random>
#include <memory>
#include <algorithm>
#include <array>
#include <chrono>
#include <mpiCommon/MPICommon.h>
#include <mpi.h>
#include <unistd.h>
#include "ospray/ospray_cpp/Camera.h"
#include "ospray/ospray_cpp/Data.h"
#include "ospray/ospray_cpp/Device.h"
#include "ospray/ospray_cpp/FrameBuffer.h"
#include "ospray/ospray_cpp/Geometry.h"
#include "ospray/ospray_cpp/Renderer.h"
#include "ospray/ospray_cpp/TransferFunction.h"
#include "ospray/ospray_cpp/Model.h"
#include "util.h"
#include "image_util.h"
#include "client_server.h"

using namespace ospcommon;
using namespace ospray::cpp;

struct Particle {
  osp::vec3f pos;
  float radius;
  int atom_type;

  Particle(float x, float y, float z, float radius, int type)
    : pos(osp::vec3f{x, y, z}), radius(radius), atom_type(type)
    {}
};

void generate_particles(std::vector<Particle> &particles, std::vector<float> &colors);

int main(int argc, char **argv) {
  int provided = 0;
  int port = -1;
  // TODO: OpenMPI sucks as always and doesn't support pt2pt one-sided
  // communication with thread multiple. This can trigger a hang in OSPRay
  // if you're not using OpenMPI you can change this to MPI_THREAD_MULTIPLE
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  std::string datasetPath;
  std::vector<std::string> timestepDirs;
  AppState app;
  AppData appdata;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp("-dataset", argv[i]) == 0) {
      datasetPath = argv[++i];
    } else if (std::strcmp("-port", argv[i]) == 0) {
      port = std::atoi(argv[++i]);
    }
  }
  if (datasetPath.empty()) {
    std::cout << "Usage: mpirun -np <N> ./pidx_render_worker [options]\n"
      << "Options:\n"
      << "-dataset <dataset.idx>\n"
      << "-port <port>\n";
    return 1;
  }

  ospLoadModule("mpi");
  Device device("mpi_distributed");
  device.set("masterRank", 0);
  device.commit();
  device.setCurrent();

  const int rank = mpicommon::world.rank;
  const int worldSize = mpicommon::world.size;

  std::unique_ptr<ClientConnection> client;
  if (rank == 0) {
    char hostname[1024] = {0};
    gethostname(hostname, 1023);
    std::cout << "Now listening for client on " << hostname << ":" << port << std::endl;
    client = ospcommon::make_unique<ClientConnection>(port);
  }

  Model model;

  // Generate some particles for now
  std::vector<Particle> particles;
  std::vector<float> atom_colors;
  generate_particles(particles, atom_colors);
  Data sphere_data(particles.size() * sizeof(Particle), OSP_CHAR,
      particles.data(), OSP_DATA_SHARED_BUFFER);
  Data color_data(atom_colors.size(), OSP_FLOAT3,
      atom_colors.data(), OSP_DATA_SHARED_BUFFER);
  sphere_data.commit();
  color_data.commit();

  Geometry spheres("spheres");
  spheres.set("spheres", sphere_data);
  spheres.set("color", color_data);
  spheres.set("bytes_per_sphere", int(sizeof(Particle)));
  spheres.set("offset_radius", int(sizeof(osp::vec3f)));
  spheres.set("offset_colorID", int(sizeof(osp::vec3f) + sizeof(float)));
  spheres.commit();

  // TODO: Make up some region grid once we've actually got some distributed stuff
  // Though technically for opaque spheres, we don't need any bricking since
  // we just do z-compositing.
  //std::vector<box3f> regions{pidxVolume->localRegion};
  //ospray::cpp::Data regionData(regions.size() * 2, OSP_FLOAT3, regions.data());
  model.addGeometry(spheres);
  model.commit();

  Camera camera("perspective");
  camera.set("pos", vec3f(0, 0, -500));
  camera.set("dir", vec3f(0, 0, 1));
  camera.set("up", vec3f(0, 1, 0));
  camera.set("aspect", static_cast<float>(app.fbSize.x) / app.fbSize.y);
  camera.commit();

  Renderer renderer("mpi_raycast");
  renderer.set("model", model);
  renderer.set("camera", camera);
  renderer.set("bgColor", vec3f(0.02));
  renderer.commit();
  assert(renderer);

  FrameBuffer fb(app.fbSize, OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM | OSP_FB_VARIANCE);
  fb.clear(OSP_FB_COLOR | OSP_FB_ACCUM | OSP_FB_VARIANCE);

  mpicommon::world.barrier();

  while (!app.quit) {
    using namespace std::chrono;

    if (app.cameraChanged) {
      camera.set("pos", app.v[0]);
      camera.set("dir", app.v[1]);
      camera.set("up", app.v[2]);
      camera.commit();

      fb.clear(OSP_FB_COLOR | OSP_FB_ACCUM | OSP_FB_VARIANCE);
      app.cameraChanged = false;
    }
    auto startFrame = high_resolution_clock::now();

    renderer.renderFrame(fb, OSP_FB_COLOR);

    auto endFrame = high_resolution_clock::now();

    if (rank == 0) {
      const int frameTime = duration_cast<milliseconds>(endFrame - startFrame).count();

      uint32_t *img = (uint32_t*)fb.map(OSP_FB_COLOR);
      client->send_frame(img, app.fbSize.x, app.fbSize.y, frameTime);
      fb.unmap(img);

      client->recieve_app_state(app, appdata);
    }

    // Send out the shared app state that the workers need to know, e.g. camera
    // position, if we should be quitting.
    MPI_Bcast(&app, sizeof(AppState), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (app.fbSizeChanged) {
      fb = FrameBuffer(app.fbSize, OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);
      fb.clear(OSP_FB_COLOR | OSP_FB_ACCUM | OSP_FB_VARIANCE);
      camera.set("aspect", static_cast<float>(app.fbSize.x) / app.fbSize.y);
      camera.commit();

      app.fbSizeChanged = false;
    }
  }

  MPI_Finalize();
  return 0;
}

void generate_particles(std::vector<Particle> &particles, std::vector<float> &colors) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<float> pos(-3.0, 3.0);
  std::uniform_real_distribution<float> radius(0.15, 0.4);
  std::uniform_int_distribution<int> type(0, 2);
  const size_t max_type = type.max() + 1;

  // Setup our particle data as a sphere geometry.
  // Each particle is an x,y,z center position + an atom type id, which
  // we'll use to apply different colors for the different atom types.
  for (size_t i = 0; i < 200; ++i) {
    particles.push_back(Particle(pos(rng), pos(rng), pos(rng),
          radius(rng), type(rng)));
  }

  std::uniform_real_distribution<float> rand_color(0.0, 1.0);
  for (size_t i = 0; i < max_type; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      colors.push_back(rand_color(rng));
    }
  }
}

