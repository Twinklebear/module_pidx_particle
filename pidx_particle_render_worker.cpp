#include <random>
#include <memory>
#include <algorithm>
#include <array>
#include <chrono>
#include <mpi.h>
#include <unistd.h>
#include <mpiCommon/MPICommon.h>
#include <ospcommon/FileName.h>
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
  vec3f pos;
  float radius;
  int atom_type;

  Particle(float x, float y, float z, float radius, int type)
    : pos(x, y, z), radius(radius), atom_type(type)
    {}
  Particle(vec3f v, float radius, int type)
    : pos(v), radius(radius), atom_type(type)
    {}
};

void generate_particles(std::vector<Particle> &particles, std::vector<float> &colors,
    box3f &local_bounds);
void load_cosmic_web_test(const FileName &filename, std::vector<Particle> &particles,
    box3f &local_bounds);

int main(int argc, char **argv) {
  int provided = 0;
  int port = -1;
  // TODO: OpenMPI sucks as always and doesn't support pt2pt one-sided
  // communication with thread multiple. This can trigger a hang in OSPRay
  // if you're not using OpenMPI you can change this to MPI_THREAD_MULTIPLE
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  FileName dataset_path;
  std::vector<FileName> cosmic_web_bricks;
  bool cosmic_web = false;
  AppState app;
  AppData appdata;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp("-dataset", argv[i]) == 0) {
      dataset_path = argv[++i];
    } else if (std::strcmp("-port", argv[i]) == 0) {
      port = std::atoi(argv[++i]);
    } else if (std::strcmp("-cosmicweb", argv[i]) == 0) {
      ++i;
      for (; i < argc; ++i) {
        if (argv[i][0] == '-') {
          break;
        }
        cosmic_web_bricks.push_back(argv[i]);
      }
    }
  }
  if (dataset_path.str().empty() && cosmic_web_bricks.empty()) {
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
  const int world_size = mpicommon::world.size;

  std::unique_ptr<ClientConnection> client;
  char hostname[1024] = {0};
  gethostname(hostname, 1023);
  if (rank == 0) {
    std::cout << "Now listening for client on " << hostname << ":" << port << std::endl;
    client = ospcommon::make_unique<ClientConnection>(port);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  Model model;

  // Generate some particles for now
  std::vector<Particle> particles;
  std::vector<float> atom_colors;
  box3f local_bounds, world_bounds;
  if (!cosmic_web_bricks.empty()) {
    const size_t bricks_per_rank = cosmic_web_bricks.size() / world_size;
    for (int i = 0; i < bricks_per_rank; ++i) {
      load_cosmic_web_test(cosmic_web_bricks[rank * bricks_per_rank + i],
          particles, local_bounds);
    }
    // We just have one type of "particle" so just randomly color on each rank
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> rand_color(0.0, 1.0);
    for (size_t j = 0; j < 3; ++j) {
      atom_colors.push_back(rand_color(rng));
    }
  } else {
    generate_particles(particles, atom_colors, local_bounds);
  }
  // Extend by the particle radius
  local_bounds.lower -= vec3f(1);
  local_bounds.upper += vec3f(1);

  MPI_Reduce(&local_bounds.lower, &world_bounds.lower, 3, MPI_FLOAT, MPI_MIN,
      0, MPI_COMM_WORLD);
  MPI_Reduce(&local_bounds.upper, &world_bounds.upper, 3, MPI_FLOAT, MPI_MAX,
      0, MPI_COMM_WORLD);
  if (rank == 0) {
    client->send_metadata(world_bounds);
  }

  std::cout << "Rank " << rank << " on " << hostname
    << " has " << particles.size() << " particles\n";
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
  spheres.set("offset_radius", int(sizeof(vec3f)));
  spheres.set("offset_colorID", int(sizeof(vec3f) + sizeof(float)));
  spheres.commit();

  // TODO: Make up some region grid once we've actually got some distributed stuff
  // Though technically for opaque spheres, we don't need any bricking since
  // we just do z-compositing.
  // TODO: This bounds will be an overestimate for the cosmology data
  std::vector<box3f> regions{local_bounds};
  ospray::cpp::Data regionData(regions.size() * 2, OSP_FLOAT3, regions.data());
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

void generate_particles(std::vector<Particle> &particles, std::vector<float> &colors,
    box3f &local_bounds)
{
  std::random_device rd;
  std::mt19937 rng(rd());
  local_bounds = box3f(vec3f(-3.f), vec3f(3.f));
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

#pragma pack(1)
struct CosmicWebHeader {
  // number of particles in this dat file
  int np_local;
  float a, t, tau;
  int nts;
  float dt_f_acc, dt_pp_acc, dt_c_acc;
  int cur_checkpoint, cur_projection, cur_halofind;
  float massp;
};
std::ostream& operator<<(std::ostream &os, const CosmicWebHeader &h) {
  os << "{\n\tnp_local = " << h.np_local
    << "\n\ta = " << h.a
    << "\n\tt = " << h.t
    << "\n\ttau = " << h.tau
    << "\n\tnts = " << h.nts
    << "\n\tdt_f_acc = " << h.dt_f_acc
    << "\n\tdt_pp_acc = " << h.dt_pp_acc
    << "\n\tdt_c_acc = " << h.dt_c_acc
    << "\n\tcur_checkpoint = " << h.cur_checkpoint
    << "\n\tcur_halofind = " << h.cur_halofind
    << "\n\tmassp = " << h.massp
    << "\n}";
  return os;
}

void load_cosmic_web_test(const FileName &filename, std::vector<Particle> &particles,
    box3f &local_bounds)
{
  std::ifstream fin(filename.c_str(), std::ios::binary);

  if (!fin.good()) {
    throw std::runtime_error("could not open particle data file " + filename.str());
  }

  CosmicWebHeader header;
  if (!fin.read(reinterpret_cast<char*>(&header), sizeof(CosmicWebHeader))) {
    throw std::runtime_error("Failed to read header");
  }

  std::cout << "Cosmic Web Header: " << header << "\n";

  // Compute the brick offset for this file, given in the last 3 numbers of the name
  std::string brick_name = filename.name();
  brick_name = brick_name.substr(brick_name.size() - 3, 3);
  const int brick_number = std::stoi(brick_name);
  // The cosmic web bricking is 8^3
  const int brick_z = brick_number / 64;
  const int brick_y = (brick_number / 8) % 8;
  const int brick_x = brick_number % 8;
  std::cout << "Brick position = { " << brick_x << ", " << brick_y
    << ", " << brick_z << " }\n";
  // Each cell is 768x768x768 units
  const float step = 768.f;
  const vec3f offset(step * brick_x, step * brick_y, step * brick_z);

  particles.reserve(particles.size() + header.np_local);
  for (int i = 0; i < header.np_local; ++i) { 
    vec3f position, velocity;

    if (!fin.read(reinterpret_cast<char*>(&position), sizeof(vec3f))) {
      throw std::runtime_error("Failed to read position for particle");
    }
    if (!fin.read(reinterpret_cast<char*>(&velocity), sizeof(vec3f))) {
      throw std::runtime_error("Failed to read velocity for particle");
    }
    position += offset;

    local_bounds.lower = min(position, local_bounds.lower);
    local_bounds.upper = max(position, local_bounds.upper);
    particles.emplace_back(position, 1.0, 0);
  }
}

