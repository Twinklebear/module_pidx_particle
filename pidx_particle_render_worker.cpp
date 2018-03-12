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
  int type;

  Particle(float x, float y, float z, float radius, int type)
    : pos(x, y, z), radius(radius), type(type)
    {}
  Particle(vec3f v, float radius, int type)
    : pos(v), radius(radius), type(type)
    {}
};

void load_pidx_particles(const FileName &filename, std::vector<Particle> &particles,
    box3f &local_bounds);

int main(int argc, char **argv) {
  int provided = 0;
  int port = -1;
  // TODO: OpenMPI sucks as always and doesn't support pt2pt one-sided
  // communication with thread multiple. This can trigger a hang in OSPRay
  // if you're not using OpenMPI you can change this to MPI_THREAD_MULTIPLE
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  FileName dataset_path;
  AppState app;
  AppData appdata;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp("-dataset", argv[i]) == 0) {
      dataset_path = argv[++i];
    } else if (std::strcmp("-port", argv[i]) == 0) {
      port = std::atoi(argv[++i]);
    } else if (std::strcmp("-f", argv[i]) == 0) {
      dataset_path = argv[++i];
    }
  }
  if (dataset_path.str().empty()) {
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

  Model model;

  // Generate some particles for now
  std::vector<Particle> particles;
  std::vector<float> atom_colors;
  box3f local_bounds, world_bounds;
  load_pidx_particles(dataset_path, particles, local_bounds);

  const auto type_range = std::minmax_element(particles.begin(), particles.end(),
      [](const Particle &a, const Particle &b) {
        return a.type < b.type;
      });
  std::cout << "type range = [" << type_range.first->type << ", "
    << type_range.second->type << "]\n";

  // We just have one type of "particle" for now, so just randomly color on each rank
  // TODO WILL: Once we start querying other attribs from PIDX this will change
  // to color by the attribute in some way.
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<float> rand_color(0.0, 1.0);
  for (size_t j = 0; j < 3 * (type_range.second->type + 1); ++j) {
    atom_colors.push_back(rand_color(rng));
  }

  // Extend by the particle radius
  local_bounds.lower -= vec3f(1);
  local_bounds.upper += vec3f(1);

  MPI_Reduce(&local_bounds.lower, &world_bounds.lower, 3, MPI_FLOAT, MPI_MIN,
      0, MPI_COMM_WORLD);
  MPI_Reduce(&local_bounds.upper, &world_bounds.upper, 3, MPI_FLOAT, MPI_MAX,
      0, MPI_COMM_WORLD);


  char hostname[1024] = {0};
  gethostname(hostname, 1023);
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

  std::unique_ptr<ClientConnection> client;
  if (rank == 0) {
    std::cout << "Now listening for client on " << hostname << ":" << port << std::endl;
    client = ospcommon::make_unique<ClientConnection>(port);
    client->send_metadata(world_bounds);
  }
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
void load_pidx_particles(const FileName &filename, std::vector<Particle> &particles,
    box3f &local_bounds)
{
  PIDX_access access;
  PIDX_physical_point pdims;
  PIDX_file pidx_file;
  PIDX_CHECK(PIDX_create_access(&access));
  PIDX_CHECK(PIDX_set_mpi_access(access, MPI_COMM_WORLD));
  PIDX_CHECK(PIDX_file_open(filename.c_str(), PIDX_MODE_RDONLY,
        access, NULL, pdims, &pidx_file));

  std::cout << "PIDX physical dims = " << pdims[0] << ", " << pdims[1] << ", " << pdims[2] << "\n";

  PIDX_CHECK(PIDX_set_current_time_step(pidx_file, 0));

  const int rank = mpicommon::world.rank;
  const int world_size = mpicommon::world.size;

  const vec3i grid = computeGrid(world_size);
  // TODO WILL: This is information we need from the file.
  const vec3f physical_dims = vec3f(pdims[0], pdims[1], pdims[2]);
  const vec3i brick_id(rank % grid.x, (rank / grid.x) % grid.y, rank / (grid.x * grid.y));

  const vec3f local_dims = physical_dims / vec3f(grid);
  const vec3f local_offset = local_dims * vec3f(brick_id);
  std::cout << "Rank " << rank << " has local offset "
    << local_offset << " and dims " << local_dims << "\n";

  PIDX_physical_point pidx_physical_offset, pidx_physical_size;
  PIDX_set_physical_point(pidx_physical_offset, local_offset[0], local_offset[1], local_offset[2]);
  PIDX_set_physical_point(pidx_physical_size, local_dims[0], local_dims[1], local_dims[2]);

  int variable_count = 0;
  PIDX_CHECK(PIDX_get_variable_count(pidx_file, &variable_count));
  PIDX_CHECK(PIDX_set_current_variable_index(pidx_file, 0));

  std::vector<std::string> var_names;
  std::vector<PIDX_variable> vars(variable_count, PIDX_variable{});
  std::vector<unsigned char*> particle_data(variable_count, nullptr);
  std::vector<size_t> num_var_particles(variable_count, 0);
  for (int i = 0; i < variable_count; ++i) {
    PIDX_CHECK(PIDX_get_next_variable(pidx_file, &vars[i]));
    PIDX_CHECK(PIDX_variable_read_particle_data_layout(vars[i], pidx_physical_offset,
          pidx_physical_size, (void**)&particle_data[i], &num_var_particles[i], PIDX_row_major));
    if (i + 1 < variable_count) {
      PIDX_CHECK(PIDX_read_next_variable(pidx_file, vars[i]));
    }
    var_names.push_back(vars[i]->var_name);
  }

  PIDX_CHECK(PIDX_close(pidx_file));
  for (int i = 0; i < variable_count; ++i) {
    std::cout << "read " << num_var_particles[i] << " particles for variable #"
      << i << ": '" << var_names[i] << "'\n";
  }

  for (auto &num : num_var_particles) {
    if (num != num_var_particles[0]) {
      std::cout << "Differing number of particles for different vars: "
        << num << " vs. " << num_var_particles[0] << std::endl;
      throw std::runtime_error("Differing number of particles for different vars!");
    }
  }

  local_bounds.lower = local_offset;
  local_bounds.upper = local_offset + local_dims;

  const double *position_data = reinterpret_cast<double*>(particle_data[0]);
  const int *type_data = reinterpret_cast<int*>(particle_data[4]);
  for (size_t i = 0; i < num_var_particles[0]; ++i) {
    particles.emplace_back(position_data[i * 3],
        position_data[i * 3 + 1], position_data[i * 3 + 2], 0.05, type_data[i]);
    std::cout << particles.back().pos << ", type = "
      << particles.back().type << "\n";
    // Sanity check on box query
    if (!local_bounds.contains(particles.back().pos)) {
      std::cout << "Read uncontained particle " << i
       << " at " << particles.back().pos << "!\n";
    }
  }

  std::cout << "Rank " << rank << " has "
    << num_var_particles[0] << " particles, in region " << local_bounds << "\n";

  for (auto data : particle_data) {
    std::free(data);
  }
}

