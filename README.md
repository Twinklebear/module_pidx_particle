# module\_pidx\_particle

App for viewing pidx particle data.

## Building

Clone the repo into your OSPRay modules directory, then run CMake to build
OSPRay and pass `-DOSPRAY_MODULE_PIDX_PARTICLE=ON` to build the module's movie
renderer and render workers and `-DOSPRAY_MODULE_PIDX_PARTICLE_VIEWER=ON` to build
the remote viewer client. PIDX and TurboJPEG 1.5.x+ are required, along
with MPI. You can pass `-DTURBOJPEG_DIR` to the root directory of your
TurboJPEG installation directory if it's not installed in a standard location.

## Running the Remote Viewer

To run the remote viewer first start the render workers on your compute nodes.

```bash
mpirun -np <N> ./pidx_particle_render_worker \
       -dataset <dataset.idx> \
       -port <port to listen on>
```

These workers will start and load the data. Once they're ready to connect to
with the viewer, rank 0 will print out "Rank 0 now listening for client". You
can then start the viewer and pass it the hostname of rank 0 and the port
to connect to.

```bash
./pidx_particle_viewer -server <rank 0 hostname> -port <port to connect>
```

