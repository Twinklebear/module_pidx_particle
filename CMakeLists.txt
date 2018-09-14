cmake_minimum_required(VERSION 3.1)
project(module_pidx_particle)

option(OSPRAY_MODULE_PIDX_PARTICLE "PIDX Particle module" OFF)
option(OSPRAY_MODULE_PIDX_PARTICLE_VIEWER "PIDX Particle remote viewer client" OFF)

if (OSPRAY_MODULE_PIDX_PARTICLE)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  find_package(PIDX REQUIRED)
  find_package(MPI REQUIRED)
  find_package(TurboJpeg REQUIRED)

  include_directories(${MPI_CXX_INCLUDE_PATH} ${PIDX_INCLUDE_DIRS})

  include_directories(
    ${CMAKE_SOURCE_DIR}/apps/common
    ${CMAKE_SOURCE_DIR}/apps/exampleViewer
    ${CMAKE_SOURCE_DIR}/apps/exampleViewer/common/imgui)

  add_library(pidx_particle_app_util
    util.cpp
    image_util.cpp
    client_server.cpp
    )
  target_link_libraries(pidx_particle_app_util PUBLIC
    ospray
    ospray_common
    ospray_mpi_common
    ${PIDX_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    TurboJpeg
    )

  ospray_create_application(pidx_particle_render_worker
    pidx_particle_render_worker.cpp
    LINK
    pidx_particle_app_util
    ospray
    ospray_common
    ospray_mpi_common
    ospray
    ${PIDX_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    TurboJpeg
    )

  if (OSPRAY_MODULE_PIDX_PARTICLE_VIEWER)
    #find_package(GLFW REQUIRED)
    include_directories(${GLFW_INCLUDE_DIRS})
    mark_as_advanced(GLFW_ROOT)

    ospray_create_application(pidx_particle_viewer
      pidx_particle_viewer.cpp
      arcball.cpp
      LINK
      pidx_particle_app_util
      ${OPENGL_LIBRARIES}
      ${GLFW_LIBRARY}
      ospray_imgui
      ospray
      ospray_common
      ospray_imgui3d_sg
      ospray_sg
      TurboJpeg
      )

  endif()  

endif()

