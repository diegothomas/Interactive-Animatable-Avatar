#ifndef __LEVELSET_GPU_H
#define __LEVELSET_GPU_H

#include "include_gpu/Utilities.h"
#include "include_gpu/cudaTypes.cuh"

/**** Function definitions ****/
float*** LevelSet_gpu(float* vertices, int* faces, float* normals, int nb_vertices, int nb_faces, int3 size_grid, float3 center_grid, float res, float disp = 0.0f);

float* TetLevelSet_gpu(float* vertices, int* faces, float* normals_face, float* nodes, int nb_vertices, int nb_faces, int nb_nodes, float disp);

#endif