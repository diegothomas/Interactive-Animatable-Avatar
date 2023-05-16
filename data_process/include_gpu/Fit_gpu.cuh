#ifndef __FIT_GPU_H
#define __FIT_GPU_H

#include "include_gpu/Utilities.h"
#include "include_gpu/cudaTypes.cuh"

void OuterLoop_gpu(float* Vertices_out_gpu, float* sdf_gpu, float* Nodes_gpu, int* Tets_gpu, int* HashTable_gpu, float* Vertices_gpu, float* Normals_gpu, int Nb_Vertices, int Nb_Nodes, int Nb_Tets, int OuterIter, float delta, float alpha_curr);

int* GenerateHashTable_gpu(float* Nodes, int* Tets, int nb_tets, int3 size_grid, float3 center_grid, float res);

void InnerLoopSurface_gpu(float* Vertices_out_gpu, float* Vertices_gpu, float* Normals_gpu, int* RBF_id_gpu, float* RBF_gpu, int Nb_Vertices);

#endif