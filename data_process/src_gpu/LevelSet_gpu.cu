#include "include_gpu/LevelSet_gpu.cuh"

//////////////////////////////////////////
///**** Device Function definitions ****/
/////////////////////////////////////////

__device__ __forceinline__ float norm(float3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ __forceinline__ float IsInterectingRayTriangle3D_gpu(float3 ray, float3 p0, float3 p1, float3 p2, float3 p3, float3 n) {
    float den = dot(ray, n);
    if (fabs(den) == 1.0e-6f)
        return 0.0f;

    float fact = (dot(p1 - p0, n) / den);
    if (fact < 1.0e-6f)
        return 0.0f;

    float3 proj = p0 + ray * fact;
    // Compute if proj is inside the triangle
    // V = p1 + s(p2-p1) + t(p3-p1)
    // find s and t
    float3 u = p2 - p1;
    float3 v = p3 - p1;
    float3 w = proj - p1;

    float s = (dot(u, v) * dot(w, v) - dot(v, v) * dot(w, u)) / (dot(u, v) * dot(u, v) - dot(u, u) * dot(v, v));
    float t = (dot(u, v) * dot(w, u) - dot(u, u) * dot(w, v)) / (dot(u, v) * dot(u, v) - dot(u, u) * dot(v, v));

    //if (t == 0.0f || s == 0.0f || s+t == 1.0f)
    //    return 0.5f;

    if (s >= 0.0f && t >= 0.0f && s + t <= 1.0f)
        return 1.0f;

    return 0.0f;
}

__device__ __forceinline__ float DistancePointFace3D_gpu(float3 p0, float3 p1, float3 p2, float3 p3, float3 n, bool approx = false) {
    float3 center = (p1 + p2 + p3) * (1.0f / 3.0f);
    if (approx) {
        float d0 = sqrt(dot(p0 - center, p0 - center));
        float d1 = sqrt(dot(p0 - p1, p0 - p1));
        float d2 = sqrt(dot(p0 - p2, p0 - p2));
        float d3 = sqrt(dot(p0 - p3, p0 - p3));
        return min(d0, min(d1, min(d2, d3)));
    }

    // a. Project point onto the plane of the triangle
    float3 p1p0 = p0 - p1;
    float dot_prod = dot(p1p0, n);
    float3 proj = p0 - n * dot_prod;

    //p1p2p3
    float3 cross_p1p2p3 = cross(p2 - p1, p3 - p1);
    float area = norm(cross_p1p2p3) / 2.0f;
    if (area < 1.0e-12) {
        return 1.0e32;
    }

    // b. Test if projection is inside the triangle
    float3 C;

    // edge 0 = p1p2
    float3 edge0 = p2 - p1;
    float3 vp0 = proj - p1;
    C = cross(edge0, vp0);
    float w = (norm(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge0
        // compute distance point to segment
        float curr_dist;
        float3 base = edge0 * (1.0f / norm(edge0));
        float Dt = dot(base, vp0);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p1);
        }
        else if (Dt > norm(edge0)) {
            curr_dist = norm(p0 - p2);
        }
        else {
            curr_dist = norm(p0 - (p1 + base * Dt));
        }
        return curr_dist;
    }

    // edge 1 = p2p3
    float3 edge1 = p3 - p2;
    float3 vp1 = proj - p2;
    C = cross(edge1, vp1);
    float u = (norm(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge1
        // compute distance point to segment
        // compute distance point to segment
        float curr_dist;
        float3 base = edge1 * (1.0f / norm(edge1));
        float Dt = dot(base, vp1);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p2);
        }
        else if (Dt > norm(edge1)) {
            curr_dist = norm(p0 - p3);
        }
        else {
            curr_dist = norm(p0 - (p2 + base * Dt));
        }
        return curr_dist;
    }

    // edge 2 = p3p1
    float3 edge2 = p1 - p3;
    float3 vp2 = proj - p3;
    C = cross(edge2, vp2);
    float v = (norm(C) / 2.0f) / area;
    if (dot(n, C) < 0.0f) {
        // P is on the right side of edge 2;
        float curr_dist;
        float3 base = edge2 * (1.0f / norm(edge2));
        float Dt = dot(base, vp2);
        if (Dt < 0.0f) {
            curr_dist = norm(p0 - p3);
        }
        else if (Dt > norm(edge2)) {
            curr_dist = norm(p0 - p1);
        }
        else {
            curr_dist = norm(p0 - (p3 + base * Dt));
        }
        return curr_dist;
    }

    if (u <= 1.00001f && v <= 1.00001f && w <= 1.00001f) {
        return sqrt(dot(p0 - proj, p0 - proj));
    }
    else {
        return 1.0e32;
    }

    return 1.0e32;
}

__device__ __forceinline__ void LevelSetProcess(float* volume, float* vertices, int* faces, float* normals, int nb_faces, int3 size_grid, float3 center_grid, float res, float disp)
{
	unsigned int i = threadIdx.x + blockIdx.x * THREAD_SIZE_X; // cols
	unsigned int j = threadIdx.y + blockIdx.y * THREAD_SIZE_Y; // rows
	unsigned int k = threadIdx.z + blockIdx.z * THREAD_SIZE_Z; // rows
	unsigned int idx = i * size_grid.y * size_grid.z + j * size_grid.z + k;

	if (i > size_grid.x - 1 || j > size_grid.y - 1 || k > size_grid.z - 1)
		return;

    // Get the 3D coordinate
	float3 p0;
	p0.x = (float(i) - float(size_grid.x) / 2.0f) * res + center_grid.x;
    p0.y = (float(j) - float(size_grid.y) / 2.0f) * res + center_grid.y;
    p0.z = (float(k) - float(size_grid.z) / 2.0f) * res + center_grid.z;

	float3 ray = make_float3(1.0f, 1.0f, 1.0f);
    ray = ray / sqrt(3.0f);
    float3 ray1 = make_float3(1.0f, 0.0f, 0.0f);
    float3 ray2 = make_float3(0.0f, 1.0f, 0.0f);

	// Compute the smallest distance to the faces
	float min_dist = 1.0e32f;
	float sdf = 1.0f;
	float intersections = 0.0f;
    float intersections1 = 0.0f;
    float intersections2 = 0.0f;
	for (int f = 0; f < nb_faces; f++) {
		// Compute distance point to face
		float3 n = make_float3(normals[3 * f], normals[3 * f + 1], normals[3 * f + 2]);
		float3 p1 = make_float3(vertices[3 * faces[3 * f]], vertices[3 * faces[3 * f] + 1], vertices[3 * faces[3 * f] + 2]);
		float3 p2 = make_float3(vertices[3 * faces[3 * f + 1]], vertices[3 * faces[3 * f + 1] + 1], vertices[3 * faces[3 * f + 1] + 2]);
		float3 p3 = make_float3(vertices[3 * faces[3 * f + 2]], vertices[3 * faces[3 * f + 2] + 1], vertices[3 * faces[3 * f + 2] + 2]);

		// Compute line plane intersection
		intersections += IsInterectingRayTriangle3D_gpu(ray, p0, p1, p2, p3, n);
        intersections1 += IsInterectingRayTriangle3D_gpu(ray1, p0, p1, p2, p3, n);
        intersections2 += IsInterectingRayTriangle3D_gpu(ray2, p0, p1, p2, p3, n);

		// Compute point to face distance
		float curr_dist = DistancePointFace3D_gpu(p0, p1, p2, p3, n);
		if (curr_dist < min_dist) {
			min_dist = curr_dist;
            sdf = curr_dist;
		}
	}

    int count_int = int(intersections) % 2 == 0 ? 1 : 0;
    count_int += int(intersections1) % 2 == 0 ? 1 : 0;
    count_int += int(intersections2) % 2 == 0 ? 1 : 0;

    volume[idx] = count_int > 1 ? min(1.0f, (sdf - disp) / TRUNCATE) : max(-1.0f, (-sdf - disp) / TRUNCATE);
}

__global__ void LevelSetKernel(float* volume, float* vertices, int* faces, float* normals, int nb_faces, int3 size_grid, float3 center_grid, float res, float disp)
{
	LevelSetProcess(volume, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
}


__device__ __forceinline__ void TetLevelSetProcess(float* volume, float* vertices, int* faces, float* normals, float *nodes, int nb_faces, int nb_nodes, float disp, int thread_size)
{
    unsigned int i = threadIdx.x + blockIdx.x * THREAD_SIZE_X; // cols
    unsigned int j = threadIdx.y + blockIdx.y * THREAD_SIZE_Y; // rows
    unsigned int idx = i * thread_size + j;

    if (idx > nb_nodes-1)
        return;

    float3 ray = make_float3(0.0f, 0.0f, 1.0f);
    float3 ray1 = make_float3(1.0f, 0.0f, 0.0f);
    float3 ray2 = make_float3(0.0f, 1.0f, 0.0f);

    float3 p0 = make_float3(nodes[3 * idx], nodes[3 * idx + 1], nodes[3 * idx + 2]);

    // Compute the smallest distance to the faces
    float min_dist = 1.0e32f;
    float sdf = 1.0f;
    float intersections = 0.0f;
    float intersections1 = 0.0f;
    float intersections2 = 0.0f;
    for (int f = 0; f < nb_faces; f++) {
        // Compute distance point to face
        float3 n = make_float3(normals[3 * f], normals[3 * f + 1], normals[3 * f + 2]);
        float3 p1 = make_float3(vertices[3 * faces[3 * f]], vertices[3 * faces[3 * f] + 1], vertices[3 * faces[3 * f] + 2]);
        float3 p2 = make_float3(vertices[3 * faces[3 * f + 1]], vertices[3 * faces[3 * f + 1] + 1], vertices[3 * faces[3 * f + 1] + 2]);
        float3 p3 = make_float3(vertices[3 * faces[3 * f + 2]], vertices[3 * faces[3 * f + 2] + 1], vertices[3 * faces[3 * f + 2] + 2]);

        // Compute line plane intersection
        intersections+= IsInterectingRayTriangle3D_gpu(ray, p0, p1, p2, p3, n);
        intersections1 += IsInterectingRayTriangle3D_gpu(ray1, p0, p1, p2, p3, n);
        intersections2 += IsInterectingRayTriangle3D_gpu(ray2, p0, p1, p2, p3, n);

        float3 center = (p1 + p2 + p3) * (1.0f / 3.0f);

        // Compute point to face distance
        float curr_dist = DistancePointFace3D_gpu(p0, p1, p2, p3, n, false);
        if (curr_dist < min_dist) {
            min_dist = curr_dist;
            sdf = curr_dist/TRUNCATE_Fine;
            //sdf = min(1.0f, curr_dist / TRUNCATE_Fine);
        }
    }

    int count_int = int(intersections) % 2 == 0 ? 1 : 0;
    count_int += int(intersections1) % 2 == 0 ? 1 : 0;
    count_int += int(intersections2) % 2 == 0 ? 1 : 0;

    volume[idx] = count_int > 1 ? sdf - disp / TRUNCATE_Fine : -sdf - disp / TRUNCATE_Fine;
}

__global__ void TetLevelSetKernel(float* volume, float* vertices, int* faces, float* normals, float *nodes, int nb_faces, int nb_nodes, float disp, int thread_size)
{
    TetLevelSetProcess(volume, vertices, faces, normals, nodes, nb_faces, nb_nodes, disp, thread_size);
}

//////////////////////////////////////////
///******* Function definitions *********/
//////////////////////////////////////////

float*** LevelSet_gpu(float* vertices, int* faces, float* normals, int nb_vertices, int nb_faces, int3 size_grid, float3 center_grid, float res, float disp) {
	// Allocate data
	float*** volume = new float** [size_grid.x];
	for (int i = 0; i < size_grid.x; i++) {
		volume[i] = new float* [size_grid.y];
		for (int j = 0; j < size_grid.y; j++) {
			volume[i][j] = new float[size_grid.z];
			for (int k = 0; k < size_grid.z; k++) {
				volume[i][j][k] = 1.0f;
			}
		}
	}

	float* volume_gpu;
	checkCudaErrors(cudaMalloc((void**)&volume_gpu, size_grid.x * size_grid.y * size_grid.z * sizeof(float)));

	for (int i = 0; i < size_grid.x; i++) {
		for (int j = 0; j < size_grid.y; j++) {
			checkCudaErrors(cudaMemcpy((void*)&volume_gpu[i * size_grid.y * size_grid.z + j * size_grid.z], (void*)volume[i][j], size_grid.z * sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	float* vertices_gpu;
	checkCudaErrors(cudaMalloc((void**)&vertices_gpu, 3 * nb_vertices * sizeof(float)));
	checkCudaErrors(cudaMemcpy((void*)vertices_gpu, (void*)vertices, 3 * nb_vertices * sizeof(float), cudaMemcpyHostToDevice));

	int* faces_gpu;
	checkCudaErrors(cudaMalloc((void**)&faces_gpu, 3 * nb_faces * sizeof(int)));
	checkCudaErrors(cudaMemcpy((void*)faces_gpu, (void*)faces, 3 * nb_faces * sizeof(int), cudaMemcpyHostToDevice));

	float* normals_gpu;
	checkCudaErrors(cudaMalloc((void**)&normals_gpu, 3 * nb_faces * sizeof(float)));
	checkCudaErrors(cudaMemcpy((void*)normals_gpu, (void*)normals, 3 * nb_faces * sizeof(float), cudaMemcpyHostToDevice));

	dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y, THREAD_SIZE_Z);
	dim3 dimGrid(1, 1, 1);
	dimGrid.x = divUp(size_grid.x, dimBlock.x); // #cols
	dimGrid.y = divUp(size_grid.y, dimBlock.y); // # rows
	dimGrid.z = divUp(size_grid.z, dimBlock.z); // # rows

    std::cout << "Start level set on GPU" << std::endl;

	LevelSetKernel << <dimGrid, dimBlock >> > (volume_gpu, vertices_gpu, faces_gpu, normals_gpu, nb_faces, size_grid, center_grid, res, disp);

	checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "End level set on GPU" << std::endl;

	for (int i = 0; i < size_grid.x; i++) {
		for (int j = 0; j < size_grid.y; j++) {
			checkCudaErrors(cudaMemcpy((void*)volume[i][j], (void*)&volume_gpu[i * size_grid.y * size_grid.z + j * size_grid.z], size_grid.z * sizeof(float), cudaMemcpyDeviceToHost));
            /*for (int k = 0; k < size_grid.z; k++) {
                std::cout << i << ", " << j << ", " << k << "-> " << volume[i][j][k] << std::endl;
            }*/
		}
	}

	checkCudaErrors(cudaFree(volume_gpu));
	checkCudaErrors(cudaFree(vertices_gpu));
	checkCudaErrors(cudaFree(faces_gpu));
	checkCudaErrors(cudaFree(normals_gpu));

	return volume;
}

float* TetLevelSet_gpu(float* vertices, int* faces, float* normals_face, float* nodes, int nb_vertices, int nb_faces, int nb_nodes, float disp) {
    // Allocate data
    float* volume = new float[nb_nodes];
    for (int i = 0; i < nb_nodes; i++) {
        volume[i] = 1.0f;
    }

    float* volume_gpu;
    checkCudaErrors(cudaMalloc((void**)&volume_gpu, nb_nodes * sizeof(float)));
    checkCudaErrors(cudaMemcpy((void*)volume_gpu, (void*)volume, nb_nodes * sizeof(float), cudaMemcpyHostToDevice));


    float* vertices_gpu;
    checkCudaErrors(cudaMalloc((void**)&vertices_gpu, 3 * nb_vertices * sizeof(float)));
    checkCudaErrors(cudaMemcpy((void*)vertices_gpu, (void*)vertices, 3 * nb_vertices * sizeof(float), cudaMemcpyHostToDevice));

    int* faces_gpu;
    checkCudaErrors(cudaMalloc((void**)&faces_gpu, 3 * nb_faces * sizeof(int)));
    checkCudaErrors(cudaMemcpy((void*)faces_gpu, (void*)faces, 3 * nb_faces * sizeof(int), cudaMemcpyHostToDevice));

    float* normals_gpu;
    checkCudaErrors(cudaMalloc((void**)&normals_gpu, 3 * nb_faces * sizeof(float)));
    checkCudaErrors(cudaMemcpy((void*)normals_gpu, (void*)normals_face, 3 * nb_faces * sizeof(float), cudaMemcpyHostToDevice));

    float* nodes_gpu;
    checkCudaErrors(cudaMalloc((void**)&nodes_gpu, 3 * nb_nodes * sizeof(float)));
    checkCudaErrors(cudaMemcpy((void*)nodes_gpu, (void*)nodes, 3 * nb_nodes * sizeof(float), cudaMemcpyHostToDevice));

    int thread_size = int(round(sqrt(nb_nodes)) + 1);

    dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y);
    dim3 dimGrid(1, 1, 1);
    dimGrid.x = divUp(thread_size, dimBlock.x); // #cols
    dimGrid.y = divUp(thread_size, dimBlock.y); // # rows

    std::cout << "Start Tet level set on GPU" << std::endl;

    TetLevelSetKernel << <dimGrid, dimBlock >> > (volume_gpu, vertices_gpu, faces_gpu, normals_gpu, nodes_gpu, nb_faces, nb_nodes, disp, thread_size);

    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "End Tet level set on GPU" << std::endl;

    checkCudaErrors(cudaMemcpy((void*)volume, (void*)volume_gpu, nb_nodes * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(volume_gpu));
    checkCudaErrors(cudaFree(vertices_gpu));
    checkCudaErrors(cudaFree(faces_gpu));
    checkCudaErrors(cudaFree(normals_gpu));
    checkCudaErrors(cudaFree(nodes_gpu));

    return volume;
}

