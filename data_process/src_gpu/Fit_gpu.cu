#include "include_gpu/Fit_gpu.cuh"

//////////////////////////////////////////
///**** Device Function definitions ****/
/////////////////////////////////////////

#define HASH_SIZE 100

__device__ __forceinline__ float ScTP(float3 a, float3 b, float3 c)
{
    // computes scalar triple product
    return dot(a, cross(b, c));
}

__device__ __forceinline__ float* bary_tet_gpu(float3 a, float3 b, float3 c, float3 d, float3 p)
{
    float3 vap = p - a;
    float3 vbp = p - b;

    float3 vab = b - a;
    float3 vac = c - a;
    float3 vad = d - a;

    float3 vbc = c - b;
    float3 vbd = d - b;
    // ScTP computes the scalar triple product
    float va6 = ScTP(vbp, vbd, vbc);
    float vb6 = ScTP(vap, vac, vad);
    float vc6 = ScTP(vap, vad, vab);
    float vd6 = ScTP(vap, vab, vac);
    float v6 = 1.0f / ScTP(vab, vac, vad);

    float* res = new float[4];
    res[0] = va6 * v6;
    res[1] = vb6 * v6;
    res[2] = vc6 * v6;
    res[3] = vd6 * v6;
    return res;
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

__device__ __forceinline__ bool IsInside2_gpu(float3 v, float* Nodes, int* Tets, int t) {
    // Get four summits:
    float3 s1 = make_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]);
    float3 s2 = make_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]);
    float3 s3 = make_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]);
    float3 s4 = make_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]);

    float3 ray = make_float3(0.0f, 0.0f, 1.0f);

    // Compute normal of each face
    float3 n1 = cross(s2 - s1, s3 - s1);
    n1 = n1 * (1.0f / length(n1));
    float3 n2 = cross(s2 - s1, s4 - s1);
    n2 = n2 * (1.0f / length(n2));
    float3 n3 = cross(s3 - s2, s4 - s2);
    n3 = n3 * (1.0f / length(n3));
    float3 n4 = cross(s3 - s1, s4 - s1);
    n4 = n4 * (1.0f / length(n4));

    // Compute line plane intersection
    float intersections = 0.0f;
    intersections += IsInterectingRayTriangle3D_gpu(ray, v, s1, s2, s3, n1);
    intersections += IsInterectingRayTriangle3D_gpu(ray, v, s1, s2, s4, n2);
    intersections += IsInterectingRayTriangle3D_gpu(ray, v, s2, s3, s4, n3);
    intersections += IsInterectingRayTriangle3D_gpu(ray, v, s1, s3, s4, n4);

    return int(intersections) % 2 != 0;
}

__device__ __forceinline__ bool IsInside_gpu(float3 v, float* Nodes, int* Tets, int t) {
    // Get four summits:
    float3 s1 = make_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]);
    float3 s2 = make_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]);
    float3 s3 = make_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]);
    float3 s4 = make_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]);

    // Compute normal of each face
    float3 n1 = cross(s2 - s1, s3 - s1);
    n1 = n1 * (1.0f / length(n1));
    float3 n2 = cross(s2 - s1, s4 - s1);
    n2 = n2 * (1.0f / length(n2));
    float3 n3 = cross(s3 - s2, s4 - s2);
    n3 = n3 * (1.0f / length(n3));
    float3 n4 = cross(s3 - s1, s4 - s1);
    n4 = n4 * (1.0f / length(n4));

    // Orient all faces outward
    if (dot(n1, s4 - s1) > 0.0f)
        n1 = n1 * (-1.0f);
    if (dot(n2, s1 - s2) > 0.0f)
        n2 = n2 * (-1.0f);
    if (dot(n3, s1 - s3) > 0.0f)
        n3 = n3 * (-1.0f);
    if (dot(n4, s2 - s1) > 0.0f)
        n4 = n4 * (-1.0f);

    // Test orientation of vertex
    float o1 = dot(n1, v - s1);
    float o2 = dot(n2, v - s2);
    float o3 = dot(n3, v - s3);
    float o4 = dot(n4, v - s1);

    return o1 < 0.0f && o2 < 0.0f && o3 < 0.0f && o4 < 0.0f;
}

__device__ __forceinline__ int GetTet_gpu(float3 v, float* Nodes, int* Tets, int* HashTable, int3 dim_grid, float3 center_grid, float res) {
    //compute the key for the vertex
    int3 v_id = make_int3(int((v.x - center_grid.x) / res) + dim_grid.x / 2,
        int((v.y - center_grid.y) / res) + dim_grid.y / 2,
        int((v.z - center_grid.z) / res) + dim_grid.z / 2);

    if (v_id.x < 0 || v_id.x >= dim_grid.x - 1 || v_id.y < 0 || v_id.y >= dim_grid.y - 1 || v_id.z < 0 || v_id.z >= dim_grid.z - 1) {
        return -1;
    }

    int offset = HashTable[v_id.x * dim_grid.y * dim_grid.z + v_id.y * dim_grid.z + v_id.z];
    int size_list = HashTable[v_id.x * dim_grid.y * dim_grid.z + v_id.y * dim_grid.z + v_id.z+1] - offset;

    int best_t = -1;
    for (int it = 0; it < size_list; it++) {
        int t = HashTable[dim_grid.x * dim_grid.y * dim_grid.z + offset + it];
        if (IsInside2_gpu(v, Nodes, Tets, t)) {
            best_t = t;
            break;
        }
    }

    return best_t;

}

__device__ __forceinline__ float InterpolateField_gpu(float3 v, float* sdf, float* Nodes, int* Tets, int* HashTable, int3 dim_grid, float3 center, float res) {

    int t = GetTet_gpu(v, Nodes, Tets, HashTable, dim_grid, center, res);
    if (t == -1) {
        // point ouside the field
        return NAN;
    }

    //Get barycentric coordinates
    float* B_coo = bary_tet_gpu(make_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]),
        make_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]),
        make_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]),
        make_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]), v);

    assert(B_coo[0] <= 1.0f && B_coo[1] <= 1.0f && B_coo[2] <= 1.0f && B_coo[3] <= 1.0f);

    float out_res = (B_coo[0] * sdf[Tets[4 * t]] + B_coo[1] * sdf[Tets[4 * t + 1]] + B_coo[2] * sdf[Tets[4 * t + 2]] + B_coo[3] * sdf[Tets[4 * t + 3]]) /
        (B_coo[0] + B_coo[1] + B_coo[2] + B_coo[3]);
    delete[] B_coo;

    return out_res;
}

__device__ __forceinline__ float3 Gradient_gpu(float3 v, float* sdf, float* Nodes, int* Tets, int* HashTable, int3 dim_grid, float3 center, float res) {

    float3 grad = make_float3(0.0f, 0.0f, 0.0f);

    float3 v_x_p = v + make_float3(0.001f, 0.0f, 0.0f);
    float sdf_x_p = InterpolateField_gpu(v_x_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
    float3 v_x_m = v + make_float3(-0.001f, 0.0f, 0.0f);
    float sdf_x_m = InterpolateField_gpu(v_x_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
    if (sdf_x_p != sdf_x_p) {
        if (sdf_x_m != sdf_x_m) {
            grad.x = 0.0f;
        }
        else {
            grad.x = 1.0f;
        }
    }
    else {
        if (sdf_x_m != sdf_x_m) {
            grad.x = -1.0f;
        }
        else {
            grad.x = (sdf_x_p - sdf_x_m);
        }
    }

    float3 v_y_p = v + make_float3(0.0f, 0.001f, 0.0f);
    float sdf_y_p = InterpolateField_gpu(v_y_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
    float3 v_y_m = v + make_float3(0.0f, -0.001f, 0.0f);
    float sdf_y_m = InterpolateField_gpu(v_y_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
    if (sdf_y_p != sdf_y_p) {
        if (sdf_y_m != sdf_y_m) {
            grad.y = 0.0f;
        }
        else {
            grad.y = 1.0f;
        }
    }
    else {
        if (sdf_y_m != sdf_y_m) {
            grad.y = -1.0f;
        }
        else {
            grad.y = (sdf_y_p - sdf_y_m);
        }
    }

    float3 v_z_p = v + make_float3(0.0f, 0.0f, 0.001f);
    float sdf_z_p = InterpolateField_gpu(v_z_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
    float3 v_z_m = v + make_float3(0.0f, 0.0f, -0.001f);
    float sdf_z_m = InterpolateField_gpu(v_z_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
    if (sdf_z_p != sdf_z_p) {
        if (sdf_z_m != sdf_z_m) {
            grad.z = 0.0f;
        }
        else {
            grad.z = 1.0f;
        }
    }
    else {
        if (sdf_z_m != sdf_z_m) {
            grad.z = -1.0f;
        }
        else {
            grad.z = (sdf_z_p - sdf_z_m);
        }
    }

    return grad;
}

__device__ __forceinline__ void FitProcess(float * Vertices_out, float* sdf, float* Nodes, int* Tets, int* HashTable, float* Vertices, float* Normals, int Nb_Vertices, int OuterIter, float delta, float alpha, int thread_size)
{
    unsigned int i = threadIdx.x + blockIdx.x * THREAD_SIZE_X; // cols
    unsigned int j = threadIdx.y + blockIdx.y * THREAD_SIZE_Y; // rows
    unsigned int s = i * thread_size + j;

    if (s > Nb_Vertices - 1)
        return;

    float3 vertex = make_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
    float3 normal = make_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
    if (OuterIter < 3) {
        vertex = vertex - normal * 0.01f;
    }
    else {
        //cout << "Interpolated" << endl;
        float sdf_val = InterpolateField_gpu(vertex, sdf, Nodes, Tets, HashTable, make_int3(100, 100, 100), make_float3(0.0f, 0.0f, 0.0f), 0.03f);
        float3 grad = Gradient_gpu(vertex, sdf, Nodes, Tets, HashTable, make_int3(100, 100, 100), make_float3(0.0f, 0.0f, 0.0f), 0.03f);
        if (sdf_val != sdf_val) {
            vertex = vertex;// -normal * 0.01f;
        }
        else {
            float n_grad = length(grad);
            if (n_grad > 0.0f)
                grad = grad * (sdf_val / length(grad));

            if (dot(grad, normal) < 0.0f)
                vertex = vertex - grad * delta + normal * n_grad * alpha;
            else
                vertex = vertex - grad * delta - normal * n_grad * alpha;
        }
    }
    Vertices_out[3 * s] = vertex.x;
    Vertices_out[3 * s + 1] = vertex.y;
    Vertices_out[3 * s + 2] = vertex.z;
}

__global__ void FitKernel(float* Vertices_out, float* sdf, float* Nodes, int* Tets, int* HashTable, float* Vertices, float* Normals, int Nb_Vertices, int OuterIter, float delta, float alpha, int thread_size)
{
    FitProcess(Vertices_out, sdf, Nodes, Tets, HashTable, Vertices, Normals, Nb_Vertices, OuterIter, delta, alpha, thread_size);
}

__device__ __forceinline__ void InnerLoopProcess(float* Vertices_out, float* Vertices, float* Normals, int* RBF_id, float* RBF, int Nb_Vertices, int thread_size)
{
    unsigned int i = threadIdx.x + blockIdx.x * THREAD_SIZE_X; // cols
    unsigned int j = threadIdx.y + blockIdx.y * THREAD_SIZE_Y; // rows
    unsigned int s = i * thread_size + j;

    if (s > Nb_Vertices - 1)
        return;

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    float3 vtx = make_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
    float3 nmle = make_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);

    for (int k = 0; k < 16; k++) {
        if (RBF_id[16 * s + k] == -1)
            break;

        // Project points into tangent plane
        float3 curr_v = make_float3(Vertices[3 * RBF_id[16 * s + k]], Vertices[3 * RBF_id[16 * s + k] + 1], Vertices[3 * RBF_id[16 * s + k] + 2]);

        float ps = dot(curr_v - vtx, nmle);

        float3 v_proj = curr_v - nmle * ps;

        x += v_proj.x * RBF[16 * s + k];
        y += v_proj.y * RBF[16 * s + k];
        z += v_proj.z * RBF[16 * s + k];
    }

    if (x != 0.0f && y != 0.0f && z != 0.0f) {
        Vertices_out[3 * s] = x;
        Vertices_out[3 * s + 1] = y;
        Vertices_out[3 * s + 2] = z;
    }
}

__global__ void InnerLoopKernel(float* Vertices_out, float* Vertices, float* Normals, int* RBF_id, float* RBF, int Nb_Vertices, int thread_size)
{
    InnerLoopProcess(Vertices_out, Vertices, Normals, RBF_id, RBF, Nb_Vertices, thread_size);
}

//////////////////////////////////////////
///**** Host Function definitions ****/
/////////////////////////////////////////

void OuterLoop_gpu(float* Vertices_out_gpu, float* sdf_gpu, float* Nodes_gpu, int* Tets_gpu, int* HashTable_gpu, float* Vertices_gpu, float* Normals_gpu, int Nb_Vertices, int Nb_Nodes, int Nb_Tets, int OuterIter, float delta, float alpha_curr) {
    int thread_size = int(round(sqrt(Nb_Vertices)) + 1);
    dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y, THREAD_SIZE_Z);
    dim3 dimGrid(1, 1, 1);
    dimGrid.x = divUp(thread_size, dimBlock.x); // #cols
    dimGrid.y = divUp(thread_size, dimBlock.y); // # rows

    //std::cout << "Start OuterLoop on GPU" << std::endl;

    FitKernel <<< dimGrid, dimBlock >>> (Vertices_out_gpu, sdf_gpu, Nodes_gpu, Tets_gpu, HashTable_gpu, Vertices_gpu, Normals_gpu, Nb_Vertices, OuterIter, delta, alpha_curr, thread_size);

    checkCudaErrors(cudaDeviceSynchronize());
}


// This function creates a hash table that associate to each voxel a list of tetrahedras for quick search
int* GenerateHashTable_gpu(float* Nodes, int* Tets, int nb_tets, int3 size_grid, float3 center_grid, float res) {
    vector<int>* T_HashTable = new vector<int>[size_grid.x * size_grid.y * size_grid.z];

    // Go through each tetrahedra
    int count = 0;
    for (int t = 0; t < nb_tets; t++) {
        float3 s1 = make_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]);
        float3 s2 = make_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]);
        float3 s3 = make_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]);
        float3 s4 = make_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]);

        // compute min max
        float min_x = min(s1.x, min(s2.x, min(s3.x, s4.x)));
        float min_y = min(s1.y, min(s2.y, min(s3.y, s4.y)));
        float min_z = min(s1.z, min(s2.z, min(s3.z, s4.z)));

        float max_x = max(s1.x, max(s2.x, max(s3.x, s4.x)));
        float max_y = max(s1.y, max(s2.y, max(s3.y, s4.y)));
        float max_z = max(s1.z, max(s2.z, max(s3.z, s4.z)));

        // Convert to grid coordinates
        int l_x = int(floor((min_x - center_grid.x) / res)) + size_grid.x / 2;
        int l_y = int(floor((min_y - center_grid.y) / res)) + size_grid.y / 2;
        int l_z = int(floor((min_z - center_grid.z) / res)) + size_grid.z / 2;

        int u_x = int(ceil((max_x - center_grid.x) / res)) + size_grid.x / 2;
        int u_y = int(ceil((max_y - center_grid.y) / res)) + size_grid.y / 2;
        int u_z = int(ceil((max_z - center_grid.z) / res)) + size_grid.z / 2;

        for (int a = l_x; a < u_x + 1; a++) {
            for (int b = l_y; b < u_y + 1; b++) {
                for (int c = l_z; c < u_z + 1; c++) {
                    T_HashTable[a * size_grid.y * size_grid.z + b * size_grid.z + c].push_back(t);
                    count++;
                }
            }
        }
    }

    cout << count << endl;
    int* HashTable = new int[size_grid.x * size_grid.y * size_grid.z + count];

    int offset = 0;
    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            for (int k = 0; k < size_grid.z; k++) {
                int list_size = T_HashTable[i * size_grid.y * size_grid.z + j * size_grid.z + k].size();
                HashTable[i * size_grid.y * size_grid.z + j * size_grid.z + k] = offset;
                for (vector<int>::iterator it = T_HashTable[i * size_grid.y * size_grid.z + j * size_grid.z + k].begin(); it != T_HashTable[i * size_grid.y * size_grid.z + j * size_grid.z + k].end(); it++) {
                    HashTable[size_grid.x * size_grid.y * size_grid.z + offset] = (*it);
                    offset++;
                }

                T_HashTable[i * size_grid.y * size_grid.z + j * size_grid.z + k].clear();
            }
        }
    }
    cout << "DONE" << endl;
    delete []T_HashTable;

    int* HashTable_gpu;
    checkCudaErrors(cudaMalloc((void**)&HashTable_gpu, (size_grid.x * size_grid.y * size_grid.z + count) * sizeof(int)));
    checkCudaErrors(cudaMemcpy((void*)HashTable_gpu, (void*)HashTable, (size_grid.x * size_grid.y * size_grid.z + count) * sizeof(int), cudaMemcpyHostToDevice));
    delete[]HashTable;

    return HashTable_gpu;

}


void InnerLoopSurface_gpu(float* Vertices_out_gpu, float* Vertices_gpu, float* Normals_gpu, int* RBF_id_gpu, float* RBF_gpu, int Nb_Vertices) {
    
    int thread_size = int(round(sqrt(Nb_Vertices)) + 1);
    dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y, THREAD_SIZE_Z);
    dim3 dimGrid(1, 1, 1);
    dimGrid.x = divUp(thread_size, dimBlock.x); // #cols
    dimGrid.y = divUp(thread_size, dimBlock.y); // # rows

    //std::cout << "Start InnerLoop on GPU" << std::endl;

    InnerLoopKernel <<< dimGrid, dimBlock >>> (Vertices_out_gpu, Vertices_gpu, Normals_gpu, RBF_id_gpu, RBF_gpu, Nb_Vertices, thread_size);

    checkCudaErrors(cudaDeviceSynchronize());
}