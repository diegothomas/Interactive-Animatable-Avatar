//
//  Fit.h
//  fitLevelSet
//
//  Created by Diego Thomas on 2021/03/23.
//

#ifndef Fit_h
#define Fit_h
#include "include/Utilities.h"


namespace fitLevelSet {

    void ComputeRBF(float* RBF, int* RBF_id, float* Vertices, int* Faces, int Nb_Faces, int Nb_Vertices) {
        // 1. for each vertex get the list of triangles that is attached to the vertex
        vector<vector<int>> neigh_face;
        vector<vector<int>> neigh_n;
        for (int n = 0; n < Nb_Vertices; n++) {
            vector<int> tmp;
            tmp.clear();
            neigh_face.push_back(tmp);
            vector<int> tmp2;
            tmp2.clear();
            neigh_n.push_back(tmp2);
        }

        for (int f = 0; f < Nb_Faces; f++) {
            neigh_face[Faces[3 * f]].push_back(f);
            neigh_face[Faces[3 * f + 1]].push_back(f);
            neigh_face[Faces[3 * f + 2]].push_back(f);

            bool valid1 = true;
            bool valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[Faces[3 * f]].begin(); it_n != neigh_n[Faces[3 * f]].end(); it_n++) {
                if (Faces[3 * f + 1] == (*it_n))
                    valid1 = false;

                if (Faces[3 * f + 2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3 * f]].push_back(Faces[3 * f + 1]);

            if (valid2)
                neigh_n[Faces[3 * f]].push_back(Faces[3 * f + 2]);

            valid1 = true;
            valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[Faces[3 * f + 1]].begin(); it_n != neigh_n[Faces[3 * f + 1]].end(); it_n++) {
                if (Faces[3 * f] == (*it_n))
                    valid1 = false;

                if (Faces[3 * f + 2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3 * f + 1]].push_back(Faces[3 * f]);

            if (valid2)
                neigh_n[Faces[3 * f + 1]].push_back(Faces[3 * f + 2]);

            valid1 = true;
            valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[Faces[3 * f + 2]].begin(); it_n != neigh_n[Faces[3 * f + 2]].end(); it_n++) {
                if (Faces[3 * f + 1] == (*it_n))
                    valid1 = false;

                if (Faces[3 * f] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3 * f + 2]].push_back(Faces[3 * f + 1]);

            if (valid2)
                neigh_n[Faces[3 * f + 2]].push_back(Faces[3 * f]);
        }

        // 2. Go through each vertex
        for (int n = 0; n < Nb_Vertices; n++) {
            if (neigh_n[n].size() > 16)
                cout << "nb tet in vicinity S: " << n << ", " << neigh_n[n].size() << endl;

            // compute the total volume of the surrounding faces.
            float area_tot = 1.0f;
            for (vector<int>::iterator it = neigh_face[n].begin(); it != neigh_face[n].end(); it++) {
                int face = (*it);
                area_tot *= 100.0f * AreaFace(Vertices, Faces[3 * face], Faces[3 * face + 1], Faces[3 * face + 2]);
            }

            // for each neighborhing summit
            float tot_weight = 0.0f;
            int count = 0;
            for (vector<int>::iterator it_n = neigh_n[n].begin(); it_n != neigh_n[n].end(); it_n++) {
                float area_cur = area_tot;
                int curr_n = (*it_n);
                int s1 = -1;
                int s2 = -1;

                for (vector<int>::iterator it = neigh_face[n].begin(); it != neigh_face[n].end(); it++) {
                    int face = (*it);
                    assert(Faces[3 * face] == n || Faces[3 * face + 1] == n || Faces[3 * face + 2] == n);
                    if (Faces[3 * face] == curr_n || Faces[3 * face + 1] == curr_n || Faces[3 * face + 2] == curr_n) {
                        area_cur = area_cur / (100.0f * AreaFace(Vertices, Faces[3 * face], Faces[3 * face + 1], Faces[3 * face + 2]));
                        if (s1 == -1) {
                            if (Faces[3 * face] != n && Faces[3 * face] != curr_n) {
                                s1 = Faces[3 * face];
                            }
                            else if (Faces[3 * face + 1] != n && Faces[3 * face + 1] != curr_n) {
                                s1 = Faces[3 * face + 1];
                            }
                            else if (Faces[3 * face + 2] != n && Faces[3 * face + 2] != curr_n) {
                                s1 = Faces[3 * face + 2];
                            }
                        }
                        else {
                            if (Faces[3 * face] != n && Faces[3 * face] != curr_n) {
                                s2 = Faces[3 * face];
                            }
                            else if (Faces[3 * face + 1] != n && Faces[3 * face + 1] != curr_n) {
                                s2 = Faces[3 * face + 1];
                            }
                            else if (Faces[3 * face + 2] != n && Faces[3 * face + 2] != curr_n) {
                                s2 = Faces[3 * face + 2];
                            }
                        }
                    }
                }

                if (s1 == -1) {
                    //cout << "s1 == -1" << endl;
                    continue;
                }
                if (s2 == -1) {
                    //cout << "s2 == -1" << endl;
                    continue;
                    cout << n << endl;
                    cout << curr_n << endl;
                    cout << neigh_face[n].size() << endl;
                    for (vector<int>::iterator it = neigh_face[n].begin(); it != neigh_face[n].end(); it++) {
                        int face = (*it);
                        cout << Faces[3 * face] << ", " << Faces[3 * face + 1] << ", " << Faces[3 * face + 2] << endl;
                    }
                }
                assert(s1 != -1);
                assert(s2 != -1);

                RBF[16 * n + count] = 100.0f * AreaFace(Vertices, s1, curr_n, s2) * area_cur;
                RBF_id[16 * n + count] = curr_n;
                //cout << n << ", " << curr_n << endl;
                tot_weight += RBF[16 * n + count];
                count++;
            }

            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * n + k] == -1)
                    break;
                RBF[16 * n + k] = RBF[16 * n + k] / tot_weight;
                //cout << n << ", " << RBF[16*n + k] << endl;
            }
        }

    }

    // This function creates a hash table that associate to each voxel a list of tetrahedras for quick search
    vector<int>* GenerateHashTable(float* Nodes, int* Tets, int nb_tets, my_int3 size_grid, my_float3 center_grid, float res) {

        vector<int>* HashTable = new vector<int>[size_grid.x * size_grid.y * size_grid.z];

        // Go through each tetrahedra
        for (int t = 0; t < nb_tets; t++) {
            my_float3 s1 = make_my_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]);
            my_float3 s2 = make_my_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]);
            my_float3 s3 = make_my_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]);
            my_float3 s4 = make_my_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]);

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
                        HashTable[a * size_grid.y * size_grid.z + b * size_grid.z + c].push_back(t);
                    }
                }
            }
        }

        return HashTable;

    }

    float VolumeTetrahedra(my_float3 s1, my_float3 s2, my_float3 s3, my_float3 s4) {
        // compute area of base s1,s2,s3
        my_float3 n_vec = cross(s2 - s1, s3 - s1);
        float area_base = norm(n_vec) / 2.0f;

        // compute height
        // project s4 on the base plan
        my_float3 n_unit = n_vec * (1.0f / area_base);
        float dot_prod = dot(s4 - s1, n_unit);

        return (area_base * fabs(dot_prod)) / 3.0f;
    }

    float* bary_tet(my_float3 a, my_float3 b, my_float3 c, my_float3 d, my_float3 p)
    {
        my_float3 vap = p - a;
        my_float3 vbp = p - b;

        my_float3 vab = b - a;
        my_float3 vac = c - a;
        my_float3 vad = d - a;

        my_float3 vbc = c - b;
        my_float3 vbd = d - b;
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

    float* BarycentricCoordinates(my_float3 v, float* Nodes, int* Tets, int t) {
        float vol_tot = VolumeTetrahedra(make_my_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]));

        //Compute the volumes of inside tetras
        float vol_1 = VolumeTetrahedra(v,
            make_my_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]));

        float vol_2 = VolumeTetrahedra(make_my_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]),
            v,
            make_my_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]));

        float vol_3 = VolumeTetrahedra(make_my_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]),
            v,
            make_my_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]));

        float vol_4 = VolumeTetrahedra(make_my_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]),
            v);

        float* res = new float[4];
        res[0] = vol_1 / vol_tot;
        res[1] = vol_2 / vol_tot;
        res[2] = vol_3 / vol_tot;
        res[3] = vol_4 / vol_tot;
        return res;
    }

    bool IsInside(my_float3 v, float* Nodes, int* Tets, int t) {
        // Get four summits:
        my_float3 s1 = make_my_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]);
        my_float3 s2 = make_my_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]);
        my_float3 s3 = make_my_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]);
        my_float3 s4 = make_my_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]);

        // Compute normal of each face
        my_float3 n1 = cross(s2 - s1, s3 - s1);
        n1 = n1 * (1.0f / norm(n1));
        my_float3 n2 = cross(s2 - s1, s4 - s1);
        n2 = n2 * (1.0f / norm(n2));
        my_float3 n3 = cross(s3 - s2, s4 - s2);
        n3 = n3 * (1.0f / norm(n3));
        my_float3 n4 = cross(s3 - s1, s4 - s1);
        n4 = n4 * (1.0f / norm(n4));

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

        return o1 <= 0.0f && o2 <= 0.0f && o3 <= 0.0f && o4 <= 0.0f;
    }

    int GetTet(my_float3 v, float* Nodes, int* Tets, vector<int>* HashTable, my_int3 dim_grid, my_float3 center_grid, float res) {
        //compute the key for the vertex
        my_int3 v_id = make_my_int3(int((v.x - center_grid.x) / res) + dim_grid.x / 2,
            int((v.y - center_grid.y) / res) + dim_grid.y / 2,
            int((v.z - center_grid.z) / res) + dim_grid.z / 2);

        if (v_id.x < 0 || v_id.x >= dim_grid.x - 1 || v_id.y < 0 || v_id.y >= dim_grid.y - 1 || v_id.z < 0 || v_id.z >= dim_grid.z - 1) {
            return -1;
        }

        vector<int> listTets = HashTable[v_id.x * dim_grid.y * dim_grid.z + v_id.y * dim_grid.z + v_id.z];

        int best_t = -1;
        for (vector<int>::iterator it = listTets.begin(); it != listTets.end(); it++) {
            int t = (*it);
            //for (int t = 0; t < 400110; t++) {

                // test if v is inside the tetrahedron
            if (IsInside(v, Nodes, Tets, t)) {
                best_t = t;
                break;
            }
        }

        return best_t;

    }

    float InterpolateField(my_float3 v, float* sdf, float* Nodes, int* Tets, vector<int>* HashTable, my_int3 dim_grid, my_float3 center, float res) {

        int t = GetTet(v, Nodes, Tets, HashTable, dim_grid, center, res);
        if (t == -1) {
            // point ouside the field
            //cout << v.x << ", " << v.y << ", " << v.z << endl;
            //cout << "point outside" << endl;
            return NAN;
        }

        //Get barycentric coordinates
        //float *B_coo = BarycentricCoordinates(v, Nodes, Tets, t);
        float* B_coo = bary_tet(make_my_float3(Nodes[3 * Tets[4 * t]], Nodes[3 * Tets[4 * t] + 1], Nodes[3 * Tets[4 * t] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 1]], Nodes[3 * Tets[4 * t + 1] + 1], Nodes[3 * Tets[4 * t + 1] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 2]], Nodes[3 * Tets[4 * t + 2] + 1], Nodes[3 * Tets[4 * t + 2] + 2]),
            make_my_float3(Nodes[3 * Tets[4 * t + 3]], Nodes[3 * Tets[4 * t + 3] + 1], Nodes[3 * Tets[4 * t + 3] + 2]), v);

        assert(B_coo[0] <= 1.0f && B_coo[1] <= 1.0f && B_coo[2] <= 1.0f && B_coo[3] <= 1.0f);

        /*cout << "BarycentricCoordinates" << B_coo[0] << ", " << B_coo[1] << ", " << B_coo[2] << ", " << B_coo[3] << endl;
        v.print();
        cout << "s1: " << Nodes[Tets[4*t]] << ", " << Nodes[Tets[4*t]+1] << ", " << Nodes[Tets[4*t]+2] << ", " << endl;
        cout << "sdf1: " << sdf[Tets[4*t]] << ", " << endl;
        cout << "s2: " << Nodes[Tets[4*t+1]] << ", " << Nodes[Tets[4*t+1]+1] << ", " << Nodes[Tets[4*t+1]+2] << ", " << endl;
        cout << "sdf3: " << sdf[Tets[4*t+1]] << ", " << endl;
        cout << "s3: " << Nodes[Tets[4*t+2]] << ", " << Nodes[Tets[4*t+2]+1] << ", " << Nodes[Tets[4*t+2]+2] << ", " << endl;
        cout << "sdf4: " << sdf[Tets[4*t+2]] << ", " << endl;
        cout << "s4: " << Nodes[Tets[4*t+3]] << ", " << Nodes[Tets[4*t+3]+1] << ", " << Nodes[Tets[4*t+3]+2] << ", " << endl;
        cout << "sdf5: " << sdf[Tets[4*t+3]] << ", " << endl;*/

        float out_res = (B_coo[0] * sdf[Tets[4 * t]] + B_coo[1] * sdf[Tets[4 * t + 1]] + B_coo[2] * sdf[Tets[4 * t + 2]] + B_coo[3] * sdf[Tets[4 * t + 3]]) /
            (B_coo[0] + B_coo[1] + B_coo[2] + B_coo[3]);
        delete[] B_coo;

        return out_res;
    }

    my_float3 Gradient(my_float3 v, float* sdf, float* Nodes, int* Tets, vector<int>* HashTable, my_int3 dim_grid, my_float3 center, float res) {

        my_float3 grad = make_my_float3(0.0f, 0.0f, 0.0f);

        my_float3 v_x_p = v + make_my_float3(0.001f, 0.0f, 0.0f);
        float sdf_x_p = InterpolateField(v_x_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        //if (sdf_x_p != sdf_x_p) {
        //    return make_my_float3(0.0f, 0.0f, 0.0f);
        //}
        my_float3 v_x_m = v + make_my_float3(-0.001f, 0.0f, 0.0f);
        float sdf_x_m = InterpolateField(v_x_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        //if (sdf_x_m != sdf_x_m) {
        //    return make_my_float3(0.0f, 0.0f, 0.0f);
        //}
        //grad.x = (sdf_x_p - sdf_x_m);

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
        //grad.x = (sdf_x_p - sdf_x_m);// / 0.002f;

        my_float3 v_y_p = v + make_my_float3(0.0f, 0.001f, 0.0f);
        float sdf_y_p = InterpolateField(v_y_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        //if (sdf_y_p != sdf_y_p) {
        //    return make_my_float3(0.0f, 0.0f, 0.0f);
        //}
        my_float3 v_y_m = v + make_my_float3(0.0f, -0.001f, 0.0f);
        float sdf_y_m = InterpolateField(v_y_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        //if (sdf_y_m != sdf_y_m) {
        //    return make_my_float3(0.0f, 0.0f, 0.0f);
        //}
        //grad.y = (sdf_y_p - sdf_y_m);

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
        //grad.y = (sdf_y_p - sdf_y_m);// / 0.002f;

        my_float3 v_z_p = v + make_my_float3(0.0f, 0.0f, 0.001f);
        float sdf_z_p = InterpolateField(v_z_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res); 
        //if (sdf_z_p != sdf_z_p) {
        //    return make_my_float3(0.0f, 0.0f, 0.0f);
        //}
        my_float3 v_z_m = v + make_my_float3(0.0f, 0.0f, -0.001f);
        float sdf_z_m = InterpolateField(v_z_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        //if (sdf_z_m != sdf_z_m) {
        //    return make_my_float3(0.0f, 0.0f, 0.0f);
        //}
        //grad.z = (sdf_z_p - sdf_z_m);
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
        //grad.z = (sdf_z_p - sdf_z_m);// / 0.002f;

        /*if (grad.x != grad.x || grad.y != grad.y || grad.z != grad.z) {
            //cout << "NAN" << endl;
            return make_my_float3(0.0f, 0.0f, 0.0f);
        }*/

        return grad;
    }


    void CleanMesh(int* Faces, float* Vertices, float* vtx_copy, int Nb_Faces, int Nb_Vertices) {
        for (int f = 0; f < Nb_Faces; f++) {
            int s = Faces[3 * f];
            int s1 = Faces[3 * f + 1];
            int s2 = Faces[3 * f + 2];
            my_float3 vertex = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
            my_float3 vertex1 = make_my_float3(vtx_copy[3 * s1], vtx_copy[3 * s1 + 1], vtx_copy[3 * s1 + 2]);
            my_float3 vertex2 = make_my_float3(vtx_copy[3 * s2], vtx_copy[3 * s2 + 1], vtx_copy[3 * s2 + 2]);

            float d1 = norm(vertex1 - vertex);
            float d2 = norm(vertex2 - vertex);
            float d3 = norm(vertex1 - vertex2);
            if (d1 > 0.03f || d2 > 0.03f || d3 > 0.03f) {
                Vertices[3 * s] = 0.0f;
                Vertices[3 * s + 1] = 0.0f;
                Vertices[3 * s + 2] = 0.0f;
            }
        }
        for (int f = 0; f < Nb_Faces; f++) {
            int s = Faces[3 * f];
            int s1 = Faces[3 * f + 1];
            int s2 = Faces[3 * f + 2];
            my_float3 vertex = make_my_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
            my_float3 vertex1 = make_my_float3(Vertices[3 * s1], Vertices[3 * s1 + 1], Vertices[3 * s1 + 2]);
            my_float3 vertex2 = make_my_float3(Vertices[3 * s2], Vertices[3 * s2 + 1], Vertices[3 * s2 + 2]);

            float d1 = norm(vertex1);
            float d2 = norm(vertex2);
            float d3 = norm(vertex);
            if (d1 == 0.0f || d2 == 0.0f || d3 == 0.0f) {
                Faces[3 * f] = 0;
                Faces[3 * f + 1] = 0;
                Faces[3 * f + 2] = 0;
            }
        }
    }

    void BuildFieldForce(float * force_field, float* sdf, float* Nodes, int* Tets, vector<int>* HashTable, int* Faces, float* Vertices, float* vtx_copy, float* Normals, int* RBF_id, float* RBF,
        int Nb_Faces, int Nb_Vertices, int OuterIter, float delta, float alpha_curr, float kappa, int id) {
        for (int f = id; f < min(Nb_Faces, id + Nb_Faces / 100); f++) {
            // compute forces for the summit 1
            int s = Faces[3 * f];
            int s1 = Faces[3 * f + 1];
            int s2 = Faces[3 * f + 2];
            my_float3 vertex = make_my_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
            my_float3 vertex1 = make_my_float3(Vertices[3 * s1], Vertices[3 * s1 + 1], Vertices[3 * s1 + 2]);
            my_float3 vertex2 = make_my_float3(Vertices[3 * s2], Vertices[3 * s2 + 1], Vertices[3 * s2 + 2]);

            my_float3 in_vertex = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
            my_float3 in_vertex1 = make_my_float3(vtx_copy[3 * s1], vtx_copy[3 * s1 + 1], vtx_copy[3 * s1 + 2]);
            my_float3 in_vertex2 = make_my_float3(vtx_copy[3 * s2], vtx_copy[3 * s2 + 1], vtx_copy[3 * s2 + 2]);

            my_float3 normal = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
            my_float3 force_curr = make_my_float3(0.0f, 0.0f, 0.0f);
            if (OuterIter < 4) {
                force_curr = normal * -0.01f;
            }
            else {
                float sdf_val = InterpolateField(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                my_float3 grad = Gradient(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                if (sdf_val != sdf_val || norm(grad) == 0.0f) {
                    force_curr = normal * -0.01f;
                }
                else {
                    //force_curr = normal * -sdf_val * alpha_curr;
                    /*grad = grad * (fabs(sdf_val) / norm(grad));
                    if (sdf_val > 0.0f) { // move inside towards negative normal direction
                        if (dot(grad, normal) < 0.0f)
                            force_curr = grad * delta - normal * norm(grad) * alpha_curr;
                        else
                            force_curr = grad * -delta - normal * norm(grad) * alpha_curr;
                    }
                    else {// move outside towards positive normal direction
                        if (dot(grad, normal) < 0.0f)
                            force_curr = grad * -delta + normal * norm(grad) * alpha_curr;
                        else
                            force_curr = grad * delta + normal * norm(grad) * alpha_curr;
                    }*/

                    if (norm(grad) > 0.0f)
                        grad = grad * (sdf_val / norm(grad));

                    if (dot(grad, normal) < 0.0f)
                        force_curr = grad * -delta + normal * norm(grad) * alpha_curr;
                    else
                        force_curr = grad * -delta - normal * norm(grad) * alpha_curr;
                }
            }
            //force_curr = force_curr + ((vertex1 - vertex) - (in_vertex1 - in_vertex)) * kappa;
            //force_curr = force_curr + ((vertex2 - vertex) - (in_vertex2 - in_vertex)) * kappa;
            force_field[4 * s] = force_field[4 * s] + force_curr.x;
            force_field[4 * s + 1] = force_field[4 * s + 1] + force_curr.y;
            force_field[4 * s + 2] = force_field[4 * s + 2] + force_curr.z;
            force_field[4 * s + 3] = force_field[4 * s + 3] + 1.0f;

            /// summit 1
            normal = make_my_float3(Normals[3 * s1], Normals[3 * s1 + 1], Normals[3 * s1 + 2]);
            force_curr = make_my_float3(0.0f, 0.0f, 0.0f);
            if (OuterIter < 4) {
                force_curr = normal * -0.01f;
            }
            else {
                float sdf_val = InterpolateField(vertex1, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                my_float3 grad = Gradient(vertex1, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                if (sdf_val != sdf_val || norm(grad) == 0.0f) {
                    force_curr = normal * -0.01f;
                }
                else {
                    //force_curr = normal * -sdf_val * alpha_curr;
                    /*grad = grad * (fabs(sdf_val) / norm(grad));
                    if (sdf_val > 0.0f) { // move inside towards negative normal direction
                        if (dot(grad, normal) < 0.0f)
                            force_curr = grad * delta - normal * norm(grad) * alpha_curr;
                        else
                            force_curr = grad * -delta - normal * norm(grad) * alpha_curr;
                    }
                    else {// move outside towards positive normal direction
                        if (dot(grad, normal) < 0.0f)
                            force_curr = grad * -delta + normal * norm(grad) * alpha_curr;
                        else
                            force_curr = grad * delta + normal * norm(grad) * alpha_curr;
                    }*/

                    if (norm(grad) > 0.0f)
                        grad = grad * (sdf_val / norm(grad));

                    if (dot(grad, normal) < 0.0f)
                        force_curr = grad * -delta + normal * norm(grad) * alpha_curr;
                    else
                        force_curr = grad * -delta - normal * norm(grad) * alpha_curr;
                }
            }
            //force_curr = force_curr + ((vertex - vertex1) - (in_vertex - in_vertex1)) * kappa;
            //force_curr = force_curr + ((vertex2 - vertex1) - (in_vertex2 - in_vertex1)) * kappa;
            force_field[4 * s1] = force_field[4 * s1] + force_curr.x;
            force_field[4 * s1 + 1] = force_field[4 * s1 + 1] + force_curr.y;
            force_field[4 * s1 + 2] = force_field[4 * s1 + 2] + force_curr.z;
            force_field[4 * s1 + 3] = force_field[4 * s1 + 3] + 1.0f;

            /// summit 2
            normal = make_my_float3(Normals[3 * s2], Normals[3 * s2 + 1], Normals[3 * s2 + 2]);
            force_curr = make_my_float3(0.0f, 0.0f, 0.0f);
            if (OuterIter < 4) {
                force_curr = normal * -0.01f;
            }
            else {
                float sdf_val = InterpolateField(vertex2, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                my_float3 grad = Gradient(vertex2, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                if (sdf_val != sdf_val || norm(grad) == 0.0f) {
                    force_curr = normal * -0.01f;
                }
                else {
                    //force_curr = normal * -sdf_val * alpha_curr;
                    /*grad = grad * (fabs(sdf_val) / norm(grad));
                    if (sdf_val > 0.0f) { // move inside towards negative normal direction
                        if (dot(grad, normal) < 0.0f)
                            force_curr = grad * delta - normal * norm(grad) * alpha_curr;
                        else
                            force_curr = grad * -delta - normal * norm(grad) * alpha_curr;
                    }
                    else {// move outside towards positive normal direction
                        if (dot(grad, normal) < 0.0f)
                            force_curr = grad * -delta + normal * norm(grad) * alpha_curr;
                        else
                            force_curr = grad * delta + normal * norm(grad) * alpha_curr;
                    }*/

                    if (norm(grad) > 0.0f)
                        grad = grad * (sdf_val / norm(grad));

                    if (dot(grad, normal) < 0.0f)
                        force_curr = grad * -delta + normal * norm(grad) * alpha_curr;
                    else
                        force_curr = grad * -delta - normal * norm(grad) * alpha_curr;
                }
            }
            //force_curr = force_curr + ((vertex1 - vertex2) - (in_vertex1 - in_vertex2)) * kappa;
            //force_curr = force_curr + ((vertex - vertex2) - (in_vertex - in_vertex2)) * kappa;
            force_field[4 * s2] = force_field[4 * s2] + force_curr.x;
            force_field[4 * s2 + 1] = force_field[4 * s2 + 1] + force_curr.y;
            force_field[4 * s2 + 2] = force_field[4 * s2 + 2] + force_curr.z;
            force_field[4 * s2 + 3] = force_field[4 * s2 + 3] + 1.0f;
        }
    }

    void OuterLoopTot(float* sdf, float* Nodes, int* Tets, vector<int>* HashTable, int *Faces, float* Vertices, float* vtx_copy, float* Normals, int* RBF_id, float* RBF,
        int Nb_Faces, int Nb_Vertices, int OuterIter, float delta, float alpha_curr, float kappa) {
        float* force_field = new float[4 * Nb_Vertices];
        for (int s = 0; s < Nb_Vertices; s++) {
            force_field[4 * s] = 0.0f;
            force_field[4 * s + 1] = 0.0f;
            force_field[4 * s + 2] = 0.0f;
            force_field[4 * s + 3] = 0.0f;
        }
        std::vector< std::thread > my_threads;
        for (int i = 0; i < 101; i++) {
            my_threads.push_back(std::thread(BuildFieldForce, force_field, sdf, Nodes, Tets, HashTable, Faces, Vertices, vtx_copy, Normals, RBF_id, RBF,
                Nb_Faces, Nb_Vertices, OuterIter, delta, alpha_curr, kappa, i * (Nb_Faces / 100)));
        }
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
               

        // Increment deplacement of surface vertices
        for (int s = 0; s < Nb_Vertices; s++) {
            if (force_field[4 * s + 3] > 0.0f) {
                Vertices[3 * s] = Vertices[3 * s] + force_field[4 * s] / force_field[4 * s + 3];
                Vertices[3 * s + 1] = Vertices[3 * s + 1] + force_field[4 * s + 1] / force_field[4 * s + 3];
                Vertices[3 * s + 2] = Vertices[3 * s + 2] + force_field[4 * s + 2] / force_field[4 * s + 3];
            }
        }

        delete[] force_field;


    }

    void Advection(float* sdf, float* Nodes, int* Tets, vector<int>* HashTable, float* Vertices, float* Normals, int Nb_Vertices, int OuterIter, float delta, float alpha_curr, int id) {
        for (int s = id; s < min(Nb_Vertices, id + Nb_Vertices / 100); s++) {
            my_float3 vertex = make_my_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);

            my_float3 normal = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
            my_float3 force_curr = make_my_float3(0.0f, 0.0f, 0.0f);
            if (OuterIter < 4) {
                force_curr = normal * -0.01f;
            }
            else {
                float sdf_val = InterpolateField(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                my_float3 grad = Gradient(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                if (sdf_val != sdf_val || norm(grad) == 0.0f) {
                    force_curr = normal * -0.01f;
                }
                else {
                    if (norm(grad) > 0.0f)
                        grad = grad * (sdf_val / norm(grad));

                    if (dot(grad, normal) < 0.0f)
                        force_curr = grad * -delta + normal * norm(grad) * alpha_curr;
                    else
                        force_curr = grad * -delta - normal * norm(grad) * alpha_curr;
                }
            }
            Vertices[3 * s] = Vertices[3 * s] + force_curr.x;
            Vertices[3 * s + 1] = Vertices[3 * s + 1] + force_curr.y;
            Vertices[3 * s + 2] = Vertices[3 * s + 2] + force_curr.z;
        }
    }

    void OuterLoopMT(float* sdf, float* Nodes, int* Tets, vector<int>* HashTable, int* Faces, float* Vertices, float* vtx_copy, float* Normals, int* RBF_id, float* RBF,
        int Nb_Faces, int Nb_Vertices, int OuterIter, float delta, float alpha_curr, float kappa) {
        
        std::vector< std::thread > my_threads;
        for (int i = 0; i < 101; i++) {
            my_threads.push_back(std::thread(Advection, sdf, Nodes, Tets, HashTable, Vertices, Normals, Nb_Vertices, OuterIter, delta, alpha_curr, i * (Nb_Vertices / 100)));
        }
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    }

    void OuterLoop(float *sdf, float *Nodes, int *Tets, vector<int>* HashTable, float* Vertices, float* vtx_copy, float* Normals, int* RBF_id, float* RBF,
        int Nb_Vertices, int OuterIter, float delta, float alpha_curr) {
        // Increment deplacement of surface vertices
        for (int s = 0; s < Nb_Vertices; s++) {
            my_float3 vertex = make_my_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
            my_float3 normal = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
            if (OuterIter < 3) {
                vertex = vertex - normal * 0.01f;
            }
            else {
                float sdf_val = InterpolateField(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                my_float3 grad = Gradient(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                //cout << "sdf_val: " << sdf_val << endl;
                if (sdf_val != sdf_val) {
                    vertex = vertex - normal * 0.01f;
                }
                else {
                    //cout << "sdf: " << sdf_val << endl;
                    if (norm(grad) > 0.0f)
                        grad = grad * (sdf_val / norm(grad));

                    //cout << grad.x << ", " << grad.y << ", " << grad.z << endl;
                    //grad.Print();
                    if (dot(grad, normal) < 0.0f)
                        vertex = vertex - grad * delta + normal * norm(grad) * alpha_curr;
                    else
                        vertex = vertex - grad * delta - normal * norm(grad) * alpha_curr;
                }
            }
            Vertices[3 * s] = vertex.x;
            Vertices[3 * s + 1] = vertex.y;
            Vertices[3 * s + 2] = vertex.z;
        }
    }


    void OuterLoopCubic(float*** tsdf_grid, my_float3*** grad_grid, my_int3 dim_grid, my_float3 center, float res,
        float* Vertices, float* Normals,
        int Nb_Vertices, int OuterIter, float delta, float alpha_curr, float iso) {
        // Increment deplacement of surface vertices
        for (int s = 0; s < Nb_Vertices; s++) {
            my_float3 vertex = make_my_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
            my_float3 normal = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
            if (OuterIter < 3) {
                vertex = vertex - normal * 0.01f;
            }
            else {
                my_float3 grad = BicubiInterpolation(vertex, tsdf_grid, grad_grid, dim_grid, center, res, iso);
                //cout << "sdf_val: " << sdf_val << endl;
                if (dot(grad, grad) == 0.0f) {
                    vertex = vertex - normal * 0.01f;
                }
                else {
                    //cout << grad.x << ", " << grad.y << ", " << grad.z << endl;
                    //grad.Print();
                    if (dot(grad, normal) > 0.0f)
                        vertex = vertex + grad * delta + normal * norm(grad) * alpha_curr;
                    else
                        vertex = vertex + grad * delta - normal * norm(grad) * alpha_curr;
                }
            }
            Vertices[3 * s] = vertex.x;
            Vertices[3 * s + 1] = vertex.y;
            Vertices[3 * s + 2] = vertex.z;
        }
    }

    void InnerLoopSurface(float *Vertices, float * vtx_copy, float*Normals, int* RBF_id, float *RBF, int Nb_Vertices) {
        for (int s = 0; s < Nb_Vertices; s++) {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
            my_float3 nmle = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);

            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * s + k] == -1)
                    break;

                // Project points into tangent plane
                my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);
                if (curr_v.x > 1000.0f)
                    continue;

                float ps = dot(curr_v - vtx, nmle);

                my_float3 v_proj = curr_v - nmle * ps;

                if (RBF[16 * s + k] == RBF[16 * s + k]) {
                    x += v_proj.x * RBF[16 * s + k];
                    y += v_proj.y * RBF[16 * s + k];
                    z += v_proj.z * RBF[16 * s + k];
                    /*x += vtx_copy[3*RBF_id[16*s + k]]*RBF[16*s + k];
                    y += vtx_copy[3*RBF_id[16*s + k]+1]*RBF[16*s + k];
                    z += vtx_copy[3*RBF_id[16*s + k]+2]*RBF[16*s + k];*/
                }
                else {
                    cout << "Nan for RBF" << endl;
                }
            }

            if (x != 0.0f && y != 0.0f && z != 0.0f) {
                Vertices[3 * s] = x;
                Vertices[3 * s + 1] = y;
                Vertices[3 * s + 2] = z;
            }
        }
    }

    void InnerLoopSurface_multi(float* Vertices, float* vtx_copy, float* Normals, int* RBF_id, float* RBF, int Nb_Vertices , int id) {
        //for (int s = 0; s < Nb_Vertices; s++) {
        for (int s = id; s < min(Nb_Vertices, id + Nb_Vertices / 100); s++) {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
            my_float3 nmle = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);

            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * s + k] == -1)
                    break;

                // Project points into tangent plane
                my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);
                if (curr_v.x > 1000.0f)
                    continue;

                float ps = dot(curr_v - vtx, nmle);

                my_float3 v_proj = curr_v - nmle * ps;

                if (RBF[16 * s + k] == RBF[16 * s + k]) {
                    x += v_proj.x * RBF[16 * s + k];
                    y += v_proj.y * RBF[16 * s + k];
                    z += v_proj.z * RBF[16 * s + k];
                    /*x += vtx_copy[3*RBF_id[16*s + k]]*RBF[16*s + k];
                    y += vtx_copy[3*RBF_id[16*s + k]+1]*RBF[16*s + k];
                    z += vtx_copy[3*RBF_id[16*s + k]+2]*RBF[16*s + k];*/
                }
                else {
                    cout << "Nan for RBF" << endl;
                }
            }

            if (x != 0.0f && y != 0.0f && z != 0.0f) {
                Vertices[3 * s] = x;
                Vertices[3 * s + 1] = y;
                Vertices[3 * s + 2] = z;
            }
        }
    }

    void InnerLoopSurface(float* Vertices, float* vtx_copy, float* Normals, int* RBF_id, float* RBF, int Nb_Vertices , bool* valid_vtx_list , bool* valid_vtx_list_copy , bool* valid_vtx_list_init) {
        for (int s = 0; s < Nb_Vertices; s++) {
            if (valid_vtx_list_init[s] == true) {
                continue;
            }

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
            my_float3 nmle = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);

            float rbf_sum  = 0.0f;
            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * s + k] == -1)
                    break;
                if (valid_vtx_list_copy[s] == false) {
                    continue;
                }
                // Project points into tangent plane
                my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);
                if (curr_v.x > 1000.0f)
                    continue;

                float ps = dot(curr_v - vtx, nmle);
                my_float3 v_proj = curr_v - nmle * ps;

                if (RBF[16 * s + k] == RBF[16 * s + k]) {
                    x += v_proj.x * RBF[16 * s + k];
                    y += v_proj.y * RBF[16 * s + k];
                    z += v_proj.z * RBF[16 * s + k];
                    /*x += vtx_copy[3*RBF_id[16*s + k]]*RBF[16*s + k];
                    y += vtx_copy[3*RBF_id[16*s + k]+1]*RBF[16*s + k];
                    z += vtx_copy[3*RBF_id[16*s + k]+2]*RBF[16*s + k];*/
                    rbf_sum += RBF[16 * s + k];
                }
                else {
                    cout << "Nan for RBF" << endl;
                }
            }

            if (rbf_sum != 0.0f) {
                Vertices[3 * s    ] = x / rbf_sum;
                Vertices[3 * s + 1] = y / rbf_sum;
                Vertices[3 * s + 2] = z / rbf_sum;
                valid_vtx_list[s] = true;
            }
        }
    }

    void InnerLoopSurface2(float* Vertices, float* vtx_copy, int* RBF_id, float* RBF, int Nb_Vertices) {
        for (int s = 0; s < Nb_Vertices; s++) {
            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * s + k] == -1)
                    break;

                // Project points into tangent plane
                my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);
                if (curr_v.x > 1000.0f)
                    continue;

                x += curr_v.x * RBF[16 * s + k];
                y += curr_v.y * RBF[16 * s + k];
                z += curr_v.z * RBF[16 * s + k];
            }

            if (x != 0.0f && y != 0.0f && z != 0.0f) {
                Vertices[3 * s] = x;
                Vertices[3 * s + 1] = y;
                Vertices[3 * s + 2] = z;
            }
        }
    }


    void InnerLoopSurface2(float* Vertices, float* vtx_copy, int* RBF_id, float* RBF, int Nb_Vertices, bool* valid_vtx_list, bool* valid_vtx_list_copy, bool* valid_vtx_list_init) {
        for (int s = 0; s < Nb_Vertices; s++) {
            if (valid_vtx_list_init[s] == true) {
                continue;
            }

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);

            float rbf_sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * s + k] == -1)
                    break;
                if (valid_vtx_list_copy[RBF_id[16 * s + k]] == false) {
                    continue;
                }

                my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);
                if (curr_v.x > 1000.0f)
                    continue;

                if (RBF[16 * s + k] == RBF[16 * s + k]) {
                    x += curr_v.x * RBF[16 * s + k];
                    y += curr_v.y * RBF[16 * s + k];
                    z += curr_v.z * RBF[16 * s + k];
                    rbf_sum += RBF[16 * s + k];
                }
                else {
                    cout << "Nan for RBF" << endl;
                }
            }

            if (rbf_sum != 0.0f) {
                Vertices[3 * s    ] = x / rbf_sum;
                Vertices[3 * s + 1] = y / rbf_sum;
                Vertices[3 * s + 2] = z / rbf_sum;
                valid_vtx_list[s] = true;
            }
        }
    }

    int InnerLoopSurface_init(float* Vertices, float* vtx_copy, int* RBF_id, float* RBF, int Nb_Vertices, bool* valid_vtx_list, bool* valid_vtx_list_copy) {
        int fin_cnt = 0;
        for (int s = 0; s < Nb_Vertices; s++) {
            if (valid_vtx_list_copy[s] == true) {
                valid_vtx_list[s] = true;
                continue;
            }

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);

            float rbf_sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * s + k] == -1)
                    break;
                if (valid_vtx_list_copy[RBF_id[16 * s + k]] == false) {
                    continue;
                }

                my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);
                if (curr_v.x > 1000.0f)
                    continue;

                if (RBF[16 * s + k] == RBF[16 * s + k]) {
                    x += curr_v.x * RBF[16 * s + k];
                    y += curr_v.y * RBF[16 * s + k];
                    z += curr_v.z * RBF[16 * s + k];
                    rbf_sum += RBF[16 * s + k];
                }
                else {
                    cout << "Nan for RBF" << endl;
                }
            }

            if (rbf_sum != 0.0f) {
                Vertices[3 * s    ] = x / rbf_sum;
                Vertices[3 * s + 1] = y / rbf_sum;
                Vertices[3 * s + 2] = z / rbf_sum;
                valid_vtx_list[s] = true;
                fin_cnt += 1;
            }
        }

        if (fin_cnt > 0) {
            return 0;
        }
        else {
            return 1;
        }
    }


    int Move_invalid_to_valid_position(float* Vertices, float* vtx_copy, int Nb_Vertices , int* Faces, int Nb_Faces, bool* valid_vtx_list, bool* valid_vtx_list_copy) {
        // 1. for each vertex get the list of triangles that is attached to the vertex
        vector<vector<int>> neigh_n;
        for (int n = 0; n < Nb_Vertices; n++) {
            vector<int> tmp2;
            tmp2.clear();
            neigh_n.push_back(tmp2);
        }

        for (int f = 0; f < Nb_Faces; f++) {
            bool valid1 = true;
            bool valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[Faces[3 * f]].begin(); it_n != neigh_n[Faces[3 * f]].end(); it_n++) {
                if (Faces[3 * f + 1] == (*it_n))
                    valid1 = false;

                if (Faces[3 * f + 2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3 * f]].push_back(Faces[3 * f + 1]);

            if (valid2)
                neigh_n[Faces[3 * f]].push_back(Faces[3 * f + 2]);

            valid1 = true;
            valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[Faces[3 * f + 1]].begin(); it_n != neigh_n[Faces[3 * f + 1]].end(); it_n++) {
                if (Faces[3 * f] == (*it_n))
                    valid1 = false;

                if (Faces[3 * f + 2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3 * f + 1]].push_back(Faces[3 * f]);

            if (valid2)
                neigh_n[Faces[3 * f + 1]].push_back(Faces[3 * f + 2]);

            valid1 = true;
            valid2 = true;
            for (vector<int>::iterator it_n = neigh_n[Faces[3 * f + 2]].begin(); it_n != neigh_n[Faces[3 * f + 2]].end(); it_n++) {
                if (Faces[3 * f + 1] == (*it_n))
                    valid1 = false;

                if (Faces[3 * f] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3 * f + 2]].push_back(Faces[3 * f + 1]);

            if (valid2)
                neigh_n[Faces[3 * f + 2]].push_back(Faces[3 * f]);
        }

        int fin_cnt = 0;
        for (int s = 0; s < Nb_Vertices; s++) {
            if (valid_vtx_list_copy[s] == true) {
                continue;
            }

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
            
            int cnt = 0;
            float min_dist = 1.0e32f;
            for (vector<int>::iterator it_n = neigh_n[s].begin(); it_n != neigh_n[s].end(); it_n++) {
                if (valid_vtx_list_copy[*it_n] == false) {
                    continue;
                }
                my_float3 curr_v = make_my_float3(vtx_copy[3 * (*it_n)], vtx_copy[3 * (*it_n) + 1], vtx_copy[3 * (*it_n) + 2]);

                // Compute distance point to point
                float curr_dist = sqrt(dot(vtx - curr_v, vtx - curr_v));
                if (curr_dist < min_dist) {
                    min_dist = curr_dist;
                    x = curr_v.x;
                    y = curr_v.y;
                    z = curr_v.z;
                    cnt += 1;
                }
            }

            if (cnt > 0) {
                Vertices[3 * s + 0] = x;
                Vertices[3 * s + 1] = y;
                Vertices[3 * s + 2] = z;
                valid_vtx_list[s] = true;
                fin_cnt += 1;
            }
        }

        if (fin_cnt > 0) {
            return 0;
        }
        else {
            return 1;
        }
    }


    void FitToLevelSet_new(float* sdf, float* Nodes, int* Tets, int nb_tets, float* Vertices, float* Normals, int* Faces, int Nb_Vertices, int Nb_Faces, int outerloop_maxiter = 20, int innerloop_maxiter = 10, float alpha = 0.01f, float delta = 0.003f, float delta_elastic = 0.01f) {
        float* RBF = new float[16 * Nb_Vertices];
        int* RBF_id = new int[16 * Nb_Vertices];
        for (int n = 0; n < Nb_Vertices; n++) {
            for (int i = 0; i < 16; i++)
                RBF_id[16 * n + i] = -1;
        }
        ComputeRBF(RBF, RBF_id, Vertices, Faces, Nb_Faces, Nb_Vertices);
        vector<int>* HashTable = GenerateHashTable(Nodes, Tets, nb_tets, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
        // Copy of vertex position
        float* vtx_copy;
        vtx_copy = new float[3 * Nb_Vertices];
        memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
        //############## Outer loop ##################
        float alpha_curr = alpha;
        for (int OuterIter = 0; OuterIter < outerloop_maxiter; OuterIter++) {
            cout << "OuterIter" << endl;
            // Copy current state
            memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
            // Increment deplacement of surface vertices
            for (int s = 0; s < Nb_Vertices; s++) {
                my_float3 vertex = make_my_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
                my_float3 normal = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
                if (OuterIter < 3) {
                    vertex = vertex - normal * 0.01f;
                }
                else {
                    my_float3 grad = Gradient(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                    float sdf_val = InterpolateField(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                    if (sdf_val != sdf_val) {
                        vertex = vertex - normal * 0.001f;
                    }
                    else {
                        //cout << "norm(grad): " << norm(grad) << endl;
                        if (norm(grad) > 0.0f)
                            grad = grad * (sdf_val / norm(grad));
                        //cout << grad.x << ", " << grad.y << ", " << grad.z << endl;
                        if (dot(grad, normal) < 0.0f)
                            vertex = vertex - grad * delta + normal * norm(grad) * alpha_curr;
                        //vertex = vertex + grad * delta - normal * norm(grad) * alpha_curr;
                        else
                            vertex = vertex - grad * delta - normal * norm(grad) * alpha_curr;
                        //vertex = vertex + grad * delta + normal* norm(grad)* alpha_curr;
                    }
                }
                Vertices[3 * s] = vertex.x;
                Vertices[3 * s + 1] = vertex.y;
                Vertices[3 * s + 2] = vertex.z;
            }
            for (int inner_iter = 0; inner_iter < innerloop_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
                for (int s = 0; s < Nb_Vertices; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
                    my_float3 nmle = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
                    for (int k = 0; k < 16; k++) {
                        if (RBF_id[16 * s + k] == -1)
                            break;
                        // Project points into tangent plane
                        my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);
                        float ps = dot(curr_v - vtx, nmle);
                        my_float3 v_proj = curr_v - nmle * ps;
                        x += v_proj.x * RBF[16 * s + k];
                        y += v_proj.y * RBF[16 * s + k];
                        z += v_proj.z * RBF[16 * s + k];
                        /*x += vtx_copy[3*RBF_id[16*s + k]]*RBF[16*s + k];
                        y += vtx_copy[3*RBF_id[16*s + k]+1]*RBF[16*s + k];
                        z += vtx_copy[3*RBF_id[16*s + k]+2]*RBF[16*s + k];*/
                    }
                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        Vertices[3 * s] = x;
                        Vertices[3 * s + 1] = y;
                        Vertices[3 * s + 2] = z;
                    }
                }
            }
            UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);
            alpha_curr = alpha_curr / 1.1f;
        }
        for (int t = 0; t < 100 * 100 * 100; t++)
            HashTable[t].clear();
        delete[] HashTable;
        delete[] RBF;
        delete[] RBF_id;
        delete[]vtx_copy;
    }
    
    void FitToLevelSet(float *sdf, float *Nodes, int *Tets, int nb_tets, float *Vertices, float *Normals, int *Faces, int Nb_Vertices, int Nb_Faces, int outerloop_maxiter = 20, int innerloop_maxiter = 10, float alpha = 0.01f, float delta = 0.003f, float delta_elastic = 0.01f ) {
               
        
        float *RBF = new float[16*Nb_Vertices];
        int *RBF_id = new int[16*Nb_Vertices];
        for (int n = 0; n < Nb_Vertices; n++) {
            for (int i = 0; i < 16; i++)
            RBF_id[16*n + i] = -1;
        }
        
        
        ComputeRBF(RBF, RBF_id, Vertices, Faces, Nb_Faces, Nb_Vertices);
        
        vector<int> *HashTable = GenerateHashTable(Nodes, Tets, nb_tets, make_my_int3(100,100,100), make_my_float3(0.0f,-0.1f,0.0f), 0.03f);
                
        // Copy of vertex position
        float *vtx_copy;
        vtx_copy = new float [3*Nb_Vertices];
        memcpy(vtx_copy, Vertices, 3*Nb_Vertices*sizeof(float));
                
        //############## Outer loop ##################
        float alpha_curr = alpha;
        for (int OuterIter = 0; OuterIter < outerloop_maxiter; OuterIter++) {
            cout << "outer iter" + to_string(OuterIter) << endl;
            // Copy current state
            memcpy(vtx_copy, Vertices, 3*Nb_Vertices*sizeof(float));
            
            // Increment deplacement of surface vertices
            for (int s = 0; s < Nb_Vertices; s++) {
                my_float3 vertex = make_my_float3(Vertices[3 * s], Vertices[3 * s + 1], Vertices[3 * s + 2]);
                my_float3 normal = make_my_float3(Normals[3 * s], Normals[3 * s + 1], Normals[3 * s + 2]);
                if (OuterIter < 3) {
                    vertex = vertex - normal * 0.01f;
                }
                else {
                    float sdf_val = InterpolateField(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                    my_float3 grad = Gradient(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                    if (sdf_val != sdf_val) {
                        vertex = vertex - normal * 0.001f;
                    }
                    else {
                        //    cout << "sdf: " << sdf_val << endl;
                        if (norm(grad) > 0.0f)
                            grad = grad * (sdf_val / norm(grad));

                        //grad.print();
                        if (dot(grad, normal) < 0.0f)
                            vertex = vertex - grad * delta + normal * norm(grad) * alpha_curr;
                        else
                            vertex = vertex - grad * delta - normal * norm(grad) * alpha_curr;
                    }
                }
                Vertices[3*s] = vertex.x;
                Vertices[3*s+1] = vertex.y;
                Vertices[3*s+2] = vertex.z;
            }
            
            for (int inner_iter = 0; inner_iter < innerloop_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3*Nb_Vertices*sizeof(float));
                for (int s = 0; s < Nb_Vertices; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    
                    my_float3 vtx = make_my_float3(vtx_copy[3*s], vtx_copy[3*s+1], vtx_copy[3*s+2]);
                    my_float3 nmle = make_my_float3(Normals[3*s], Normals[3*s+1], Normals[3*s+2]);
                    
                    for (int k = 0; k < 16; k++) {
                        if (RBF_id[16*s + k] == -1)
                            break;
                        
                        // Project points into tangent plane
                        my_float3 curr_v = make_my_float3(vtx_copy[3*RBF_id[16*s + k]], vtx_copy[3*RBF_id[16*s + k]+1], vtx_copy[3*RBF_id[16*s + k]+2]);
                        
                        float ps = dot(curr_v - vtx, nmle);
                        
                        my_float3 v_proj = curr_v - nmle * ps;
                        
                        x += v_proj.x*RBF[16*s + k];
                        y += v_proj.y*RBF[16*s + k];
                        z += v_proj.z*RBF[16*s + k];
                        /*x += vtx_copy[3*RBF_id[16*s + k]]*RBF[16*s + k];
                        y += vtx_copy[3*RBF_id[16*s + k]+1]*RBF[16*s + k];
                        z += vtx_copy[3*RBF_id[16*s + k]+2]*RBF[16*s + k];*/
                    }
                    
                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        Vertices[3*s] = x;
                        Vertices[3*s+1] = y;
                        Vertices[3*s+2] = z;
                    }
                }
                
            }
            
            UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);
            //alpha_curr = alpha_curr/1.5f;
        }
        
        for (int t = 0; t < 100*100*100; t++)
            HashTable[t].clear();
        delete[] HashTable;
        //delete[] RBF;
        //delete[] RBF_id;
        delete []vtx_copy;
    }

    float* LoopSurface_parallel(float* Vertices, float* Normals, int* RBF_id, float* RBF, int Nb_Vertices) {
        float* vertex_out = new float[3 * Nb_Vertices];
        std::vector< std::thread > my_threads;
        for (int i = 0; i < 101; i++) {
            //InnerLoopSurface(vertex_out, Vertices, Normals, RBF_id, RBF, Nb_Vertices, inner_iter * (Nb_Vertices / 100));
            my_threads.push_back(std::thread(InnerLoopSurface_multi, vertex_out, Vertices, Normals, RBF_id, RBF, Nb_Vertices, i * (Nb_Vertices / 100)));
        }
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
        return vertex_out;
    }

    /// <summary>
    /// Fit the 3D mesh to the tetra sdf field
    /// </summary>
    /// <param name="output_dir"></param>
    /// <param name="input_dir"></param>
    /// <param name="template_dir"></param>
    /// <param name="idx_file"></param>
    void FitToLevelSet(float* sdf, float* Nodes, int* Tets, float* RBF, int* RBF_id, int nb_tets, float* Vertices, float* Normals, int* Faces, int Nb_Vertices, int Nb_Faces, int outerloop_maxiter = 20, int innerloop_maxiter = 10, float alpha = 0.01f, float delta = 0.003f, float delta_elastic = 0.01f) {

        /*
        float *RBF = new float[16*Nb_Vertices];
        int *RBF_id = new int[16*Nb_Vertices];
        for (int n = 0; n < Nb_Vertices; n++) {
            for (int i = 0; i < 16; i++)
            RBF_id[16*n + i] = -1;
        }
        */

        ComputeRBF(RBF, RBF_id, Vertices, Faces, Nb_Faces, Nb_Vertices);
        vector<int>* HashTable = GenerateHashTable(Nodes, Tets, nb_tets, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);

        // Copy of vertex position
        float* vtx_copy;
        vtx_copy = new float[3 * Nb_Vertices];
        memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));

        //float* vtx_prev;
        //vtx_prev = new float[3 * Nb_Vertices];
        //memcpy(vtx_prev, Vertices, 3 * Nb_Vertices * sizeof(float));

        //############## Outer loop ##################
        float alpha_curr = alpha;
        for (int OuterIter = 4; OuterIter < outerloop_maxiter; OuterIter++) {
            cout << "outer iter" + to_string(OuterIter) << " " << Nb_Vertices << endl;
            // Copy current state
            //memcpy(vtx_prev, vtx_copy, 3 * Nb_Vertices * sizeof(float));
            memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
            
            OuterLoopMT(sdf, Nodes, Tets, HashTable, Faces, Vertices, vtx_copy, Normals, RBF_id, RBF, Nb_Faces, Nb_Vertices, OuterIter, delta, alpha_curr, 0.0f);

            //OuterLoopTot(sdf, Nodes, Tets, HashTable, Faces, Vertices, vtx_copy, Normals, RBF_id, RBF, Nb_Faces, Nb_Vertices, OuterIter, delta, alpha_curr, 0.0f);
            //OuterLoop(sdf, Nodes, Tets, HashTable, Vertices, vtx_copy, Normals, RBF_id, RBF, Nb_Vertices, OuterIter, delta, alpha_curr);
            
            for (int inner_iter = 0; inner_iter < innerloop_maxiter; inner_iter++) {
                //cout << "inner iter" + to_string(inner_iter) << endl;
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
                //Vertices = LoopSurface_parallel(vtx_copy, Normals, RBF_id, RBF, Nb_Vertices);
                InnerLoopSurface(Vertices, vtx_copy, Normals, RBF_id, RBF, Nb_Vertices);
                //InnerLoopSurface2(Vertices, vtx_copy, RBF_id, RBF, Nb_Vertices);
            }

            UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);
            //alpha_curr = alpha_curr/1.5f;

            //SaveMeshToPLY("D:/Human_data/4D_data/data/FittedMesh_ply_smpl/FittedTest_" + to_string(OuterIter) + ".ply", Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);

            if (OuterIter > 15) {
                innerloop_maxiter = 0;
                alpha_curr = 0.0f;
                delta = 0.01f;
            }
        }
        //UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);

        //memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
        //CleanMesh(Faces, Vertices, vtx_copy, Nb_Faces, Nb_Vertices);

        /*for (int f = 0; f < Nb_Faces; f++) {
            //compute ratio of edges length
            my_float3 s1 = make_my_float3(vtx_copy[3 * Faces[3 * f]], vtx_copy[3 * Faces[3 * f] + 1], vtx_copy[3 * Faces[3 * f] + 2]);
            my_float3 s2 = make_my_float3(vtx_copy[3 * Faces[3 * f + 1]], vtx_copy[3 * Faces[3 * f + 1] + 1], vtx_copy[3 * Faces[3 * f + 1] + 2]);
            my_float3 s3 = make_my_float3(vtx_copy[3 * Faces[3 * f + 2]], vtx_copy[3 * Faces[3 * f + 2] + 1], vtx_copy[3 * Faces[3 * f + 2] + 2]);

            float e_1 = norm(s1 - s2);
            float e_2 = norm(s1 - s3);
            float e_3 = norm(s3 - s2);

            if (e_1 > 0.04f || e_2 > 0.04f || e_3 > 0.04f) {
                Faces[3 * f] = 0;
                Faces[3 * f + 1] = 0;
                Faces[3 * f + 2] = 0;
            }
        }*/

        /*memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
        for (int f = 0; f < Nb_Faces; f++) {
            //compute ratio of edges length
            my_float3 s1 = make_my_float3(vtx_copy[3 * Faces[3 * f]], vtx_copy[3 * Faces[3 * f] + 1], vtx_copy[3 * Faces[3 * f] + 2]);
            my_float3 s2 = make_my_float3(vtx_copy[3 * Faces[3 * f + 1]], vtx_copy[3 * Faces[3 * f + 1] + 1], vtx_copy[3 * Faces[3 * f + 1] + 2]);
            my_float3 s3 = make_my_float3(vtx_copy[3 * Faces[3 * f + 2]], vtx_copy[3 * Faces[3 * f + 2] + 1], vtx_copy[3 * Faces[3 * f + 2] + 2]);

            float e_1 = norm(s1 - s2);
            float e_2 = norm(s1 - s3);
            float e_3 = norm(s3 - s2);

            if (e_1 > 0.02f || 
                e_2 > 0.02f ||
                e_3 > 0.02f) {
                Vertices[3 * Faces[3 * f]] = 0.0f;
                Vertices[3 * Faces[3 * f] + 1] = 0.0f;
                Vertices[3 * Faces[3 * f] + 2] = 0.0f;

                Vertices[3 * Faces[3 * f+1]] = 0.0f;
                Vertices[3 * Faces[3 * f+1] + 1] = 0.0f;
                Vertices[3 * Faces[3 * f+1] + 2] = 0.0f;

                Vertices[3 * Faces[3 * f+2]] = 0.0f;
                Vertices[3 * Faces[3 * f+2] + 1] = 0.0f;
                Vertices[3 * Faces[3 * f+2] + 2] = 0.0f;
            }            
        }

        memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
        for (int s = 0; s < Nb_Vertices; s++) {
            my_float3 vtx = make_my_float3(vtx_copy[3 * s], vtx_copy[3 * s + 1], vtx_copy[3 * s + 2]);
            if (vtx.x != 0.0f || vtx.y != 0.0f || vtx.z != 0.0f)
                continue;

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            float w = 0.0f;
            for (int k = 0; k < 16; k++) {
                if (RBF_id[16 * s + k] == -1)
                    break;

                if (vtx_copy[3 * RBF_id[16 * s + k]] == 0.0f && vtx_copy[3 * RBF_id[16 * s + k] + 1] == 0.0f &&
                    vtx_copy[3 * RBF_id[16 * s + k] + 2] == 0.0f)
                    continue;

                // Project points into tangent plane
                my_float3 curr_v = make_my_float3(vtx_copy[3 * RBF_id[16 * s + k]], vtx_copy[3 * RBF_id[16 * s + k] + 1], vtx_copy[3 * RBF_id[16 * s + k] + 2]);

                x += curr_v.x * RBF[16 * s + k];
                y += curr_v.y * RBF[16 * s + k];
                z += curr_v.z * RBF[16 * s + k];
                w += RBF[16 * s + k];
            }

            if (x != 0.0f && y != 0.0f && z != 0.0f) {
                Vertices[3 * s] = x/w;
                Vertices[3 * s + 1] = y/w;
                Vertices[3 * s + 2] = z/w;
            }
        }

        UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);*/

        for (int t = 0; t < 100 * 100 * 100; t++)
            HashTable[t].clear();
        delete[] HashTable;
        delete[]vtx_copy;
        //delete[]vtx_prev;
    }

    void FitVolumeToLevelSet(float* sdf, float* Nodes, int* Tets, float* RBF, int* RBF_id, int nb_tets, 
        float* Vertices, float* Normals, int* Faces, int* Surface, int *Inside, int Nb_Vertices, int Nb_Faces, int Nb_Surface, int Nb_Inside,
        int outerloop_maxiter = 20, int innerloop_surf_maxiter = 1, int innerloop_inner_maxiter = 10, float alpha = 0.01f, float delta = 0.003f, float delta_elastic = 0.01f) {

        vector<int>* HashTable = GenerateHashTable(Nodes, Tets, nb_tets, make_my_int3(100, 100, 100), 
            make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);

        // Copy of vertex position
        float* vtx_copy;
        vtx_copy = new float[3 * Nb_Vertices];
        memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));

        //############## Outer loop ##################
        float alpha_curr = alpha;
        float delta_curr = delta;
        for (int OuterIter = 0; OuterIter < outerloop_maxiter; OuterIter++) {
            cout << "outer iter" + to_string(OuterIter) << endl;
            //cout << "innerloop_inner_maxiter" << innerloop_inner_maxiter << endl;
            //cout << "innerloop_surf_maxiter" << innerloop_surf_maxiter << endl;

            // Copy current state
            memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));

            // Increment deplacement of surface vertices
            for (int s = 0; s < Nb_Surface; s++) {
                my_float3 vertex = make_my_float3(Vertices[3 * Surface[s]], Vertices[3 * Surface[s] + 1], Vertices[3 * Surface[s] + 2]);
                my_float3 normal = make_my_float3(Normals[3 * Surface[s]], Normals[3 * Surface[s] + 1], Normals[3 * Surface[s] + 2]);
                if (OuterIter < 3) {
                    vertex = vertex - normal * 0.001f;
                }
                else {
                    float sdf_val = InterpolateField(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                    my_float3 grad = Gradient(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100, 100, 100), make_my_float3(0.0f, -0.1f, 0.0f), 0.03f);
                    if (sdf_val != sdf_val) {
                        vertex = vertex - normal * 0.01f;
                    }
                    else {
                        //    cout << "sdf: " << sdf_val << endl;
                        if (norm(grad) > 0.0f)
                            grad = grad * (sdf_val / norm(grad));

                        //grad.print();
                        if (dot(grad, normal) < 0.0f)
                            vertex = vertex - grad * delta_curr + normal * norm(grad) * alpha_curr;
                        else
                            vertex = vertex - grad * delta_curr - normal * norm(grad) * alpha_curr;
                    }
                }
                Vertices[3 * Surface[s]] = vertex.x;
                Vertices[3 * Surface[s] + 1] = vertex.y;
                Vertices[3 * Surface[s] + 2] = vertex.z;
            }

            for (int inner_iter = 0; inner_iter < innerloop_surf_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
                for (int s = 0; s < Nb_Surface; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    for (int k = 0; k < 16; k++) {
                        if (RBF_id[16 * Surface[s] + k] == -1)
                            break;

                        //cout << _RBF_id[16*_Surface[s] + k] << endl;
                        x += vtx_copy[3 * RBF_id[16 * Surface[s] + k]] * RBF[16 * Surface[s] + k];
                        y += vtx_copy[3 * RBF_id[16 * Surface[s] + k] + 1] * RBF[16 * Surface[s] + k];
                        z += vtx_copy[3 * RBF_id[16 * Surface[s] + k] + 2] * RBF[16 * Surface[s] + k];
                    }

                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        Vertices[3 * Surface[s]] = x;
                        Vertices[3 * Surface[s] + 1] = y;
                        Vertices[3 * Surface[s] + 2] = z;
                    }
                }
            }

            for (int inner_iter = 0; inner_iter < innerloop_inner_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));

                for (int s = 0; s < Nb_Inside; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    for (int k = 0; k < 16; k++) {
                        if (RBF_id[16 * Inside[s] + k] == -1)
                            break;

                        //cout << _RBF_id[16*_Surface[s] + k] << endl;
                        x += vtx_copy[3 * RBF_id[16 * Inside[s] + k]] * RBF[16 * Inside[s] + k];
                        y += vtx_copy[3 * RBF_id[16 * Inside[s] + k] + 1] * RBF[16 * Inside[s] + k];
                        z += vtx_copy[3 * RBF_id[16 * Inside[s] + k] + 2] * RBF[16 * Inside[s] + k];
                    }

                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        Vertices[3 * Inside[s]] = x;
                        Vertices[3 * Inside[s] + 1] = y;
                        Vertices[3 * Inside[s] + 2] = z;
                    }
                }
            }

            for (int inner_iter = 0; inner_iter < innerloop_surf_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));

                for (int s = 0; s < Nb_Vertices; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    for (int k = 0; k < 16; k++) {
                        if (RBF_id[16 * s + k] == -1)
                            break;

                        x += vtx_copy[3 * RBF_id[16 * s + k]] * RBF[16 * s + k];
                        y += vtx_copy[3 * RBF_id[16 * s + k] + 1] * RBF[16 * s + k];
                        z += vtx_copy[3 * RBF_id[16 * s + k] + 2] * RBF[16 * s + k];
                    }

                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        Vertices[3 * s] = x;
                        Vertices[3 * s + 1] = y;
                        Vertices[3 * s + 2] = z;
                    }
                }
            }

            UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);
            //alpha_curr = alpha_curr/1.2f;
            //delta_curr = delta_curr / 1.1f;

            if (OuterIter > 15) {
                innerloop_inner_maxiter = 0;
                innerloop_surf_maxiter = 0;
            }
        }

        for (int t = 0; t < 100 * 100 * 100; t++)
            HashTable[t].clear();
        delete[] HashTable;
        //delete[] RBF;
        //delete[] RBF_id;
        delete[]vtx_copy;
    }

    /// <summary>
    /// Fit the 3D mesh to the tetra sdf field
    /// </summary>
    /// <param name="output_dir"></param>
    /// <param name="input_dir"></param>
    /// <param name="template_dir"></param>
    /// <param name="idx_file"></param>
    void FitToLevelSetCubic(float*** tsdf_grid, my_int3 dim_grid, my_float3 center, float res, float* RBF, int* RBF_id, float* Vertices, float* Normals, int* Faces, int Nb_Vertices, int Nb_Faces, int outerloop_maxiter = 20, int innerloop_maxiter = 10, float alpha = 0.01f, float delta = 0.003f, float delta_elastic = 0.01f) {

        ComputeRBF(RBF, RBF_id, Vertices, Faces, Nb_Faces, Nb_Vertices);

        // Compute gradient of tsdf
        my_float3*** grad_grid = VolumetricGrad(tsdf_grid, dim_grid);
        
        // Copy of vertex position
        float* vtx_copy;
        vtx_copy = new float[3 * Nb_Vertices];
        memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));

        //############## Outer loop ##################
        float alpha_curr = alpha;
        for (int OuterIter = 4; OuterIter < outerloop_maxiter; OuterIter++) {
            cout << "outer iter" + to_string(OuterIter) << " " << Nb_Vertices << endl;
            // Copy current state
            memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
            OuterLoopCubic(tsdf_grid, grad_grid, dim_grid, center, res, Vertices, Normals, Nb_Vertices, OuterIter, delta, alpha_curr, 0.0f);
            
            for (int inner_iter = 0; inner_iter < innerloop_maxiter; inner_iter++) {
                //cout << "inner iter" + to_string(inner_iter) << " " << alpha_curr << ", " << delta << endl;
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3 * Nb_Vertices * sizeof(float));
                InnerLoopSurface(Vertices, vtx_copy, Normals, RBF_id, RBF, Nb_Vertices);
            }

            UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);
            //alpha_curr = alpha_curr/1.5f;

            if (OuterIter > 15) {
                innerloop_maxiter = 0;
                alpha_curr = 0.0f;
                delta = 0.01f;
            }
        }

        for (int i = 1; i < dim_grid.x - 1; i++) {
            for (int j = 1; j < dim_grid.y - 1; j++) {
                delete[]grad_grid[i][j];
            }
            delete[]grad_grid[i];
        }
        delete[]grad_grid;

        delete[]vtx_copy;
    }


}

#endif /* Fit_h */
