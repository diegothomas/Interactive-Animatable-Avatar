//
//  LevelSet.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/14.
//

#ifndef LevelSet_h
#define LevelSet_h


void InnerLoopThin(float ***volume, int i, float *vertices, int *faces, float *normals, int nb_faces, my_int3 size_grid, my_float3 center_grid, float res, float disp) {
    my_float3 p0;
    p0.x = (float(i)-float(size_grid.x)/2.0f)*res + center_grid.x;
    
    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    //my_float3 ray = make_my_float3(1.0f, 1.0f, 1.0f)*(1.0f/sqrt(3.0f));
    
    for (int j = 0; j < size_grid.y; j++) {
        if (i == 0)
            cout << 100.0f*float(j)/float(size_grid.y) << "%\r";
        p0.y = (float(j)-float(size_grid.y)/2.0f)*res + center_grid.y;
        for (int k = 0; k < size_grid.z; k++) {
            //cout << i << ", " << j << ", " << k << endl;
            // Get the 3D coordinate
            p0.z = (float(k)-float(size_grid.z)/2.0f)*res + center_grid.z;
            
            // Compute the smallest distance to the faces
            float min_dist = 1.0e32f;
            float sdf = 1.0f;
            for (int f = 0; f < nb_faces; f++) {
                // Compute distance point to face
                my_float3 n = make_my_float3(normals[3*f], normals[3*f+1], normals[3*f+2]);
                my_float3 p1 = make_my_float3(vertices[3*faces[3*f]], vertices[3*faces[3*f]+1], vertices[3*faces[3*f]+2]);
                my_float3 p2 = make_my_float3(vertices[3*faces[3*f+1]], vertices[3*faces[3*f+1]+1], vertices[3*faces[3*f+1]+2]);
                my_float3 p3 = make_my_float3(vertices[3*faces[3*f+2]], vertices[3*faces[3*f+2]+1], vertices[3*faces[3*f+2]+2]);
                
                // Compute point to face distance
                float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
                if (curr_dist < min_dist) {
                    min_dist = curr_dist;
                    float sign = dot(p1-p0, n);
                    sdf = sign > 0.0 ? -curr_dist : curr_dist;
                }
            }
            
            volume[i][j][k] = (sdf-disp)/0.05f;
        }
    }
}


void InnerLoop(float ***volume, int i, float *vertices, int *faces, float *normals, int nb_faces, my_int3 size_grid, my_float3 center_grid, float res, float disp) {
    my_float3 p0;
    p0.x = (float(i)-float(size_grid.x)/2.0f)*res + center_grid.x;
    
    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    //my_float3 ray = make_my_float3(1.0f, 1.0f, 1.0f)*(1.0f/sqrt(3.0f));
    
    for (int j = 0; j < size_grid.y; j++) {
        if (i == 0) 
            cout << 100.0f*float(j)/float(size_grid.y) << "%\r";
        p0.y = (float(j)-float(size_grid.y)/2.0f)*res + center_grid.y;
        for (int k = 0; k < size_grid.z; k++) {
            //cout << i << ", " << j << ", " << k << endl;
            // Get the 3D coordinate
            p0.z = (float(k)-float(size_grid.z)/2.0f)*res + center_grid.z;
            
            // Compute the smallest distance to the faces
            float min_dist = 1.0e32f;
            float sdf = 1.0f;
            float intersections = 0.0f;
            for (int f = 0; f < nb_faces; f++) {
                // Compute distance point to face
                my_float3 n = make_my_float3(normals[3*f], normals[3*f+1], normals[3*f+2]);
                my_float3 p1 = make_my_float3(vertices[3*faces[3*f]], vertices[3*faces[3*f]+1], vertices[3*faces[3*f]+2]);
                my_float3 p2 = make_my_float3(vertices[3*faces[3*f+1]], vertices[3*faces[3*f+1]+1], vertices[3*faces[3*f+1]+2]);
                my_float3 p3 = make_my_float3(vertices[3*faces[3*f+2]], vertices[3*faces[3*f+2]+1], vertices[3*faces[3*f+2]+2]);
                            
                // Compute line plane intersection
                //if (IsInterectingRayTriangle3D(ray, p0, p1, p2, p3, n))
                //    intersections++;
                intersections += IsInterectingRayTriangle3D(ray, p0, p1, p2, p3, n);
                
                // Compute point to face distance
                float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
                if (curr_dist < min_dist) {
                    min_dist = curr_dist;
                    sdf = curr_dist;///0.4f;
                }
            }
            
            //cout << int(intersections) <<  endl;
            //volume[i][j][k] = intersections % 2 == 0 ? sdf : -sdf;
            volume[i][j][k] = int(intersections) % 2 == 0 ? sdf-disp : -sdf-disp;
        }
    }
}

void InnerLoopSDFFromLevelSet(float ***volume, int i, float ***sdf, my_int3 size_grid, my_float3 center_grid, float res) {
    my_float3 p0;
    p0.x = (float(i)-float(size_grid.x)/2.0f)*res + center_grid.x;
    
    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    //my_float3 ray = make_my_float3(1.0f, 1.0f, 1.0f)*(1.0f/sqrt(3.0f));
    
    for (int j = 0; j < size_grid.y; j++) {
        if (i == 0)
            cout << 100.0f*float(j)/float(size_grid.y) << "%\r";
        p0.y = (float(j)-float(size_grid.y)/2.0f)*res + center_grid.y;
        for (int k = 0; k < size_grid.z; k++) {
            // Get the 3D coordinate
            p0.z = (float(k)-float(size_grid.z)/2.0f)*res + center_grid.z;
            
            float sdf_val = sdf[i][j][k];
            float sdf_curr = sdf_val;
            
            int s = 1;
            while (sdf_val*sdf_curr > 0.0f && s < size_grid.x+1) {
                for (int a = max(0,i-s); a < min(size_grid.x,i+s+1); a++) {
                    for (int b = max(0,j-s); b < min(size_grid.y,j+s+1); b++) {
                        if (k-s >= 0) {
                            sdf_curr = sdf[a][b][k-s];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                        
                        if (k+s < size_grid.z) {
                            sdf_curr = sdf[a][b][k+s];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                    }
                }
                
                if (sdf_val*sdf_curr <= 0.0f)
                    break;
                
                
                for (int a = max(0,i-s); a < min(size_grid.x,i+s+1); a++) {
                    for (int c = max(0,k-s); c < min(size_grid.z,k+s+1); c++) {
                        if (j-s >= 0) {
                            sdf_curr = sdf[a][j-s][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                        
                        if (j+s < size_grid.y) {
                            sdf_curr = sdf[a][j+s][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                    }
                }
                
                if (sdf_val*sdf_curr <= 0.0f)
                    break;
                
                
                for (int c = max(0,k-s); c < min(size_grid.z,k+s+1); c++) {
                    for (int b = max(0,j-s); b < min(size_grid.y,j+s+1); b++) {
                        if (i-s >= 0) {
                            sdf_curr = sdf[i-s][b][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                        
                        if (i+s < size_grid.x) {
                            sdf_curr = sdf[i+s][b][c];
                            if (sdf_val*sdf_curr <= 0.0f)
                                break;
                        }
                    }
                }
                
                if (sdf_val*sdf_curr <= 0.0f)
                    break;
                
                s++;
            }
            
            volume[i][j][k] = sdf_val < 0.0f ? -float(s)*res : float(s)*res;
        }
    }
}

/**
 This function creates a volumetric level set from a mesh with define resolution (=size of 1 voxel)
 */
float ***LevelSet(float *vertices, int *faces, float *normals, int nb_faces, my_int3 size_grid, my_float3 center_grid, float res, float disp = 0.0f) {
    // Allocate data
    float ***volume = new float **[size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        volume[i] = new float *[size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume[i][j] = new float[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume[i][j][k] = 1.0f;
            }
        }
    }
    
    // Compute the signed distance to the mesh for each voxel
    std::vector< std::thread > my_threads;
    for (int i = 0; i < size_grid.x; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back( std::thread(InnerLoop, volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return volume;
}

/**
 This function creates a volumetric level set of a sphere
 */
float ***LevelSet_sphere(my_float3 center, float radius, int *size_grid, float res) {
    // Allocate data
    float ***volume = new float **[size_grid[0]];
    for (int i = 0; i < size_grid[0]; i++) {
        volume[i] = new float *[size_grid[1]];
        for (int j = 0; j < size_grid[1]; j++) {
            volume[i][j] = new float[size_grid[2]];
        }
    }
    
    // Compute the signed distance to the mesh for each voxel
    my_float3 p0;
    for (int i = 0; i < size_grid[0]; i++) {
        p0.x = float(i)*res-float(size_grid[0])*res/2.0f;
        for (int j = 0; j < size_grid[1]; j++) {
            if (i == 0)
                cout << 100.0f*float(j)/float(size_grid[1]) << "%\r";
            p0.y = float(j)*res-float(size_grid[1])*res/2.0f;
            for (int k = 0; k < size_grid[2]; k++) {
                // Get the 3D coordinate
                p0.z = float(k)*res-float(size_grid[2])*res/2.0f;
                
                // Compute distance point to sphere
                volume[i][j][k] = (dot(p0-center, p0-center) - radius*radius);
            }
        }
    }
    
    return volume;
}

void TetInnerLoop(float *volume, float *volume_weights, int i, float *vertices, float *skin_weights, int *faces, float *normals, int nb_nodes, int nb_faces, float *nodes) {

    my_float3 ray = make_my_float3(0.0f, 0.0f, 1.0f);
    
    for (int idx = i; idx < i + nb_nodes/100; idx++) {
        if (idx >= nb_nodes)
            break;
        if (i == 0)
            cout << 100.0f*float(idx-i)/float(nb_nodes/100) << "%\r";
        
        my_float3 p0 = make_my_float3(nodes[3*idx], nodes[3*idx+1], nodes[3*idx+2]);
        
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        float sdf = 1.0f;
        float *s_w = new float[24];
        int intersections = 0;
        for (int f = 0; f < nb_faces; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals[3*f], normals[3*f+1], normals[3*f+2]);
            my_float3 p1 = make_my_float3(vertices[3*faces[3*f]], vertices[3*faces[3*f]+1], vertices[3*faces[3*f]+2]);
            my_float3 p2 = make_my_float3(vertices[3*faces[3*f+1]], vertices[3*faces[3*f+1]+1], vertices[3*faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(vertices[3*faces[3*f+2]], vertices[3*faces[3*f+2]+1], vertices[3*faces[3*f+2]+2]);
                        
            // Compute line plane intersection
            if (IsInterectingRayTriangle3D(ray, p0, p1, p2, p3, n))
                intersections++;
            
            my_float3 center = (p1 + p2 + p3) * (1.0f/3.0f);
            
            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n, false);
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //sdf = curr_dist/THRESH;
                sdf = min(1.0f, curr_dist/THRESH);
                // Compute skin weights
                //for (int k = 0; k < 24; k++)
                //    s_w[k] = skin_weights[24*faces[3*f] + k];
                // <===== For skin weights
                /*SkinWeightsFromFace3D(s_w, &skin_weights[24*faces[3*f]], &skin_weights[24*faces[3*f+1]], &skin_weights[24*faces[3*f+2]], p0, p1, p2, p3, n);*/
            }
        }
    
        volume[idx] = intersections % 2 == 0 ? sdf : -sdf;
        /*for (int k = 0; k < 24; k++)
            volume_weights[24*idx+k] = s_w[k];*/
        
        delete[] s_w;
        
    }
}
/**
 This function creates a tet volumetric level set from a mesh
 */
float *TetLevelSet(float *volume_weights, float *vertices, float *skin_weights, int *faces, float *normals, int nb_faces, float *nodes, int nb_nodes) {
    // Allocate data
    float *volume = new float [nb_nodes];
    for (int i = 0; i < nb_nodes; i++) {
            volume[i] = 1.0f;
    }
    
    // Compute the signed distance to the mesh for each voxel
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        my_threads.push_back( std::thread(TetInnerLoop, volume, volume_weights, i*(nb_nodes/100), vertices, skin_weights, faces, normals, nb_nodes, nb_faces, nodes) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    /*for (int i = 0; i < nb_nodes; i++) {
        if (i % 100 == 0)
            cout << 100.0f*float(i)/float(nb_nodes) << "%\r";
        TetInnerLoop(volume, volume_weights, i, vertices, skin_weights, faces, normals, nb_nodes, nb_faces, nodes);
    }*/
    
    return volume;
}

/**
 This function creates a volumetric level set of a sphere
 */
float *TetLevelSet_sphere(my_float3 center, float radius, float *nodes, int nb_nodes) {
    // Allocate data
    float *volume = new float [nb_nodes];
    for (int i = 0; i < nb_nodes; i++) {
            volume[i] = 1.0f;
    }
    
    // Compute the signed distance to the mesh for each voxel
    my_float3 p0;
    for (int i = 0; i < nb_nodes; i++) {
        // Get the 3D coordinate
        p0 = make_my_float3(nodes[3*i], nodes[3*i+1], nodes[3*i+2]);
        
        // Compute distance point to sphere
        volume[i] = (dot(p0-center, p0-center) - radius*radius);
    }
    
    return volume;
}


/**
 This function recompute the level set from the current 0 crossing
 */
float ***SDFFromLevelSet(float ***sdf, my_int3 size_grid, my_float3 center_grid, float res) {
    // Allocate data
    float ***volume = new float **[size_grid.x];
    for (int i = 0; i < size_grid.x; i++) {
        volume[i] = new float *[size_grid.y];
        for (int j = 0; j < size_grid.y; j++) {
            volume[i][j] = new float[size_grid.z];
            for (int k = 0; k < size_grid.z; k++) {
                volume[i][j][k] = 1.0f;
            }
        }
    }
    
    // Compute the signed distance to the mesh for each voxel
    std::vector< std::thread > my_threads;
    for (int i = 0; i < size_grid.x; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back( std::thread(InnerLoopSDFFromLevelSet, volume, i, sdf, size_grid, center_grid, res) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return volume;
}


#endif /* LevelSet_h */
