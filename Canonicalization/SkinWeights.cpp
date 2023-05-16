#include "include/Utilities.h"
#include "include/MeshUtils.h"

using namespace std;


void InnerLoop2(float *skin_weights, float *vertices_s, int Nb_Vertices_s, float *vertices_m, int Nb_Vertices_m, float *result, int id) {
    
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s/100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3*i], vertices_s[3*i+1], vertices_s[3*i+2]);
        // Compute the smallest distance to the points
        float min_dist = 1.0e32f;
        int closest_point = 0;
        for (int j = 0; j < Nb_Vertices_m; j++) {
            // Compute distance point to point
            my_float3 p1 = make_my_float3(vertices_m[3*j], vertices_m[3*j+1], vertices_m[3*j+2]);
            
            // Compute point to face distance
            float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                closest_point = j;
            }
        }
        
        for (int k = 0; k < 24; k++) {
            //cout << k << "  " << skin_weights[closest_point*24 + k] << endl;
            result[i*24 + k] = skin_weights[closest_point*24 + k];
        }
    }
}

float *GetCorrespondences2(float *skin_weights, float *vertices_s, int Nb_Vertices_s, float *vertices_m, int Nb_Vertices_m) {
    float *result = new float[24*Nb_Vertices_s];
    
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        InnerLoop2(skin_weights, vertices_s, Nb_Vertices_s, vertices_m, Nb_Vertices_m, result, i*(Nb_Vertices_s/100));
        //my_threads.push_back( std::thread(InnerLoop, skin_weights, vertices_s, Nb_Vertices_s, vertices_m, Nb_Vertices_m, result, i*(Nb_Vertices_s/100)) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return result;
}

void InnerLoop(float *skin_weights, float *vertices_s, float *normals_s, int Nb_Vertices_s, float *vertices_m, float *normals_faces_m, int *faces_m, int Nb_Faces_m, float *result, int id) {
    
    float *corr = new float[24];
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s/100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3*i], vertices_s[3*i+1], vertices_s[3*i+2]);
        my_float3 ray = make_my_float3(normals_s[3*i], normals_s[3*i+1], normals_s[3*i+2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        int closest_point = 0;
        for (int f = 0; f < Nb_Faces_m; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_faces_m[3*f], normals_faces_m[3*f+1], normals_faces_m[3*f+2]);
            //if (dot(n,ray) < 0.2f)
            //   continue;
                
            my_float3 p1 = make_my_float3(vertices_m[3*faces_m[3*f]], vertices_m[3*faces_m[3*f]+1], vertices_m[3*faces_m[3*f]+2]);
            my_float3 p2 = make_my_float3(vertices_m[3*faces_m[3*f+1]], vertices_m[3*faces_m[3*f+1]+1], vertices_m[3*faces_m[3*f+1]+2]);
            my_float3 p3 = make_my_float3(vertices_m[3*faces_m[3*f+2]], vertices_m[3*faces_m[3*f+2]+1], vertices_m[3*faces_m[3*f+2]+2]);
                        
            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //closest_point = faces_m[3*f];
                SkinWeightsFromFace3D(corr, &skin_weights[24*faces_m[3*f]], &skin_weights[24*faces_m[3*f+1]], &skin_weights[24*faces_m[3*f+2]], p0, p1, p2, p3, n);
            }
        }
        
        for (int k = 0; k < 24; k++) {
            //cout << k << "  " << skin_weights[closest_point*24 + k] << endl;
            //result[i*24 + k] = skin_weights[closest_point*24 + k];
            result[i*24 + k] = corr[k];
        }
    }
    delete[] corr;
}

float *GetCorrespondences(float *skin_weights, float *vertices_s, float *normals_s, int Nb_Vertices_s, float *vertices_m, float *normals_faces_m, int *faces_m, int Nb_Faces_m) {
    float *result = new float[24*Nb_Vertices_s];
    
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back( std::thread(InnerLoop, skin_weights, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, Nb_Faces_m, result, i*(Nb_Vertices_s/100)) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return result;
}


void Align(float* vertices_s, int Nb_Vertices_s, float* trans) {
    for (int i = 0; i < Nb_Vertices_s; i++) {
        vertices_s[3 * i] = vertices_s[3 * i] + trans[0];
        vertices_s[3 * i + 1] = vertices_s[3 * i + 1] + trans[1];
        vertices_s[3 * i + 2] = vertices_s[3 * i + 2] + trans[2];
    }
}

float *Align( float *vertices_s, int Nb_Vertices_s, float *vertices_m_f, int Nb_Vertices_m_f, float *vertices_m, int Nb_Vertices_m) {
    my_float3 avg_s = make_my_float3(0.0f,0.0f,0.0f);
    for (int i = 0; i < Nb_Vertices_m_f; i++) {
        avg_s = avg_s + make_my_float3(vertices_m_f[3*i], vertices_m_f[3*i+1], vertices_m_f[3*i+2]);
    }
    avg_s = avg_s * (1.0f/float(Nb_Vertices_m_f));
    
    my_float3 avg_m = make_my_float3(0.0f,0.0f,0.0f);
    for (int i = 0; i < Nb_Vertices_m; i++) {
        avg_m = avg_m + make_my_float3(vertices_m[3*i], vertices_m[3*i+1], vertices_m[3*i+2]);
    }
    avg_m = avg_m * (1.0f/float(Nb_Vertices_m));
    
    my_float3 trans = avg_m - avg_s;
    
    for (int i = 0; i < Nb_Vertices_s; i++) {
        vertices_s[3*i] = vertices_s[3*i] + trans.x;
        vertices_s[3*i+1] = vertices_s[3*i+1] + trans.y;
        vertices_s[3*i+2] = vertices_s[3*i+2] + trans.z;
    }
    
    /*for (int i = 0; i < Nb_Vertices_m_f; i++) {
        vertices_m_f[3*i] = vertices_m_f[3*i] + trans.x;
        vertices_m_f[3*i+1] = vertices_m_f[3*i+1] + trans.y;
        vertices_m_f[3*i+2] = vertices_m_f[3*i+2] + trans.z;
    }*/

    float* res = new float[3];
    res[0] = trans.x; res[1] = trans.y; res[2] = trans.z;
    return res;
}

void CleanSkinningWeights(float *weights, float *vertices_scan, int *faces_scan, int Nb_Vertices, int Nb_Faces, float thresh = 1.0e-1) {
    
    int *Counter = new int[Nb_Vertices];
    for (int v = 0; v < Nb_Vertices; v++)
        Counter[v] = -1;
        
    // Delete erroneous skinning weights for faces with too long edges
    for (int f = 0; f < Nb_Faces; f++) {
        my_float3 v1 = make_my_float3(vertices_scan[3*faces_scan[3*f]], vertices_scan[3*faces_scan[3*f]+1], vertices_scan[3*faces_scan[3*f]+2]);
        my_float3 v2 = make_my_float3(vertices_scan[3*faces_scan[3*f+1]], vertices_scan[3*faces_scan[3*f+1]+1], vertices_scan[3*faces_scan[3*f+1]+2]);
        my_float3 v3 = make_my_float3(vertices_scan[3*faces_scan[3*f+2]], vertices_scan[3*faces_scan[3*f+2]+1], vertices_scan[3*faces_scan[3*f+2]+2]);
                
        float dist1 = norm(v1 - v2);
        float dist2 = norm(v2 - v3);
        float dist3 = norm(v1 - v3);
        
        if (dist1 > thresh) {
            for (int k = 0; k < 24; k++) {
                weights[24*faces_scan[3*f] + k] = 0.0f;
                weights[24*faces_scan[3*f+1] + k] = 0.0f;
            }
            Counter[faces_scan[3*f]] = 0;
            Counter[faces_scan[3*f+1]] = 0;
        }
        
        if (dist2 > thresh) {
            for (int k = 0; k < 24; k++) {
                weights[24*faces_scan[3*f+1] + k] = 0.0f;
                weights[24*faces_scan[3*f+2] + k] = 0.0f;
            }
            Counter[faces_scan[3*f+1]] = 0;
            Counter[faces_scan[3*f+2]] = 0;
        }
        
        if (dist3 > thresh) {
            for (int k = 0; k < 24; k++) {
                weights[24*faces_scan[3*f] + k] = 0.0f;
                weights[24*faces_scan[3*f+2] + k] = 0.0f;
            }
            Counter[faces_scan[3*f]] = 0;
            Counter[faces_scan[3*f+2]] = 0;
        }
    }
    
    int res = 0;
    for (int v = 0; v < Nb_Vertices; v++)
        res = Counter[v] != -1? res+1: res;
    cout << "remaining vertices: " << res << endl;

    
    int iter = 0;
    while(res > 0 && iter < 4) {
        iter++;
        
        // Do interpolation for unassigned skinning weights
        for (int f = 0; f < Nb_Faces; f++) {
            
            if (Counter[faces_scan[3*f]] != -1) {
                //if (Counter[faces_scan[3*f]] == 0)
                //    cout << "Counter: " << Counter[faces_scan[3*f+1]] << ", " << Counter[faces_scan[3*f+2]] << endl;
                if (Counter[faces_scan[3*f+1]] != 0) {
                    for (int k = 0; k < 24; k++)
                        weights[24*faces_scan[3*f] + k] =  (float(Counter[faces_scan[3*f]])*weights[24*faces_scan[3*f] + k] +  weights[24*faces_scan[3*f+1] + k]) / (float(Counter[faces_scan[3*f]] + 1));
                    Counter[faces_scan[3*f]] = Counter[faces_scan[3*f]]  + 1;
                }
                
                if (Counter[faces_scan[3*f+2]] != 0) {
                    for (int k = 0; k < 24; k++)
                        weights[24*faces_scan[3*f] + k] =  (float(Counter[faces_scan[3*f]])*weights[24*faces_scan[3*f] + k] +  weights[24*faces_scan[3*f+2] + k]) / (float(Counter[faces_scan[3*f]] + 1));
                    Counter[faces_scan[3*f]] = Counter[faces_scan[3*f]]  + 1;
                }
            }
            
            if (Counter[faces_scan[3*f+1]] != -1) {
                //if (Counter[faces_scan[3*f+1]] == 0)
                //    cout << "Counter: " << Counter[faces_scan[3*f]] << ", " << Counter[faces_scan[3*f+2]] << endl;
                if (Counter[faces_scan[3*f]]  != 0) {
                    for (int k = 0; k < 24; k++)
                        weights[24*faces_scan[3*f+1] + k] =  (float(Counter[faces_scan[3*f+1]])*weights[24*faces_scan[3*f+1] + k] +  weights[24*faces_scan[3*f] + k]) / (float(Counter[faces_scan[3*f+1]] + 1));
                    Counter[faces_scan[3*f+1]] = Counter[faces_scan[3*f+1]]  + 1;
                }
                
                if (Counter[faces_scan[3*f+2]]  != 0) {
                    for (int k = 0; k < 24; k++)
                        weights[24*faces_scan[3*f+1] + k] =  (float(Counter[faces_scan[3*f+1]])*weights[24*faces_scan[3*f+1] + k] +  weights[24*faces_scan[3*f+2] + k]) / (float(Counter[faces_scan[3*f+1]] + 1));
                    Counter[faces_scan[3*f+1]] = Counter[faces_scan[3*f+1]]  + 1;
                }
            }
            
            if (Counter[faces_scan[3*f+2]] != -1) {
                //if (Counter[faces_scan[3*f+2]] == 0)
                //    cout << "Counter: " << Counter[faces_scan[3*f]] << ", " << Counter[faces_scan[3*f+1]] << endl;
                if (Counter[faces_scan[3*f]]  != 0) {
                    for (int k = 0; k < 24; k++)
                        weights[24*faces_scan[3*f+2] + k] =  (float(Counter[faces_scan[3*f+2]])*weights[24*faces_scan[3*f+2] + k] +  weights[24*faces_scan[3*f] + k]) / (float(Counter[faces_scan[3*f+2]] + 1));
                    Counter[faces_scan[3*f+2]] = Counter[faces_scan[3*f+2]]  + 1;
                }
                
                if (Counter[faces_scan[3*f+1]]  != 0) {
                    for (int k = 0; k < 24; k++)
                        weights[24*faces_scan[3*f+2] + k] =  (float(Counter[faces_scan[3*f+2]])*weights[24*faces_scan[3*f+2] + k] +  weights[24*faces_scan[3*f+1] + k]) / (float(Counter[faces_scan[3*f+2]] + 1));
                    Counter[faces_scan[3*f+2]] = Counter[faces_scan[3*f+2]]  + 1;
                }
            }
            
        }
        
        res = 0;
        int counrem = 0;
        for (int v = 0; v < Nb_Vertices; v++) {
            if (Counter[v] > 0) {
                float sum_w = 0.0f;
                for (int k = 0; k < 24; k++) {
                    //weights[24*v + k] = weights[24*v + k] / float(Counter[v]);
                    sum_w += weights[24*v + k];
                }
                
                //cout << "weights " << v << ": ";
                for (int k = 0; k < 24; k++) {
                    weights[24*v + k] = weights[24*v + k] / sum_w;
                    //cout << weights[24*v + k] << ", ";
                }
                //cout << "<<<<>>>>" << endl;
                Counter[v] = 1;
            }
            if (Counter[v] == 0)
                res++;
            
            bool valid = false;
            for (int k = 0; k < 24; k++) {
                if (weights[24*v + k] > 0.0f)
                    valid = true;
            }
            if (!valid)
                counrem ++;
        }
        cout << "remaining vertices: " << res << endl;
    }
    
    delete[] Counter;
    
}

void CleanMesh(float *vertices_scan, float *normals_scan, int *faces_scan, int Nb_Vertices, int Nb_Faces, float thresh = 1.0e-1, float thresh_n = 0.3f) {
    
    bool *Valid = new bool[Nb_Vertices];
    int *Counter = new int[Nb_Vertices];
    for (int v = 0; v < Nb_Vertices; v++) {
        Counter[v] = -1;
        Valid[v] = true;
    }
    
    float *vertices_scan_s = new float[3*Nb_Vertices];
    memcpy(vertices_scan_s, vertices_scan, 3 * Nb_Vertices * sizeof(float));
        
    // Delete erroneous skinning weights for faces with too long edges
    for (int f = 0; f < Nb_Faces; f++) {
        my_float3 v1 = make_my_float3(vertices_scan_s[3*faces_scan[3*f]], vertices_scan_s[3*faces_scan[3*f]+1], vertices_scan_s[3*faces_scan[3*f]+2]);
        my_float3 v2 = make_my_float3(vertices_scan_s[3*faces_scan[3*f+1]], vertices_scan_s[3*faces_scan[3*f+1]+1], vertices_scan_s[3*faces_scan[3*f+1]+2]);
        my_float3 v3 = make_my_float3(vertices_scan_s[3*faces_scan[3*f+2]], vertices_scan_s[3*faces_scan[3*f+2]+1], vertices_scan_s[3*faces_scan[3*f+2]+2]);
        
        my_float3 n1 = make_my_float3(normals_scan[3*faces_scan[3*f]], normals_scan[3*faces_scan[3*f]+1], vertices_scan[3*faces_scan[3*f]+2]);
        my_float3 n2 = make_my_float3(normals_scan[3*faces_scan[3*f+1]], normals_scan[3*faces_scan[3*f+1]+1], vertices_scan[3*faces_scan[3*f+1]+2]);
        my_float3 n3 = make_my_float3(normals_scan[3*faces_scan[3*f+2]], normals_scan[3*faces_scan[3*f+2]+1], vertices_scan[3*faces_scan[3*f+2]+2]);
            
                
        float dist1 = norm(v1 - v2);
        float dist2 = norm(v2 - v3);
        float dist3 = norm(v1 - v3);
        
        float distn1 = dot(n1,n2);
        float distn2 = dot(n2,n3);
        float distn3 = dot(n1,n3);
        
        if (dist1 > thresh /*&& distn1 < thresh_n*/) {
            for (int k = 0; k < 3; k++) {
                faces_scan[3*f] = 0;
                faces_scan[3*f+1] = 0;
                faces_scan[3*f+2] = 0;
                /*vertices_scan[3*faces_scan[3*f]+k] = 0.0f;
                vertices_scan[3*faces_scan[3*f+1]+k] = 0.0f;
                vertices_scan[3*faces_scan[3*f+2]+k] = 0.0f;*/
            }
            Counter[faces_scan[3*f]] = 0;
            Counter[faces_scan[3*f+1]] = 0;
            Valid[faces_scan[3*f]] = false;
            Valid[faces_scan[3*f+1]] = false;
        }
        
        if (dist2 > thresh /*&& distn2 < thresh_n*/) {
            for (int k = 0; k < 3; k++) {
                faces_scan[3*f] = 0;
                faces_scan[3*f+1] = 0;
                faces_scan[3*f+2] = 0;
                /*vertices_scan[3*faces_scan[3*f]+k] = 0.0f;
                vertices_scan[3*faces_scan[3*f+1]+k] = 0.0f;
                vertices_scan[3*faces_scan[3*f+2]+k] = 0.0f;*/
            }
            Counter[faces_scan[3*f+1]] = 0;
            Counter[faces_scan[3*f+2]] = 0;
            Valid[faces_scan[3*f+1]] = false;
            Valid[faces_scan[3*f+2]] = false;
        }
        
        if (dist3 > thresh /*&& distn3 < thresh_n*/) {
            for (int k = 0; k < 3; k++) {
                faces_scan[3*f] = 0;
                faces_scan[3*f+1] = 0;
                faces_scan[3*f+2] = 0;
                /*vertices_scan[3*faces_scan[3*f]+k] = 0.0f;
                vertices_scan[3*faces_scan[3*f+1]+k] = 0.0f;
                vertices_scan[3*faces_scan[3*f+2]+k] = 0.0f;*/
            }
            Counter[faces_scan[3*f]] = 0;
            Counter[faces_scan[3*f+2]] = 0;
            Valid[faces_scan[3*f]] = false;
            Valid[faces_scan[3*f+2]] = false;
        }
    }
    
    
    /*int res = 0;
    for (int v = 0; v < Nb_Vertices; v++)
        res = Counter[v] != -1? res+1: res;
    cout << "remaining vertices: " << res << endl;
    
    memcpy(vertices_scan_s, vertices_scan, 3 * Nb_Vertices * sizeof(float));
    
    int iter = 0;
    while(res > 0 && iter < 4) {
        iter++;
        
        // Do interpolation for unassigned skinning weights
        for (int f = 0; f < Nb_Faces; f++) {
            
            if (Counter[faces_scan[3*f]] != -1) {
                //if (Counter[faces_scan[3*f]] == 0)
                //    cout << "Counter: " << Counter[faces_scan[3*f+1]] << ", " << Counter[faces_scan[3*f+2]] << endl;
                if (Valid[faces_scan[3*f+1]]) {
                    for (int k = 0; k < 3; k++)
                        vertices_scan[3*faces_scan[3*f] + k] = vertices_scan[3*faces_scan[3*f] + k] +  vertices_scan_s[3*faces_scan[3*f+1] + k];
                    Counter[faces_scan[3*f]] = Counter[faces_scan[3*f]]  + 1;
                }
                
                if (Valid[faces_scan[3*f+2]]) {
                    for (int k = 0; k < 3; k++)
                        vertices_scan[3*faces_scan[3*f] + k] = vertices_scan[3*faces_scan[3*f] + k] +  vertices_scan_s[3*faces_scan[3*f+2] + k];
                    Counter[faces_scan[3*f]] = Counter[faces_scan[3*f]]  + 1;
                }
            }
            
            if (Counter[faces_scan[3*f+1]] != -1) {
                //if (Counter[faces_scan[3*f+1]] == 0)
                //    cout << "Counter: " << Counter[faces_scan[3*f]] << ", " << Counter[faces_scan[3*f+2]] << endl;
                if (Valid[faces_scan[3*f]]) {
                    for (int k = 0; k < 3; k++)
                        vertices_scan[3*faces_scan[3*f+1] + k] = vertices_scan[3*faces_scan[3*f+1] + k] +  vertices_scan_s[3*faces_scan[3*f] + k];
                    Counter[faces_scan[3*f+1]] = Counter[faces_scan[3*f+1]]  + 1;
                }
                
                if (Valid[faces_scan[3*f+2]]) {
                    for (int k = 0; k < 3; k++)
                        vertices_scan[3*faces_scan[3*f+1] + k] = vertices_scan[3*faces_scan[3*f+1] + k] +  vertices_scan_s[3*faces_scan[3*f+2] + k];
                    Counter[faces_scan[3*f+1]] = Counter[faces_scan[3*f+1]]  + 1;
                }
            }
            
            if (Counter[faces_scan[3*f+2]] != -1) {
                //if (Counter[faces_scan[3*f+2]] == 0)
                //    cout << "Counter: " << Counter[faces_scan[3*f]] << ", " << Counter[faces_scan[3*f+1]] << endl;
                if (Valid[faces_scan[3*f]]) {
                    for (int k = 0; k < 3; k++)
                        vertices_scan[3*faces_scan[3*f+2] + k] = vertices_scan[3*faces_scan[3*f+2] + k] +  vertices_scan_s[3*faces_scan[3*f] + k];
                    Counter[faces_scan[3*f+2]] = Counter[faces_scan[3*f+2]]  + 1;
                }
                
                if (Valid[faces_scan[3*f+1]]) {
                    for (int k = 0; k < 3; k++)
                        vertices_scan[3*faces_scan[3*f+2] + k] = vertices_scan[3*faces_scan[3*f+2] + k] +  vertices_scan_s[3*faces_scan[3*f+1] + k];
                    Counter[faces_scan[3*f+2]] = Counter[faces_scan[3*f+2]]  + 1;
                }
            }
            
        }
        
        res = 0;
        for (int v = 0; v < Nb_Vertices; v++) {
            if (Counter[v] > 0) {
                for (int k = 0; k < 3; k++) {
                    vertices_scan_s[3*v + k] = vertices_scan[3*v + k] / float(Counter[v]);
                    vertices_scan[3*v + k] = 0.0f;
                }
                Counter[v] = 0;
                Valid[v] = true;
            }
            if (!Valid[v])
                res++;
        }
        cout << "remaining vertices: " << res << endl;
    }
    memcpy(vertices_scan, vertices_scan_s, 3 * Nb_Vertices * sizeof(float));*/
    
    delete[] vertices_scan_s;
    delete[] Valid;
    delete[] Counter;
    
}

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}


bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main( int argc, char **argv )
{
    cout << "=======================Program by Diego Thomas ==========================" << endl;
    
    // <==== Set up path variables
    std::string path_model = "D:/Data/Human/Template-star-0.015-0.05/";
    if(cmdOptionExists(argv, argv+argc, "--path_model"))
    {
        std::string tmp (getCmdOption(argv, argv + argc, "--path_model"));
        path_model = tmp;
    } else {
        cout << "Using default path for template: " << path_model << endl;
    }
    
    //std::string path_fit = "F:/HUAWEI/Iwamoto/data/Fitted_smpl/mesh_";
    //std::string path_scan = "F:/HUAWEI/CAPE/data/centered_meshes/mesh_";
    //std::string path_fit = "D:/Human_data/4D_data/data/Fitted_smpl/";
    //std::string path_scan = "D:/Human_data/4D_data/data/centered_meshes/";
    //std::string path_scan = "Z:/Human/b20-kitamura/Human_datasets/4D-people/rp_regina_4d_001_walking_BLD/data/centered_meshes/";
    //std::string path_fit = "F:/HUAWEI/CAPE-33/data/fitted_smpl/mesh_";
    //std::string path_scan = "F:/HUAWEI/CAPE-33/data/GT/mesh_";

    std::string scan_name = "";
    if(cmdOptionExists(argv, argv+argc, "--scan_name"))
    {
        std::string tmp (getCmdOption(argv, argv + argc, "--scan_name"));
        scan_name = tmp;
    } else {
        cout << "Using default name for input 3D scan: " << scan_name << endl;
    }
    
    
    //std::string path_output = "F:/HUAWEI/CAPE-33/data/";
    //std::string path_output = "D:/Human_data/4D_data/data/";
    //std::string path_output = "Z:/Human/b20-kitamura/Human_datasets/4D-people/rp_regina_4d_001_walking_BLD/data
    std::string path;
    if(cmdOptionExists(argv, argv+argc, "--path"))
    {
        std::string tmp (getCmdOption(argv, argv + argc, "--path"));
        path = tmp;
    } else {
        cout << "Using default path for output: " << path << endl;
    }
    
    bool D_data = false;
    
    // ================== START MAIN PROCESS =========================
    
    int nb_scans = 1;
    if(cmdOptionExists(argv, argv+argc, "--size"))
    {
        std::string size_st(getCmdOption(argv, argv + argc, "--size"));
        nb_scans = std::stoi(size_st);
    }

    bool smpl_flg = false;
    if (cmdOptionExists(argv, argv + argc, "--smpl_flg"))
    {
        std::string tmp2(getCmdOption(argv, argv + argc, "--smpl_flg"));
        if (tmp2 == "True") {
            smpl_flg = true;
        }
        else if(tmp2 == "False") {
            smpl_flg = false;
        }
        else {
            cout << "select from True of False" << endl;
        }
    }
    else {
        cout << "Using default smpl_flg : " << smpl_flg << endl;
    }

    std::string smplparams_name = "smplparams";
    if (cmdOptionExists(argv, argv + argc, "--smplparams_dir_name"))
    {
        std::string tmp3(getCmdOption(argv, argv + argc, "--smplparams_dir_name"));
        smplparams_name = tmp3;
    }
    else {
        cout << "Using default smplparams name: " << smplparams_name << endl;
    }


    for (int id_scan = 1; id_scan < nb_scans; id_scan++) {
        
        // <==== load the model obj file
        float *vertices;
        float *normals_face;
        int *faces;
        //int *res = LoadPLY_Mesh(path_model+"Template.ply", &vertices, &normals_face, &faces);
        int* res = LoadOBJ_Mesh(path + smplparams_name + "/T-shape.obj", &vertices, &normals_face, &faces);
        //int *res = LoadPLY_Mesh(path_model+"basic_model_m.ply", &vertices, &normals_face, &faces);
        float *normals = new float[3*res[0]];
        UpdateNormals(vertices, normals, faces, res[0], res[1]);
        cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
        
        // <==== load skin weights
        float *skin_weights = new float[24*res[0]];
        ifstream SWfile (path_model+"TshapeWeights.bin", ios::binary);
        if (!SWfile.is_open()) {
            cout << "Could not load skin weights : " << endl;
            cout << path_model + "TshapeWeights.bin" << endl;
            SWfile.close();
            exit(1);
        }
        SWfile.read((char *) skin_weights, 24*res[0]*sizeof(float));
        SWfile.close();

        string idx_p = to_string(id_scan);
        idx_p.insert(idx_p.begin(), 4 - idx_p.length(), '0');

        string idx_s = to_string(id_scan);
        idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');

        /*if (!D_data) {
            idx_s.insert(idx_s.begin(), 3 - idx_s.length(), '0');
        } else {
            idx_s = to_string(id_scan-1);
            idx_s.insert(idx_s.begin(), 3 - idx_s.length(), '0');
         }*/
        
        string idx_name = to_string(id_scan);
        idx_name.insert(idx_name.begin(), 4 - idx_name.length(), '0');
        /*if (!D_data) {
            if (id_scan < 10)
                idx_name.insert(idx_name.begin(), 2 - idx_name.length(), '0');
            else
                idx_name.insert(idx_name.begin(), 3 - idx_name.length(), '0');
        } else {
            idx_name.insert(idx_name.begin(), 6 - idx_name.length(), '0');
        }*/
                        
        // <==== load the scan ply file
        float *vertices_scan;
        float *normals_face_scan;
        float *uv_scan;
        int *faces_scan;
        int *faces_uv_scan;
        int *res_scan;

        if (smpl_flg) {
            res_scan = LoadOBJ_Mesh(path + "resampled_smpls/" + scan_name + idx_name + ".obj", &vertices_scan, &normals_face_scan, &faces_scan);

            if (res_scan[0] == -1) {
                cout << "Couldn't load : " << path + "resampled_smpls/" + scan_name + idx_name + ".obj" << endl;
                continue;
            }
            cout << "The scan file : " << path + "resampled_smpls/" + scan_name + idx_name + ".obj" << " \nhas " << res_scan[0] << " vertices and " << res_scan[1] << " faces" << endl;
        }
        else {
            res_scan = LoadOBJ_Mesh(path + "resampled_meshes/" + scan_name + idx_name + ".obj", &vertices_scan, &normals_face_scan, &faces_scan);

            if (res_scan[0] == -1) {
                cout << "Couldn't load : " << path + "resampled_meshes/" + scan_name + idx_name + ".obj" << endl;
                continue;
            }
            cout << "The scan file : " << path + "resampled_meshes/" + scan_name + idx_name + ".obj" << " \nhas " << res_scan[0] << " vertices and " << res_scan[1] << " faces" << endl;
        }
        
        
        //Scale(vertices_scan, res_scan[0], 0.001f);
        /*if (!D_data) {
            res_scan = LoadOBJ_Mesh_uv(path_scan + scan_name + idx_name + "/" + scan_name + idx_name + ".obj", &vertices_scan, &normals_face_scan, &uv_scan, &faces_scan, &faces_uv_scan);
        } else {
            res_scan = LoadOBJ_Mesh_uv(path_scan + scan_name + idx_name + ".obj", &vertices_scan, &normals_face_scan, &uv_scan, &faces_scan, &faces_uv_scan);
        }*/

        float *normals_scan = new float[3*res_scan[0]];
        UpdateNormals(vertices_scan, normals_scan, faces_scan, res_scan[0], res_scan[1]);
        //SaveMeshToPLY(path_output + "posed_smpl/test" + idx_s + ".ply", vertices_scan, normals_scan, faces_scan, res_scan[0], res_scan[1]);
        // <==== Load texture image
        /*cv::Mat I;
        if (!D_data) {
            I = cv::imread(path_scan + scan_name + idx_name + "/" + scan_name + idx_name + ".jpg", -1);
        } else {
            idx_name = to_string(id_scan);
            idx_name.insert(idx_name.begin(), 5 - idx_name.length(), '0');
            I = cv::imread(path_scan + "tex." + idx_name + ".png", -1);
        }
        if (I.empty())
        {
            std::cout << "!!! Failed imread(): texture image not found" << std::endl;
            // don't let the execution continue, else imshow() will crash.
            exit(3);
        }
            
        cout << "Texture size: " << I.rows << ", " << I.cols << endl;
        cout << "Pixel size: " << I.channels() << ", " << I.elemSize() << ", " << I.elemSize1() << ", " << I.step << endl;*/

        /*cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::imshow( "Display window", I );
        cv::waitKey(1);*/
            
        /*unsigned char* colors_scan;
        GetColors(&colors_scan, I, I.rows, I.cols, uv_scan, faces_scan, faces_uv_scan, res_scan[0], res_scan[1], res_scan[2]);
        cout << "The OBJ file has " << res_scan[0] << " vertices and " << res_scan[2] << " faces" << endl;*/
                        
        // <==== Pose the template model into the scan pose
        // load kinematic table
        ifstream KTfile (path_model+"kintree.bin", ios::binary);
        if (!KTfile.is_open()) {
            cout << "Could not load kintree" << endl;
            KTfile.close();
            exit(2);
        }
        int *kinTree_Table = new int[48];
        KTfile.read((char *) kinTree_Table, 48*sizeof(int));
        KTfile.close();
        // Load the Skeleton
        Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
        //LoadStarSkeleton( path_output + "smplparams_centered/T_joints.bin", Joints_T, kinTree_Table, 0.0f);
        LoadStarSkeleton(path + smplparams_name + "/T_joints.bin", Joints_T, kinTree_Table, 0.0f);
        //LoadStarSkeleton(path_model + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.0f);       
        //LoadStarSkeleton(path_model+"pose_001.bin_joints.bin", Joints_T, kinTree_Table, 0.0f);
            
        Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
        //LoadSkeleton( path_output + "smplparams_centered/T_joints.bin", path_output + "smplparams_centered/skeleton_" + idx_p + ".bin", Joints, kinTree_Table);
        //LoadSkeleton(path_output + "smplparams_centered/T_joints.bin", path_output + "smplparams_centered/pose_" + idx_p + ".bin", Joints, kinTree_Table);
        LoadSkeleton(path + smplparams_name + "/T_joints.bin", path + smplparams_name + "/pose_" + idx_p + ".bin", Joints, kinTree_Table);
        //LoadSkeleton(path_model+"Tshapecoarsejoints.bin", path_output + "smplparams_bin/gLO_sBM_cAll_d14_mLO1_ch05/" + idx_s + ".bin", Joints, kinTree_Table);

 
        DualQuaternion skeleton[24];
        Eigen::Matrix4f *T_skel = new Eigen::Matrix4f[24]();
        for (int s = 0; s < 24; s++) {
            T_skel[s] = Joints[s] * InverseTransfo(Joints_T[s]);
            //DualQuaternion j_star = DualQuaternion(Joints_T[s]);
            //DualQuaternion j_pose = DualQuaternion(Joints[s]);
            //skeleton[s] = j_pose * j_star.Inv(j_star);
        }
            
        SkinMeshLBS(vertices, skin_weights, T_skel, res[0]);
        //SkinMesh(vertices, skin_weights, skeleton, res[0]);
        UpdateNormalsFaces(vertices, normals_face, faces, res[1]);

        //SaveMeshToPLY(path_output + "posed_smpl/skin_template_" + idx_s + ".ply", vertices, vertices, faces, res[0], res[1]);
        //SaveMeshToPLY(path_output + smplparams_name + "/pose_" + idx_s + ".binfitted.ply", vertices, vertices, faces, res[0], res[1]);
        
            
        // <==== load the fitted template file
        float* vertices_m_f;
        float *normals_face_m_f;
        int *faces_m_f;
        int *res_m_f;
        //cout << path_fit + idx_s + ".obj" << endl;
        //res_m_f = LoadOBJ_Mesh(path_fit + scan_name + idx_s + ".obj", &vertices_m_f, &normals_face_m_f, &faces_m_f);
        res_m_f = LoadOBJ_Mesh(path + smplparams_name + "/pose_" + idx_s + ".binfitted.obj", &vertices_m_f, &normals_face_m_f, &faces_m_f);      //if not retopo  
        //res_m_f = LoadOBJ_Mesh("D:/Data/Human/HUAWEI/4D_datasets/Iwamoto/data/retopo/4dviews_nobi_retopo_0000.obj", &vertices_m_f, &normals_face_m_f, &faces_m_f);        //if retopo

        /*if (!D_data) {
            res_m_f = LoadOBJ_Mesh(path_output + "fitted_smpl/" + scan_name + idx_s + ".obj", &vertices_m_f, &normals_face_m_f, &faces_m_f);
        } else {
            idx_s = to_string(id_scan-1);
            idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
            res_m_f = LoadOBJ_Mesh(path_output + "fitted_smpl/" + "4dviews_nobi_retopo_" + idx_s + ".obj", &vertices_m_f, &normals_face_m_f, &faces_m_f);
        }*/
        cout << "The OBJ file has " << res_m_f[0] << " vertices and " << res_m_f[1] << " faces" << endl;
        
        
        // <=== Compute translation vector between fitted and scan
        // 
        // <=== FOR CAPE DATASET
        // load translation table
        /*ifstream Tfile(path_output + "smplparams_centered/trans_" + idx_p + ".bin", ios::binary);
        //ifstream Tfile(path_output + smplparams_name + "/trans_" + idx_p + ".bin", ios::binary);
        if (!Tfile.is_open()) {
            cout << "Could not load translation" << endl;
            Tfile.close();
            exit(2);
        }
        float* translation = new float[3];
        Tfile.read((char*)translation, 3 * sizeof(float));
        Tfile.close();
        cout << "Translation: " << translation[0] << ", " << translation[1] << ", " << translation[2] << endl;
        Align(vertices_scan, res_scan[0], translation);
        SaveMeshToOBJ(path_output + "centered_meshes/mesh" + idx_s + ".obj", vertices_scan, uv_scan, faces_scan, faces_uv_scan, res_scan[0], res_scan[1], res_scan[2]);*/
        //SaveMeshToPLY(path_output + "centered_meshes/mesh_" + idx_s + ".ply", vertices_scan, faces_scan, res_scan[0], res_scan[1]);

        //res = LoadOBJ_Mesh(path_output + "smplparams_centered/pose_" + idx_s + ".binfitted.obj", &vertices, &normals_face, &faces);
        //Align(vertices_m_f, res[0], translation);
        //SaveMeshToPLY(path_output + "posed_smpl/skin_template_" + idx_s + ".ply", vertices, vertices, faces, res[0], res[1]);


        
        //Align(vertices_scan, res_scan[0], vertices_m_f, res_m_f[0]);
        //Align(vertices_scan, res_scan[0], vertices_m_f, res_m_f[0], vertices, res[0]);
        float* trans = Align(vertices_m_f, res_m_f[0], vertices_m_f, res_m_f[0], vertices, res[0]);
        Align(vertices_scan, res_scan[0], trans);

        cout << "aligned => " << trans[0] << ", " << trans[1] << ", " << trans[2] << endl;
        //SaveMeshToOBJ(path_output + "centered_meshes/mesh_" + idx_s+ ".obj"    , vertices_scan, uv_scan, faces_scan, faces_uv_scan, res_scan[0], res_scan[2], res_scan[1]);
        if (smpl_flg) {
            SaveMeshToPLY(path + "centered_smpls/mesh_" + idx_s + ".ply", vertices_scan, vertices_scan, faces_scan, res_scan[0], res_scan[1]);
            SaveMeshToPLY(path + "centered_smpls/mesh_mf_" + idx_s + ".ply", vertices_m_f, vertices_m_f, faces_m_f, res_m_f[0], res_m_f[1]);
        }
        else {
            SaveMeshToPLY(path + "centered_meshes/mesh_" + idx_s + ".ply", vertices_scan, vertices_scan, faces_scan, res_scan[0], res_scan[1]);
            SaveMeshToPLY(path + "centered_meshes/mesh_mf_" + idx_s + ".ply", vertices_m_f, vertices_m_f, faces_m_f, res_m_f[0], res_m_f[1]);
        }
        cout << "done" << endl;

        // <=== output translation
        auto fileT = std::fstream(path + smplparams_name + "/trans_" + idx_s + ".bin", std::ios::out | std::ios::binary);
        fileT.write((char*)trans, 3 * sizeof(float));
        fileT.close();
        delete[] trans;
        //continue;*/

        // <=== find correspondences from scan to model
        float* weights = GetCorrespondences(skin_weights, vertices_scan, normals_scan, res_scan[0], vertices_m_f, normals_face_m_f, faces_m_f, res_m_f[1]);
        cout << "done" << endl;
           
        std::fstream file;
        // <=== output skin weights
        if (smpl_flg) {
            file = std::fstream(path + "centered_smpls/skin_weights_" + idx_s + ".bin", std::ios::out | std::ios::binary);
        }
        else {
            file = std::fstream(path + "centered_meshes/skin_weights_" + idx_s + ".bin", std::ios::out | std::ios::binary);
        }
        for (int i = 0; i < 24; i++)
            cout << weights[i] << " ";
        cout << endl;
        int size = 24*res_scan[0];
        file.write((char*)weights, size*sizeof(float));
        file.close();

        // <==== load skin weights
        cout << "load skin weights" << endl;
        /*float* weights = new float[24 * res_scan[0]];
        //ifstream SWfile (path_output+"TshapeWeights.bin", ios::binary);
        ifstream file(path_output + "centered_meshes/skin_weights.bin", ios::binary);
        if (!file.is_open()) {
            cout << "Could not load skin weights" << endl;
            file.close();
            exit(1);
        }
        file.read((char*)weights, 24 * res_scan[0] * sizeof(float));
        file.close();*/
        
        // Re-Load the Skeleton
        //LoadStarSkeleton(path_output+"smplparams_centered/T_joints.bin", Joints_T, kinTree_Table, 0.5f);
        LoadStarSkeleton(path + smplparams_name + "/T_joints.bin", Joints_T, kinTree_Table, 0.5f);
        for (int s = 0; s < 24; s++) {
            T_skel[s] = Joints_T[s] * InverseTransfo(Joints[s]);
            //T_skel[s] = Joints[s] * InverseTransfo(Joints_T[s]);
            //DualQuaternion j_star = DualQuaternion(Joints_T[s]);
            //DualQuaternion j_pose = DualQuaternion(Joints[s]);
            //skeleton[s] = j_star * j_pose.Inv(j_pose);
        }
        
        //Copy input vertices  
        /*
        float *vertices_scan_s = new float[3*res_scan[0]];
        memcpy(vertices_scan_s, vertices_scan, 3 * res_scan[0] * sizeof(float));
        */

        // Transform the mesh to the star pose
        cout << "Skin" << endl;
        SkinMeshLBS(vertices_scan, weights, T_skel, res_scan[0]);
        SkinMeshLBS(vertices_m_f, skin_weights, T_skel, res_m_f[0]);
        //SkinMesh(vertices_scan, weights, skeleton, res_scan[0]);
        //SaveMeshToOBJ(viz_path + "/skin_001.obj", vertices_scan, uv_scan, faces_scan, faces_uv_scan, res_scan[0], res_scan[1], res_scan[2]);
        if (smpl_flg) {
            SaveMeshToPLY(path + "canonical_smpls/Fitted_mesh_" + idx_s + ".ply", vertices_m_f, faces_m_f, res_m_f[0], res_m_f[1]);
        }
        else {
            SaveMeshToPLY(path + "canonical_meshes/Fitted_mesh_" + idx_s + ".ply", vertices_m_f, faces_m_f, res_m_f[0], res_m_f[1]);
        }
        cout << "Skinning done" << endl;
        
        
        ///<=====================  CONSISTENCY CHECK ==========================
        ///<=====================  CONSISTENCY CHECK ==========================
        ///<=====================  CONSISTENCY CHECK ==========================
        
        /*
        for (int inner_loop = 0; inner_loop < 10; inner_loop++) {
            CleanSkinningWeights(weights, vertices_scan, faces_scan, res_scan[0], res_scan[1]);
            memcpy(vertices_scan, vertices_scan_s, 3 * res_scan[0] * sizeof(float));
            SkinMeshLBS(vertices_scan, weights, T_skel, res_scan[0]);
        }
        */


        cout << "update normals => " << res_scan[1] << endl;
        UpdateNormals(vertices_scan, normals_scan, faces_scan, res_scan[0], res_scan[1]);
        if (smpl_flg) {
            SavePCasPLY(path + "canonical_smpls/PC_" + idx_s + ".ply", vertices_scan, normals_scan, NULL, res_scan[0]);
        }
        else {
            SavePCasPLY(path + "canonical_meshes/PC_" + idx_s + ".ply", vertices_scan, normals_scan, NULL, res_scan[0]);
        }
        //SavePCasPLY(path_output + "canonical_meshes/PC_" + idx_s+ ".ply", vertices_scan, normals_scan, colors_scan, res_scan[0]);
        //SaveMeshToPLY(path_output + "canonical_meshes/mesh_" + idx_s+ ".ply", vertices_scan, (unsigned char *) NULL, faces_scan, res_scan[0], res_scan[1]);

        /*
        for (int s = 0; s < 24; s++) {
            DualQuaternion j_star = DualQuaternion(Joints_T[s]);
            DualQuaternion j_pose = DualQuaternion(Joints[s]);
            skeleton[s] = j_pose * j_star.Inv(j_star);
        }
        */
        
        SkinMeshLBS(vertices_scan, weights, T_skel, res_scan[0], true);
        //SkinMesh(vertices_scan, weights, skeleton, res_scan[0]);
        //SaveMeshToPLY(path_output + "reposed_meshes/mesh_" + idx_s+ ".ply", vertices_scan, faces_scan, res_scan[0], res_scan[1]);
        //SaveMeshToPLY(path_output + "reposed_meshes/mesh_" + idx_s+ ".ply", vertices_scan, colors_scan, faces_scan, res_scan[0], res_scan[2]);
        
        //CleanMesh(vertices_scan, normals_scan, faces_scan, res_scan[0], res_scan[2]);
        //SaveMeshToPLY(path_output + "canonical_meshes/clean_mesh_" + idx_s+ ".ply", vertices_scan, colors_scan, faces_scan, res_scan[0], res_scan[2]);
        
        /*
        // Skin fitted template to Star shape pose
        SkinMeshLBS(vertices_m_f, skin_weights, T_skel, res[0]);
        UpdateNormalsFaces(vertices_m_f, normals_face_m_f, faces_m_f, res[1]);
        SaveMeshToPLY(path_output + "star_template.ply", vertices_m_f, vertices_m_f, faces_m_f, res[0], res[1]);
        
        // Get new skinning weights from the canonical space
        //float *weights_c = GetCorrespondences(skin_weights, vertices_scan, normals_scan, res_scan[0], vertices_m_f, normals_face_m_f, faces_m_f, res_m_f[1]);
        //cout << "done" << endl;
        
        // Skin back canonical model to pose
        //LoadSkeleton(path_model+"Tshapecoarsejoints.bin", path_output + "smplparams_centered/pose_01.bin", Joints, kinTree_Table);
        for (int s = 0; s < 24; s++) {
            T_skel[s] = Joints[s] * InverseTransfo(Joints_T[s]);
        }
        SkinMeshLBS(vertices_scan, weights, T_skel, res_scan[0]);
        SaveMeshToPLY(path_output + "reposed_meshes/mesh_" + idx_s+ ".ply", vertices_scan, colors_scan, faces_scan, res_scan[0], res_scan[2]);
        
        // <==== Clean up weights with cycle consistency
        //CicleConsistencyCheck(weights, vertices_scan, vertices_scan_s, res_scan[0]);*/
        
            
        //delete[] weights;
        //delete[] vertices_m_f;
        
        cout << "DONE" << endl;
        delete[] vertices_scan;
        delete[] normals_face_scan;
        delete[] normals_scan;
        //delete[] colors_scan;
        //delete[] uv_scan;
        delete[] faces_scan;
        //delete[] faces_uv_scan;
        delete[] res_scan;
        
        delete[] vertices;
        delete[] normals_face;
        delete[] normals;
        delete[] faces;
        delete[] res;
    }
    
	return 0 ;
}

