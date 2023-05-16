#include "../include/Utilities.h"
#include "../include/MeshUtils.h"

using namespace std;


void InnerLoop3(float *vertices_m, float *normals_m, int Nb_Vertices_m, float *vertices_s, float *normals_s, int Nb_Vertices_s, float *result, int id) {
    
    for (int i = id; i < min(Nb_Vertices_m, id + Nb_Vertices_m/100); i++) {
        my_float3 p0 = make_my_float3(vertices_m[3*i], vertices_m[3*i+1], vertices_m[3*i+2]);
        my_float3 ray = make_my_float3(normals_m[3*i], normals_m[3*i+1], normals_m[3*i+2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        float corr = 0.0f;
        for (int f = 0; f < Nb_Vertices_s; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_s[3*f], normals_s[3*f+1], normals_s[3*f+2]);
            if (dot(n,ray) < 0.6f)
               continue;
                
            my_float3 p1 = make_my_float3(vertices_s[3*f], vertices_s[3*f+1], vertices_s[3*f+2]);
            
            // Compute point to face distance
            float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                corr = float(f);
            }
        }
        //cout << corr.x << ", " << corr.y << ", " << corr.z << endl;
        if (min_dist < 1.0e4f) {
            result[2*i] = corr;
            result[2*i+1] = 1.0f;
        } else {
            result[2*i] = 0.0f;
            result[2*i+1] = 0.0f;
        }
    }
}

float *GetCorrespondencesIdx(float *vertices_m, float *normals_m, int Nb_Vertices_m, float *vertices_s, float *normals_s, int Nb_Vertices_s) {
    float *result = new float[2*Nb_Vertices_m];
    
    cout << Nb_Vertices_s << ", " << Nb_Vertices_m << endl;
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back( std::thread(InnerLoop3, vertices_m, normals_m, Nb_Vertices_m, vertices_s, normals_s, Nb_Vertices_s, result, i*(Nb_Vertices_m/100)) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return result;
}

void InnerLoop2(float *vertices_m, float *normals_m, int Nb_Vertices_m, float *vertices_s, float *normals_s, int Nb_Vertices_s, float *result, int id) {
    for (int i = id; i < id + 1 /*Nb_Vertices_m / 100*/; i++) {
        my_float3 p0 = make_my_float3(vertices_m[3*i], vertices_m[3*i+1], vertices_m[3*i+2]);
        my_float3 ray = make_my_float3(normals_m[3*i], normals_m[3*i+1], normals_m[3*i+2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        my_float3 corr = make_my_float3(0.0f,0.0f,0.0f);
        for (int f = 0; f < Nb_Vertices_s; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_s[3*f], normals_s[3*f+1], normals_s[3*f+2]);
            if (dot(n,ray) < 0.6f)
               continue;
                
            my_float3 p1 = make_my_float3(vertices_s[3*f], vertices_s[3*f+1], vertices_s[3*f+2]);
            
            // Compute point to face distance
            float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                corr = p1;
            }
        }
        //cout << corr.x << ", " << corr.y << ", " << corr.z << endl;
        if (min_dist < 1.0e4f) {
            result[4*i] = corr.x;
            result[4*i+1] = corr.y;
            result[4*i+2] = corr.z;
            result[4*i+3] = 1.0f;
        } else {
            result[4*i] = 0.0f;
            result[4*i+1] = 0.0f;
            result[4*i+2] = 0.0f;
            result[4*i+3] = 0.0f;
        }
       //cout << ".";
    }
}

float *GetCorrespondences2(float *vertices_m, float *normals_m, int Nb_Vertices_m, float *vertices_s, float *normals_s, int Nb_Vertices_s) {
    float *result = new float[4*Nb_Vertices_m];
    
    std::vector< std::thread > my_threads;
    for (int i = 0; i < Nb_Vertices_m; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back(std::thread(InnerLoop2, vertices_m, normals_m, Nb_Vertices_m, vertices_s, normals_s, Nb_Vertices_s, result, i)); // i * (Nb_Vertices_m / 100)) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

    /*for (int i = 0; i < Nb_Vertices_m; i++) {
        cout << "source: " << vertices_m[3*i] << ", " << vertices_m[3 * i + 1] << ", " << vertices_m[3 * i + 2] << endl;
        cout << "match: " << result[4 * i] << ", " << result[4 * i + 1] << ", " << result[4 * i + 2] << endl;
    }*/

    return result;
}

void InnerLoop(float *vertices_m, float *normals_m, int Nb_Vertices_m, float *vertices_s, float *normals_faces_s, int *faces_s, int Nb_Faces_s, float *result, int id) {
    
    for (int i = id; i < id + Nb_Vertices_m/100; i++) {
        my_float3 p0 = make_my_float3(vertices_m[3*i], vertices_m[3*i+1], vertices_m[3*i+2]);
        my_float3 ray = make_my_float3(normals_m[3*i], normals_m[3*i+1], normals_m[3*i+2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        my_float3 corr = make_my_float3(0.0f,0.0f,0.0f);
        for (int f = 0; f < Nb_Faces_s; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_faces_s[3*f], normals_faces_s[3*f+1], normals_faces_s[3*f+2]);
            if (dot(n,ray) > 0.0f)
               continue;
                
            my_float3 p1 = make_my_float3(vertices_s[3*faces_s[3*f]], vertices_s[3*faces_s[3*f]+1], vertices_s[3*faces_s[3*f]+2]);
            my_float3 p2 = make_my_float3(vertices_s[3*faces_s[3*f+1]], vertices_s[3*faces_s[3*f+1]+1], vertices_s[3*faces_s[3*f+1]+2]);
            my_float3 p3 = make_my_float3(vertices_s[3*faces_s[3*f+2]], vertices_s[3*faces_s[3*f+2]+1], vertices_s[3*faces_s[3*f+2]+2]);
            
            if (IsInterectingRayTriangle3D(ray, p0, p1, p2, p3, n) == 0.0f) {
                continue;
            }
            
            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                corr = p1; //ProjPointFace3D(p0, p1, p2, p3, n);
            }
        }
        //cout << corr.x << ", " << corr.y << ", " << corr.z << endl;
        if (min_dist < 1.0e4f) {
            result[4*i] = corr.x;
            result[4*i+1] = corr.y;
            result[4*i+2] = corr.z;
            result[4*i+3] = 1.0f;
        } else {
            result[4*i] = 0.0f;
            result[4*i+1] = 0.0f;
            result[4*i+2] = 0.0f;
            result[4*i+3] = 0.0f;
        }
    }
}

float *GetCorrespondences(float *vertices_m, float *normals_m, int Nb_Vertices_m, float *vertices_s, float *normals_faces_s, int *faces_s, int Nb_Faces_s) {
    float *result = new float[4*Nb_Vertices_m];
    
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back( std::thread(InnerLoop, vertices_m, normals_m, Nb_Vertices_m, vertices_s, normals_faces_s, faces_s, Nb_Faces_s, result, i*(Nb_Vertices_m/100)) );
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    
    return result;
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
    
    if(cmdOptionExists(argv, argv+argc, "--path_model"))
    {
        std::string ply_path (getCmdOption(argv, argv + argc, "--path_model"));
        
        // <==== load the model ply file
        float *vertices;
        float *normals_face;
        int *faces;
        int *res = LoadPLY_Mesh(ply_path, &vertices, &normals_face, &faces);
        float *normals = new float[3*res[0]];
        UpdateNormals(vertices, normals, faces, res[0], res[1]);
        cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
        
        
        if(cmdOptionExists(argv, argv+argc, "--path_scan"))
        {
            std::string ply_scan(getCmdOption(argv, argv + argc, "--path_scan"));
            //std::string obj_scan(getCmdOption(argv, argv + argc, "--path_scan"));
            
            // <==== load the scan ply file
            float *vertices_scan;
            float *normals_face_scan;
            int *faces_scan;
            int *res_scan = LoadPLY_Mesh(ply_scan, &vertices_scan, &normals_face_scan, &faces_scan);
            //int* res_scan = LoadOBJ_Mesh(obj_scan, &vertices_scan, &normals_face_scan, &faces_scan);
            float *normals_scan = new float[3*res_scan[0]];
            UpdateNormals(vertices_scan, normals_scan, faces_scan, res_scan[0], res_scan[1]);
            cout << "The PLY file has " << res_scan[0] << " vertices and " << res_scan[1] << " faces" << endl;
            
            // <=== find correspondences between model and scan
            //float *corr = GetCorrespondences(vertices, normals, res[0], vertices_scan, normals_face_scan, faces_scan, res_scan[1]);
            //float *cross_corr = GetCorrespondences(vertices_scan, normals_scan, res_scan[0], vertices, normals_face, faces, res[1]);

            //cout << "GetCorrespondences2" << endl;
            float *corr = GetCorrespondences2(vertices, normals, res[0], vertices_scan, normals_scan, res_scan[0]);
            //cout << "GetCorrespondencesIdx" << endl;
            float *cross_corr = GetCorrespondencesIdx(vertices_scan, normals_scan, res_scan[0], vertices, normals, res[0]);

           // cout << "Done" << endl;
            if(cmdOptionExists(argv, argv+argc, "--path_out"))
            {
                std::string out_path(getCmdOption(argv, argv + argc, "--path_out"));
                // <=== output correspondences
                
                //cout << out_path + "corr.bin" << endl;
                auto file = std::fstream(out_path + "corr.bin", std::ios::out | std::ios::binary);
                int size = 4*res[0];
                file.write((char*)corr, size*sizeof(float));
                file.close();
                
                auto cross_file = std::fstream(out_path + "cross_corr.bin", std::ios::out | std::ios::binary);
                int cross_size = 2*res_scan[0];
                cross_file.write((char*)cross_corr, cross_size*sizeof(float));
                cross_file.close();
            }
            
            delete[] corr;
            delete[] cross_corr;
            
            delete[] vertices_scan;
            delete[] normals_face_scan;
            delete[] normals_scan;
            delete[] faces_scan;
            delete[] res_scan;
        }
        
        delete[] vertices;
        delete[] normals_face;
        delete[] normals;
        delete[] faces;
        delete[] res;
    } else {
        return -1;
    }
    
	return 0 ;
}

