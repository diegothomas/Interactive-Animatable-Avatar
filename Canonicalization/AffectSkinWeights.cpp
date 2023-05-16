#include "include/Utilities.h"
#include "include/MeshUtils.h"

using namespace std;

void InnerLoop(float* skin_weights, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m, float* result, unsigned char* debug_color, int id) {
    float* corr = new float[24];
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s / 100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3 * i], vertices_s[3 * i + 1], vertices_s[3 * i + 2]);
        my_float3 ray = make_my_float3(normals_s[3 * i], normals_s[3 * i + 1], normals_s[3 * i + 2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        int closest_point = 0;
        for (int f = 0; f < Nb_Faces_m; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_faces_m[3 * f], normals_faces_m[3 * f + 1], normals_faces_m[3 * f + 2]);
            if (dot(n, ray) < 0.4f)
                continue;

            my_float3 p1 = make_my_float3(vertices_m[3 * faces_m[3 * f]], vertices_m[3 * faces_m[3 * f] + 1], vertices_m[3 * faces_m[3 * f] + 2]);
            my_float3 p2 = make_my_float3(vertices_m[3 * faces_m[3 * f + 1]], vertices_m[3 * faces_m[3 * f + 1] + 1], vertices_m[3 * faces_m[3 * f + 1] + 2]);
            my_float3 p3 = make_my_float3(vertices_m[3 * faces_m[3 * f + 2]], vertices_m[3 * faces_m[3 * f + 2] + 1], vertices_m[3 * faces_m[3 * f + 2] + 2]);

            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //closest_point = faces_m[3*f];
                SkinWeightsFromFace3D(corr, &skin_weights[24 * faces_m[3 * f]], &skin_weights[24 * faces_m[3 * f + 1]], &skin_weights[24 * faces_m[3 * f + 2]], p0, p1, p2, p3, n);
            }
        }

        if (min_dist < 0.04f) {
            for (int k = 0; k < 24; k++) {
                result[i * 24 + k] = corr[k];
            }
            debug_color[3 * i] = 128;
            debug_color[3 * i + 1] = 128;
            debug_color[3 * i + 2] = 128;
        }
        else {
            /*
            vertices_s[3 * i] = 0.0f;
            vertices_s[3 * i + 1] = 0.0f;
            vertices_s[3 * i + 2] = 0.0f;
            */
            /*for (int k = 0; k < 24; k++) {
                result[i * 24 + k] = -1.0f;
            }*/

            for (int k = 0; k < 24; k++) {
                result[i * 24 + k] = 0.0f;
            }

            debug_color[3 * i] = 255;
            debug_color[3 * i + 1] = 0;
            debug_color[3 * i + 2] = 0;
        }

    }
    delete[] corr;
}


void InnerLoopPC(float* skin_weights, float* vertices_s, int Nb_Vertices_s, float* vertices_m, int Nb_Vertices_m, float* result, unsigned char* debug_color ,int id) {

    float* corr = new float[24];
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s / 100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3 * i], vertices_s[3 * i + 1], vertices_s[3 * i + 2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;

        for (int f = 0; f < Nb_Vertices_m; f++) {
            // Compute distance point to point
            my_float3 p1 = make_my_float3(vertices_m[3 * f], vertices_m[3 * f + 1], vertices_m[3 * f + 2]);

            float curr_dist = sqrt(dot(p0 - p1, p0 - p1));

            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                for (int k = 0; k < 24; k++) {
                    corr[k] = skin_weights[24 * f + k];
                }
            }
        }

        if (min_dist < 0.01f) {
            for (int k = 0; k < 24; k++) {
                result[i * 24 + k] = corr[k];
            }
            debug_color[3 * i]     = 128;
            debug_color[3 * i + 1] = 128;
            debug_color[3 * i + 2] = 128;
        }
        else {
            /*
            vertices_s[3 * i]     = 0.0f;
            vertices_s[3 * i + 1] = 0.0f;
            vertices_s[3 * i + 2] = 0.0f;
            */

            for (int k = 0; k < 24; k++) {
                result[i * 24 + k] = 0.0f;
            }

            debug_color[3 * i] = 255;
            debug_color[3 * i + 1] = 0;
            debug_color[3 * i + 2] = 0;
        }
    }
    delete[] corr;
}


void InnerLoop_from_closestMesh(float* skin_weights, float* skin_weights2, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m, float* vertices_m2, float* normals_faces_m2, int* faces_m2, int Nb_Faces_m2, float* result, int id) {

    float* corr = new float[24];
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s / 100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3 * i], vertices_s[3 * i + 1], vertices_s[3 * i + 2]);
        my_float3 ray = make_my_float3(normals_s[3 * i], normals_s[3 * i + 1], normals_s[3 * i + 2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        int closest_point = 0;
        for (int f = 0; f < Nb_Faces_m; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_faces_m[3 * f], normals_faces_m[3 * f + 1], normals_faces_m[3 * f + 2]);
            if (dot(n, ray) < 0.4f)
                continue;

            my_float3 p1 = make_my_float3(vertices_m[3 * faces_m[3 * f]], vertices_m[3 * faces_m[3 * f] + 1], vertices_m[3 * faces_m[3 * f] + 2]);
            my_float3 p2 = make_my_float3(vertices_m[3 * faces_m[3 * f + 1]], vertices_m[3 * faces_m[3 * f + 1] + 1], vertices_m[3 * faces_m[3 * f + 1] + 2]);
            my_float3 p3 = make_my_float3(vertices_m[3 * faces_m[3 * f + 2]], vertices_m[3 * faces_m[3 * f + 2] + 1], vertices_m[3 * faces_m[3 * f + 2] + 2]);

            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //closest_point = faces_m[3*f];
                SkinWeightsFromFace3D(corr, &skin_weights[24 * faces_m[3 * f]], &skin_weights[24 * faces_m[3 * f + 1]], &skin_weights[24 * faces_m[3 * f + 2]], p0, p1, p2, p3, n);
            }
        }

        for (int f = 0; f < Nb_Faces_m2; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_faces_m2[3 * f], normals_faces_m2[3 * f + 1], normals_faces_m2[3 * f + 2]);
            if (dot(n, ray) < 0.4f)
                continue;

            my_float3 p1 = make_my_float3(vertices_m2[3 * faces_m2[3 * f    ]], vertices_m2[3 * faces_m2[3 * f]     + 1], vertices_m2[3 * faces_m2[3 * f]     + 2]);
            my_float3 p2 = make_my_float3(vertices_m2[3 * faces_m2[3 * f + 1]], vertices_m2[3 * faces_m2[3 * f + 1] + 1], vertices_m2[3 * faces_m2[3 * f + 1] + 2]);
            my_float3 p3 = make_my_float3(vertices_m2[3 * faces_m2[3 * f + 2]], vertices_m2[3 * faces_m2[3 * f + 2] + 1], vertices_m2[3 * faces_m2[3 * f + 2] + 2]);

            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //closest_point = faces_m[3*f];
                SkinWeightsFromFace3D(corr, &skin_weights2[24 * faces_m2[3 * f]], &skin_weights2[24 * faces_m2[3 * f + 1]], &skin_weights2[24 * faces_m2[3 * f + 2]], p0, p1, p2, p3, n);
            }
        }

        for (int k = 0; k < 24; k++) {
            //cout << k << "  " << skin_weights[closest_point*24 + k] << endl;
            //result[i*24 + k] = skin_weights[closest_point*24 + k];
            result[i * 24 + k] = corr[k];
        }
    }
    delete[] corr;
}



float* GetCorrespondences(float* skin_weights, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m , unsigned char* debug_color) {
    float* result = new float[24 * Nb_Vertices_s];

    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back(std::thread(InnerLoop, skin_weights, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, Nb_Faces_m, result , debug_color, i * (Nb_Vertices_s / 100)));
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

    return result;
}

float* GetCorrespondencesPC(float* skin_weights, float* vertices_s, int Nb_Vertices_s, float* vertices_m, int Nb_Vertices_m , unsigned char* debug_color) {
    float *result      = new float[24 * Nb_Vertices_s];

    for (int i = 0; i < 24 * Nb_Vertices_s; i++) {
        result[i] = 0.0f;
    }

    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoopPC(skin_weights, vertices_s, Nb_Vertices_s, vertices_m, Nb_Vertices_m, result, i * (Nb_Vertices_s / 100));
        my_threads.push_back(std::thread(InnerLoopPC, skin_weights, vertices_s, Nb_Vertices_s, vertices_m, Nb_Vertices_m, result , debug_color, i * (Nb_Vertices_s / 100)));
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

    return result;
}

float* GetCorrespondences_from_closestMesh(float* skin_weights, float* skin_weights2, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m , float* vertices_m2, float* normals_faces_m2, int* faces_m2, int Nb_Faces_m2) {
    float* result = new float[24 * Nb_Vertices_s];

    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        my_threads.push_back(std::thread(InnerLoop_from_closestMesh, skin_weights, skin_weights2, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, Nb_Faces_m, vertices_m2, normals_faces_m2, faces_m2, Nb_Faces_m2 , result, i * (Nb_Vertices_s / 100)));
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

    return result;
}

int CleanMesh(float *skin_weights, float *vertices_s, int *faces_s, int Nb_Vertices_s, int NB_Faces) {
    int *counter = new int[Nb_Vertices_s];
    for (int i = 0; i < Nb_Vertices_s; i++)
        counter[i] = 0;
    
    int valid_summit[3];
    for (int f = 0; f < NB_Faces; f++) {
        for (int s = 0; s < 3; s++) {
            valid_summit[s] = 0;
            for (int k = 0; k < 24; k++) {
                if (skin_weights[faces_s[3*f+s]*24 + k] != 0.0f)
                    valid_summit[s] = 1;
            }
        }
        if ((valid_summit[0]+valid_summit[1]+valid_summit[2]) == 0 ||
            (valid_summit[0]+valid_summit[1]+valid_summit[2]) == 3)
            continue;
        
        for (int s = 0; s < 3; s++) {
            if (valid_summit[s] == 0) {
                for (int s_c = 0; s_c < 3; s_c++) {
                    if (valid_summit[s_c] == 1) {
                        for (int k = 0; k < 24; k++) {
                            skin_weights[faces_s[3*f+s]*24 + k] = skin_weights[faces_s[3*f+s]*24 + k] + skin_weights[faces_s[3*f+s_c]*24 + k];
                            counter[faces_s[3*f+s]] = counter[faces_s[3*f+s]] + 1;
                        }
                    }
                }
            }
        }
        
    }
    
    int nb_clean = 0;
    int nb_remain = 0;
    for (int i = 0; i < Nb_Vertices_s; i++) {
        if (counter[i] > 0) {
            nb_clean++;
            float tot = 0.0f;
            for (int k = 0; k < 24; k++)
                tot += skin_weights[i*24 + k];
            
            if (tot > 0.0f) {
                for (int k = 0; k < 24; k++)
                    skin_weights[i*24 + k] = skin_weights[i*24 + k]/tot;
            }
        }
        
        bool valid_summit = false;
        for (int k = 0; k < 24; k++) {
            if (skin_weights[i*24 + k] != 0.0f)
                valid_summit = true;
        }
        if (!valid_summit)
            nb_remain ++;
    }
    
    cout << "number of vertex cleaned: " << nb_clean << endl;
    cout << "remaining number of vertex to clean: " << nb_remain << endl;
    return nb_remain;
}


void Align( float *vertices_s, int Nb_Vertices_s, float *vertices_m_f, int Nb_Vertices_m_f) {
    my_float3 avg_m = make_my_float3(0.0f,0.0f,0.0f);
    for (int i = 0; i < Nb_Vertices_m_f; i++) {
        avg_m = avg_m + make_my_float3(vertices_m_f[3*i], vertices_m_f[3*i+1], vertices_m_f[3*i+2]);
    }
    avg_m = avg_m * (1.0f/float(Nb_Vertices_m_f));
    
    my_float3 avg_s = make_my_float3(0.0f,0.0f,0.0f);
    for (int i = 0; i < Nb_Vertices_s; i++) {
        avg_s = avg_s + make_my_float3(vertices_s[3*i], vertices_s[3*i+1], vertices_s[3*i+2]);
    }
    avg_s = avg_s * (1.0f/float(Nb_Vertices_s));
    
    my_float3 trans = avg_s-avg_m;
    
    for (int i = 0; i < Nb_Vertices_m_f; i++) {
        vertices_m_f[3*i] = vertices_m_f[3*i] + trans.x;
        vertices_m_f[3*i+1] = vertices_m_f[3*i+1] + trans.y;
        vertices_m_f[3*i+2] = vertices_m_f[3*i+2] + trans.z;
    }
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
    //std::string path_model = "C:/Users/thomas/Documents/Projects/Human-AI/Kitamura/huawei-deep-animation/Canonicalization/models/";
    std::string path_model = "D:/Data/Human/Template-star-0.015-0.05/";
    if (cmdOptionExists(argv, argv + argc, "--path_model"))
    {
        std::string tmp(getCmdOption(argv, argv + argc, "--path_model"));
        path_model = tmp;
    }
    else {
        cout << "Using default path for path: " << path_model << endl;
    }
    // <==== Set up path variables
    //std::string path = "F:/HUAWEI/CAPE-33/data/";
    std::string path = "";
    if(cmdOptionExists(argv, argv+argc, "--path"))
    {
        std::string tmp (getCmdOption(argv, argv + argc, "--path"));
        path = tmp;
    } else {
        cout << "Using default path for path: " << path << endl;
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
        else if (tmp2 == "False") {
            smpl_flg = false;
        }
        else {
            cout << "select from True of False" << endl;
        }
    }
    else {
        cout << "Using default smpl_flg : " << smpl_flg << endl;
    }

    for (int id_scan = 1; id_scan < nb_scans; id_scan++) {
        
        string idx_s = to_string(id_scan);
        idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
                
        // <==== load the model obj file
        float* vertices;
        float *normals_face;
        int *faces;
        //int *res = LoadPLY_Mesh(path_model+"basic_model_m.ply", &vertices, &normals_face, &faces);
        int* res = LoadPLY_Mesh(path_model + "Template.ply", &vertices, &normals_face, &faces);
        float *normals = new float[3*res[0]];
        UpdateNormals(vertices, normals, faces, res[0], res[1]);
        cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
        
       
        string canonical_mesh_path;
        string canonical_PC_path;
        string skinweights_PC_path;
        string out_skinweights_path;

        if (smpl_flg) {
            canonical_mesh_path  = path + "canonical_smpls/mesh_"         + idx_s + ".ply";
            canonical_PC_path    = path + "canonical_smpls/PC_"           + idx_s + ".ply";
            skinweights_PC_path  = path + "centered_smpls/skin_weights_"  + idx_s + ".bin";
            out_skinweights_path = path + "canonical_smpls/skin_weights_" + idx_s + ".bin";
        }
        else {
            canonical_mesh_path  = path + "canonical_meshes/mesh_"         + idx_s + ".ply";
            canonical_PC_path    = path + "canonical_meshes/PC_"           + idx_s + ".ply";
            skinweights_PC_path  = path + "centered_meshes/skin_weights_"  + idx_s + ".bin";
            out_skinweights_path = path + "canonical_meshes/skin_weights_" + idx_s + ".bin";
        }

        float* vertices_m_f;
        float* normals_face_m_f;
        int* faces_m_f;
        int* res_m_f;
        res_m_f = LoadPLY_Mesh(canonical_mesh_path, &vertices_m_f, &normals_face_m_f, &faces_m_f);
        if (res_m_f[0] == -1)
            continue;
        cout << "Load : " + canonical_mesh_path << endl;
        float *normals_m_f = new float[3*res_m_f[0]];
        UpdateNormals(vertices_m_f, normals_m_f, faces_m_f, res_m_f[0], res_m_f[1]);
        
        cout << "The OBJ file has " << res_m_f[0] << " vertices and " << res_m_f[1] << " faces" << endl;
        
        //string SW_src = "FittedMesh";
        string SW_src = "scanPC";

        float* vertices_f;
        float* normals_face_f;
        int* faces_f;
        int* res_f;
        

        float* vertices_PC;
        float* normals_face_PC;
        int* faces_PC;
        int res_PC;
        
        /**
            1. Load the Canonical FittedMesh or scanPC and these corresponding skin weights
        */
        //if use FittedMesh
        /*
        cout << "==== Step 1.1 :   Load the Canonical FittedMesh ====" << endl;
            
        if (!D_data) {
            //res_f = LoadOBJ_Mesh(path + "fitted_smpl/mesh_" + idx_s + ".obj", &vertices_f, &normals_face_f, &faces_f);
            res_f = LoadPLY_Mesh(path + "canonical_meshes/Fitted_mesh_" + idx_s + ".ply", &vertices_f, &normals_face_f, &faces_f);
        }
        else {
            idx_s = to_string(id_scan);
            idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
            res_f = LoadOBJ_Mesh(path + "fitted_smpl/" + "4dviews_nobi_retopo_" + idx_s + ".obj", &vertices_f, &normals_face_f, &faces_f);
        }
        float* normals_f = new float[3 * res_f[0]];
        UpdateNormals(vertices_f, normals_f, faces_f, res_f[0], res_f[1]);
        cout << "The OBJ file has " << res_f[0] << " vertices and " << res_f[1] << " faces" << endl;

        cout << "==== Step 1.2 :  Load FittedMesh's skinning weights ====" << endl;

        float* skin_weights = new float[24 * res[0]];
        ifstream SWfile(path_model + "TshapeWeights.bin", ios::binary);
        if (!SWfile.is_open()) {
            cout << "Could not load skin weights" << endl;
            SWfile.close();
            exit(1);
        }
        SWfile.read((char*)skin_weights, 24 * res[0] * sizeof(float));
        SWfile.close();
        */

        //if use scanPC
        cout << "==== Step 1-1 :   Load the scanPC in canonical pose ====" << endl;
       
        res_PC = LoadPLY(canonical_PC_path, &vertices_PC);
        if (res_PC == -1)
            return -1;
        cout << "The PLY file has " << res_PC << " vertices" << endl;

        float* skin_weights_PC = new float[24 * res_PC];

        cout << "==== Step 1-2 :  Load skinning weights ====" << endl;
        // Load skinning weights    
        ifstream WPfile(skinweights_PC_path, ios::binary);
        if (!WPfile.is_open()) {
            cout << "Could not load skin weights" << endl;
            cout << "¨ : " << skinweights_PC_path << endl;
            WPfile.close();
            return -1;
        }
        WPfile.read((char*)skin_weights_PC, 24 * res_PC * sizeof(float));
        WPfile.close();
        

        // <=== Compute translation vector between template and scan (and affect vertices_f)
        
        // <==== load skin weights
        /*float* skin_weights = new float[24 * res_f[0]];
        //ifstream SWfile (path_model+"TshapeWeights.bin", ios::binary);
        ifstream SWfile (path_model+"basicModel_m_skinweights.bin", ios::binary);
        if (!SWfile.is_open()) {
            cout << "Could not load skin weights" << endl;
            SWfile.close();
            exit(1);
        }
        SWfile.read((char *) skin_weights, 24*res_f[0]*sizeof(float));
        SWfile.close();*/
        
        
        // ================================
        // <=== make it canonical pose
        // ================================
        
        // <==== Pose the template model into the scan pose
        // load kinematic table
        ifstream KTfile(path_model + "kintree.bin", ios::binary);
        if (!KTfile.is_open()) {
            cout << "Could not load kintree" << endl;
            KTfile.close();
            exit(2);
        }
        int *kinTree_Table = new int[48];
        KTfile.read((char *) kinTree_Table, 48*sizeof(int));
        KTfile.close();
        
        
        // Load the Skeleton
        //Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
        //LoadStarSkeleton( path + "smplparams_centered/T_joints.bin", Joints_T, kinTree_Table, 0.0f);
            
        //Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
        //LoadSkeleton( path + "smplparams_centered/T_joints.bin", path + "smplparams_centered/pose_" + idx_p + ".bin", Joints, kinTree_Table);
        
        //Eigen::Matrix4f *T_skel = new Eigen::Matrix4f[24]();
        /*
        for (int s = 0; s < 24; s++) {
            //T_skel[s] = Joints[s] * InverseTransfo(Joints_T[s]);
            T_skel[s] = Joints_T[s] * InverseTransfo(Joints[s]);
        }
            
        SkinMeshLBS(vertices, skin_weights, T_skel, res[0], true);
        UpdateNormalsFaces(vertices, normals_face, faces, res[1]);
                
        Align(vertices, res[0], vertices_f, res_f[0]);
        //SaveMeshToPLY(path + "canonical_meshes/centered_mesh_" + idx_s+ ".ply", vertices_f, normals_f, faces_f, res_f[0], res_f[1]);
        */             
        // Re-Load the Skeleton
        /*
        LoadStarSkeleton( path + "smplparams/T_joints.bin", Joints_T, kinTree_Table, 0.5f);  
        LoadSkeleton( path + "smplparams/T_joints.bin", path + "smplparams/pose_" + idx_s + ".bin", Joints, kinTree_Table);
        for (int s = 0; s < 24; s++) {
            T_skel[s] = Joints_T[s] * InverseTransfo(Joints[s]);
        }*/
        //SkinMeshLBS(vertices_f, skin_weights, T_skel, res_f[0]);
        //UpdateNormalsFaces(vertices_f, normals_face_f, faces_f, res_f[1]);
        //SaveMeshToPLY(path + "canonical_meshes/fitted_mesh_" + idx_s+ ".ply", vertices_f, vertices_f, faces_f, res_f[0], res_f[1]);

        // <==== load PC_xxxx.ply and skin weights 
        cout << "==== Step 2 :  Get new skinning weights from the canonical spaces ====" << endl;
        unsigned char* debug_color = new unsigned char[3 * res_m_f[0]];
        float* weights_c;
        // Get new skinning weights from the canonical space
        //if use "FittedMesh"
        //weights_c = GetCorrespondences(skin_weights, vertices_m_f, normals_m_f, res_m_f[0], vertices_f, normals_face_f, faces_f, res_f[1] , debug_color);
        //if use "scanPC"
        weights_c = GetCorrespondencesPC(skin_weights_PC, vertices_m_f, res_m_f[0], vertices_PC, res_PC , debug_color);
        
        cout << "done" << endl;          

        // <=== output skin weights
        auto file = std::fstream(out_skinweights_path , std::ios::out | std::ios::binary);
        int size = 24*res_m_f[0];
        file.write((char*)weights_c, size*sizeof(float));
        file.close();
        
        //SaveMeshToPLY(path + "canonical_meshes/debug.ply", vertices_m_f , debug_color, faces_m_f ,  res_m_f[0] , res_m_f[1]);

        //debug
        /*
        int* debug_color1 = new int[3 * res_m_f[0]];
        
        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 0] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 1] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 2] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw1.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);

        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 3] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 4] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 5] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw2.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);

        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 6] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 7] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 8] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw3.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);

        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 9] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 10] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 11] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw4.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);

        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 12] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 13] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 14] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw5.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);

        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 15] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 16] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 17] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw6.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);

        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 18] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 19] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 20] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw7.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);

        for (int l = 0; l < res_m_f[0]; l++) {
            debug_color1[3 * l + 0] = weights_c[24 * l + 21] * 255;
            debug_color1[3 * l + 1] = weights_c[24 * l + 22] * 255;
            debug_color1[3 * l + 2] = weights_c[24 * l + 23] * 255;
        }
        SaveMeshToPLY(path + "canonical_meshes/debug_sw8.ply", vertices_m_f, debug_color1, faces_m_f, res_m_f[0], res_m_f[1]);
        */

        // ================================
        // <=== test if skin weights are OK
        // ================================
        // Load the Skeleton
        /*
        Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
        Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
        Eigen::Matrix4f* T_skel = new Eigen::Matrix4f[24]();

        LoadStarSkeleton(path + "smplparams/T_joints.bin", Joints_T, kinTree_Table, 0.5f);
        LoadSkeleton(path + "smplparams/T_joints.bin", path + "smplparams/pose_" + idx_s + ".bin", Joints, kinTree_Table);

        for (int s = 0; s < 24; s++) {
            T_skel[s] = Joints_T[s] * InverseTransfo(Joints[s]);
        }
        SkinMeshLBS(vertices_m_f, weights_c, T_skel, res_m_f[0], true);
        SaveMeshToPLY(path + "canonical_meshes/rep_mesh_" + idx_s+ ".ply", vertices_m_f, normals_m_f, faces_m_f, res_m_f[0], res_m_f[1]);
        */  
        delete[] vertices_m_f;
        delete[] normals_face_m_f;
        delete[] faces_m_f;
        delete[] res_m_f;
        delete[] normals_m_f;
        delete[] weights_c;
        
        //if use "FittedMesh"
        //delete[] skin_weights;
        //delete[] vertices_f;
        //delete[] normals_f;
        
        //if use "scanPC"
        delete[] skin_weights_PC;
        delete[] vertices_PC;
        

        delete[] vertices;
        delete[] normals;
        delete[] faces;
    }
    
    
	return 0 ;
}

