#include "include_gpu/Utilities.h"
#include "include_gpu/MarchingCubes.h"
#include "include_gpu/LevelSet.h"
#include "include_gpu/MeshUtils.h"
#include "include_gpu/TetraMesh.h"
#include "include_gpu/CubicMesh.h"
#include "include_gpu/Fit.h"

#include <iomanip>


//using namespace std;

SYSTEMTIME current_time, last_time;

string folder_name = "D:/Data/KinectV2/2020_02_26_10_27_13/camera_0/";
//string output_mesh_folder_name = "D:/Project/Human/HUAWEI/Output/";
string output_mesh_folder_name = "D:/Data/Human/Template-star-0.015/";



int GRID_SIZE = 128;
float GRID_RES = 0.022f;
int COARSE_GRID_SIZE = 64;
float COARSE_GRID_RES = 0.04f;
//int GRID_SIZE = 64;
//float GRID_RES = 0.04f;

void TestArticulated() {
    cout << "==============Test for data format==================" << endl;
    
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    float *vertices;
    float *normals_face;
    int *faces;
    int *res = LoadOBJ_Mesh(folder_name + string("meshes/mesh_0000.obj"), &vertices, &normals_face, &faces);
    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
    
    // load kinematic table
    ifstream KTfile (folder_name+"kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int *kinTree_Table = new int[48];
    KTfile.read((char *) kinTree_Table, 48*sizeof(int));
    KTfile.close();
    
    Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(folder_name+"Tshapecoarsejoints.bin", Joints_T, kinTree_Table);
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    //LoadSkeleton(folder_name+"Tshapecoarsejoints.bin", folder_name + "smplparams_centered/skeleton_" + to_string(0) + ".bin", Joints, kinTree_Table);
    LoadSkeletonVIBE(folder_name+"Tshapecoarsejoints.bin", folder_name+"smplparams_vibe/", Joints, kinTree_Table, 0);

    float *skeletonT = new float[3*24];
    float *skeletonPC = new float[3*24];
    
    DualQuaternion *skeleton = new DualQuaternion[24];
    for (int s = 0; s < 24; s++) {
        //cout << Joints[s] << endl;
        skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
        skeletonT[3*s] = Joints_T[s](0,3);
        skeletonT[3*s+1] = Joints_T[s](1,3);
        skeletonT[3*s+2] = Joints_T[s](2,3);
        skeletonPC[3*s] = Joints[s](0,3);
        skeletonPC[3*s+1] = Joints[s](1,3);
        skeletonPC[3*s+2] = Joints[s](2,3);
    }

    SavePCasPLY(output_mesh_folder_name + string("skel_T.ply"), skeletonT, 24);
    SavePCasPLY(output_mesh_folder_name + string("skel_in.ply"), skeletonPC, 24);
    delete []skeletonPC;
    delete []skeletonT;
    
    // Transform the mesh to the star pose
    //SkinMeshARTICULATED(vertices, skin_weights, skeleton, res[0]);
    
    //UpdateNormalsFaces(vertices, normals_face, faces, res[1]);
    
    SaveMeshToPLY(output_mesh_folder_name + string("skin.ply"), vertices, vertices, faces, res[0], res[1]);
    
    delete []vertices;
    delete []normals_face;
    delete []faces;
    delete []Joints;
    delete []Joints_T;
}

void InnerLoop2(float* skin_weights, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m, float* result, int id) {

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
            /*if (dot(n,ray) > 0.5f)
               continue;*/

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

        for (int k = 0; k < 24; k++) {
            //cout << k << "  " << skin_weights[closest_point*24 + k] << endl;
            //result[i*24 + k] = skin_weights[closest_point*24 + k];
            result[i * 24 + k] = corr[k];
        }
    }
    delete[] corr;
}

void InnerLoopColor(unsigned char* input_color, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m, unsigned char* result_color, int id) {

    unsigned char* corr = new unsigned char[3];
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s / 100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3 * i], vertices_s[3 * i + 1], vertices_s[3 * i + 2]);
        my_float3 ray = make_my_float3(normals_s[3 * i], normals_s[3 * i + 1], normals_s[3 * i + 2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        int closest_point = 0;
        for (int f = 0; f < Nb_Faces_m; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_faces_m[3 * f], normals_faces_m[3 * f + 1], normals_faces_m[3 * f + 2]);
            /*if (dot(n,ray) > 0.5f)
               continue;*/

            my_float3 p1 = make_my_float3(vertices_m[3 * faces_m[3 * f]], vertices_m[3 * faces_m[3 * f] + 1], vertices_m[3 * faces_m[3 * f] + 2]);
            my_float3 p2 = make_my_float3(vertices_m[3 * faces_m[3 * f + 1]], vertices_m[3 * faces_m[3 * f + 1] + 1], vertices_m[3 * faces_m[3 * f + 1] + 2]);
            my_float3 p3 = make_my_float3(vertices_m[3 * faces_m[3 * f + 2]], vertices_m[3 * faces_m[3 * f + 2] + 1], vertices_m[3 * faces_m[3 * f + 2] + 2]);


            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //closest_point = faces_m[3*f];
                ColorFromFace3D(corr, &input_color[3 * faces_m[3 * f]], &input_color[3 * faces_m[3 * f + 1]], &input_color[3 * faces_m[3 * f + 2]], p0, p1, p2, p3, n);
            }
        }

        for (int k = 0; k < 3; k++) {
            //cout << k << "  " << skin_weights[closest_point*24 + k] << endl;
            //result[i*24 + k] = skin_weights[closest_point*24 + k];
            result_color[i * 3 + k] = corr[k];
        }
    }
    delete[] corr;
}

/*
void InnerLoopCorr_uv(cv::Mat I, float* uv, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int* faces_uv_m, int Nb_Faces_m, unsigned char* result, int id) {
    float* corr = new float[2];
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s / 100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3 * i], vertices_s[3 * i + 1], vertices_s[3 * i + 2]);
        my_float3 ray = make_my_float3(normals_s[3 * i], normals_s[3 * i + 1], normals_s[3 * i + 2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        int closest_point = 0;
        for (int f = 0; f < Nb_Faces_m; f++) {
            // Compute distance point to face
            my_float3 n = make_my_float3(normals_faces_m[3 * f], normals_faces_m[3 * f + 1], normals_faces_m[3 * f + 2]);
            if (dot(n, ray) < 0.8f)
                continue;
            my_float3 p1 = make_my_float3(vertices_m[3 * faces_m[3 * f]], vertices_m[3 * faces_m[3 * f] + 1], vertices_m[3 * faces_m[3 * f] + 2]);
            my_float3 p2 = make_my_float3(vertices_m[3 * faces_m[3 * f + 1]], vertices_m[3 * faces_m[3 * f + 1] + 1], vertices_m[3 * faces_m[3 * f + 1] + 2]);
            my_float3 p3 = make_my_float3(vertices_m[3 * faces_m[3 * f + 2]], vertices_m[3 * faces_m[3 * f + 2] + 1], vertices_m[3 * faces_m[3 * f + 2] + 2]);
            //cout << "Color => " << color[3 * faces_m[3 * f]] << " " << color[3 * faces_m[3 * f + 1]] << " " << color[3 * faces_m[3 * f + 2]] << endl;
            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //closest_point = faces_m[3*f];
                UVFromFace3D(corr, &uv[3 * faces_uv_m[3 * f]], &uv[3 * faces_uv_m[3 * f + 1]], &uv[3 * faces_uv_m[3 * f + 2]], p0, p1, p2, p3, n);
            }
        }
        int u = int(corr[1] * I.rows);
        if (u < 0)
            u = I.rows + u;
        while (u > I.rows - 1)
            u = u - I.rows;
        int v = int(corr[0] * I.cols);
        if (v < 0)
            v = I.cols + v;
        while (v > I.cols - 1)
            v = v - I.cols;
        u = I.rows - u - 1;
        result[i * 3] = I.at<cv::Vec3b>(u, v)[2];
        result[i * 3 + 1] = I.at<cv::Vec3b>(u, v)[1];
        result[i * 3 + 2] = I.at<cv::Vec3b>(u, v)[0];
    }
    delete[] corr;
}
*/

void InnerLoopCorr_uv(cv::Mat I, float* uv, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int* faces_uv_m, int Nb_Faces_m, unsigned char* result, int id) {
    float* corr = new float[2];
    for (int i = id; i < min(Nb_Vertices_s, id + Nb_Vertices_s / 100); i++) {
        my_float3 p0 = make_my_float3(vertices_s[3 * i], vertices_s[3 * i + 1], vertices_s[3 * i + 2]);
        my_float3 ray = make_my_float3(normals_s[3 * i], normals_s[3 * i + 1], normals_s[3 * i + 2]);
        // Compute the smallest distance to the faces
        float min_dist = 1.0e32f;
        int closest_point = 0;
        for (int f = 0; f < Nb_Faces_m; f++) {
            my_float3 n = make_my_float3(normals_faces_m[3 * f], normals_faces_m[3 * f + 1], normals_faces_m[3 * f + 2]);
            if (dot(n, ray) < 0.8f)
                continue;
            my_float3 p1 = make_my_float3(vertices_m[3 * faces_m[3 * f]], vertices_m[3 * faces_m[3 * f] + 1], vertices_m[3 * faces_m[3 * f] + 2]);
            my_float3 p2 = make_my_float3(vertices_m[3 * faces_m[3 * f + 1]], vertices_m[3 * faces_m[3 * f + 1] + 1], vertices_m[3 * faces_m[3 * f + 1] + 2]);
            my_float3 p3 = make_my_float3(vertices_m[3 * faces_m[3 * f + 2]], vertices_m[3 * faces_m[3 * f + 2] + 1], vertices_m[3 * faces_m[3 * f + 2] + 2]);
            //cout << "Color => " << color[3 * faces_m[3 * f]] << " " << color[3 * faces_m[3 * f + 1]] << " " << color[3 * faces_m[3 * f + 2]] << endl;
            // Compute point to face distance
            float curr_dist = DistancePointFace3D(p0, p1, p2, p3, n);
            //float curr_dist = sqrt(dot(p0-p1,p0-p1));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                //closest_point = faces_m[3*f];
                UVFromFace3D(corr, &uv[2 * faces_uv_m[3 * f]], &uv[2 * faces_uv_m[3 * f + 1]], &uv[2 * faces_uv_m[3 * f + 2]], p0, p1, p2, p3, n);
            }
        }
        int u = int(corr[1] * I.rows);
        if (u < 0)
            u = I.rows + u;
        while (u > I.rows - 1)
            u = u - I.rows;
        int v = int(corr[0] * I.cols);
        if (v < 0)
            v = I.cols + v;
        while (v > I.cols - 1)
            v = v - I.cols;
        u = I.rows - u - 1;
        result[i * 3] = I.at<cv::Vec3b>(u, v)[2];
        result[i * 3 + 1] = I.at<cv::Vec3b>(u, v)[1];
        result[i * 3 + 2] = I.at<cv::Vec3b>(u, v)[0];
    }
    delete[] corr;
}



float* GetCorrespondences(float* skin_weights, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m) {
    float* result = new float[24 * Nb_Vertices_s];
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoop2(skin_weights, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, Nb_Faces_m, result, i * (Nb_Vertices_s / 100));
        my_threads.push_back(std::thread(InnerLoop2, skin_weights, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, Nb_Faces_m, result, i * (Nb_Vertices_s / 100)));
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    return result;

}

unsigned char* GetCorrespondencesColor(unsigned char* input_color, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int Nb_Faces_m) {
    unsigned char* result_color = new unsigned char[3 * Nb_Vertices_s];

    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        if (i % 10 == 0) {
            cout << "thread : " << i << endl;
        }
        InnerLoopColor(input_color, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, Nb_Faces_m, result_color, i * (Nb_Vertices_s / 100));
        //InnerLoop(volume, i, vertices, faces, normals, nb_faces, size_grid, center_grid, res, disp);
        //my_threads.push_back(std::thread(InnerLoop, skin_weights, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, Nb_Faces_m, result, i * (Nb_Vertices_s / 100)));
    }
    //std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

    return result_color;

}

/*
unsigned char* GetCorrespondences_uv(cv::Mat I, float* uv, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int* faces_uv_m, int Nb_Faces_m) {
    unsigned char* result = new unsigned char[3 * Nb_Vertices_s];
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoopCorr_uv(I, uv, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, faces_uv_m, Nb_Faces_m , result, i * (Nb_Vertices_s / 100));
        my_threads.push_back(std::thread(InnerLoopCorr_uv, I, uv, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, faces_uv_m, Nb_Faces_m, result, i * (Nb_Vertices_s / 100)));
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    return result;
}
*/

unsigned char* GetCorrespondences_uv(cv::Mat I, float* uv, float* vertices_s, float* normals_s, int Nb_Vertices_s, float* vertices_m, float* normals_faces_m, int* faces_m, int* faces_uv_m, int Nb_Faces_m) {
    unsigned char* result = new unsigned char[3 * Nb_Vertices_s];
    std::vector< std::thread > my_threads;
    for (int i = 0; i < 101; i++) {
        //InnerLoopCorr_uv(I, uv, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, faces_uv_m, Nb_Faces_m, result, i * (Nb_Vertices_s / 100));
        my_threads.push_back(std::thread(InnerLoopCorr_uv, I, uv, vertices_s, normals_s, Nb_Vertices_s, vertices_m, normals_faces_m, faces_m, faces_uv_m, Nb_Faces_m, result, i * (Nb_Vertices_s / 100)));
    }
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
    return result;
}


void GenerateOuterShell(string template_dir) {
    cout << "==============Build the user specific Outer shell==================" << endl;
    
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    float *vertices;
    float *normals_face;
    int *faces;
    //int *res = LoadPLY_Mesh(folder_name + string("meshes_centered/mesh_0.ply"), &vertices, &normals_face, &faces);
    int *res = LoadPLY_Mesh(template_dir + string("/Template.ply"), &vertices, &normals_face, &faces);
    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
    
    /**
            2. Transform the mesh into the T-shape
     */
    // Load skinning weights
    float *skin_weights = new float[24*res[0]];
    
    //ifstream SWfile (folder_name+"weights_smpl.bin", ios::binary);
    ifstream SWfile (template_dir+"/TshapeWeights.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }

    SWfile.read((char *) skin_weights, 24*res[0]*sizeof(float));
    SWfile.close();
    
    // load kinematic table
    ifstream KTfile (template_dir+"/kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int *kinTree_Table = new int[48];
    KTfile.read((char *) kinTree_Table, 48*sizeof(int));
    KTfile.close();
    
    // Load the Skeleton
    DualQuaternion *skeleton = new DualQuaternion[24];
    
    Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir+"/Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.2f);
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    //LoadSkeleton(folder_name+"Tshapecoarsejoints.bin", folder_name+"smplparams_centered/skeleton_0.bin", Joints, kinTree_Table);
    LoadSkeleton(template_dir+"/Tshapecoarsejoints.bin", template_dir+"/TshapeSMPLjoints.bin", Joints, kinTree_Table);
    
    float *skeletonPC = new float[3*24];
    for (int s = 0; s < 24; s++) {
        //cout << Joints[s] << endl;
        skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
        skeletonPC[3*s] = Joints[s](0,3);
        skeletonPC[3*s+1] = Joints[s](1,3);
        skeletonPC[3*s+2] = Joints[s](2,3);
    }
    
    SavePCasPLY(output_mesh_folder_name + string("skel_in.ply"), skeletonPC, 24);
    delete []skeletonPC;
    
    // Transform the mesh to the star pose
    SkinMeshDBS(vertices, skin_weights, skeleton, res[0]);
    
    UpdateNormalsFaces(vertices, normals_face, faces, res[1]);
    
    SaveMeshToPLY(output_mesh_folder_name + string("skin.ply"), vertices, vertices, faces, res[0], res[1]);
    
    /**
            3. Build the level-set in a coarse canonical grid
     */
    
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f,-0.1f,0.0f);
    //my_float3 center_grid = make_my_float3(0.0f,1.0f,-0.5f);
    float ***tsdf = LevelSet(vertices, faces, normals_face, res[1], size_grid, center_grid, GRID_RES, 0.1f);
    //float ***tsdf = LevelSet_sphere(make_my_float3(0.0f, 0.0f, 0.0f), 2.0f, size_grid, 0.1f);
    cout << "The level set is generated" << endl;
    
    /**
            4. Create tetrahedral mesh
     */
    
    float *nodes_out;
    int *tetra_out;
    int *res_tetra = TetraFromGrid(tsdf, &nodes_out, &tetra_out, 0.0f, size_grid, center_grid, GRID_RES) ;
    
    /**
            5. Extract the 0-level set with Marching Tets
     */
    int *faces_out;
    float *surface_out;
    int *res_surf = SurfaceFromTetraGrid(tsdf, &surface_out, &faces_out, 0.0f, size_grid, center_grid, GRID_RES) ;
    //SaveMeshToPLY(output_mesh_folder_name + string("shell.ply"), surface_out, surface_out, faces_out, res_surf[0], res_surf[1]);
    
    /**
            6. Reassign faces
     */
    int nb_faces_out = ReorganizeSurface(surface_out, faces_out, nodes_out, res_surf[0], res_surf[1], res_tetra[0]);
    
    /**
            5. Extract the 0-level set with Marching Cubes
     */
    
    /*float *vertices_out;
    float *normals_out;
    int *faces_out;
    int *res_out = MarchingCubes(tsdf, &vertices_out, &normals_out, &faces_out, size_grid, center_grid, GRID_RES, 0.0f);
    // Save 3D mesh to the disk for visualization
    SaveMeshToPLY(output_mesh_folder_name + string("shell.ply"), vertices_out, normals_out, faces_out, res_out[0], res_out[1]);*/
    
    /**
            6. Link the surface to the tetrahedras
     */
    /*int *surface_edges;
    LinkSurfaceToTetra(&surface_edges, vertices_out, nodes_out, res_out[0], res_tetra[0]);*/
    
    // Save tetrahedral mesh to disk
    //SaveTetraMeshToPLY(output_mesh_folder_name + string("TetraShell.ply"), vertices_out, normals_out, faces_out, nodes_out, surface_edges, tetra_out, res_tetra[0], res_out[0], res_tetra[1], res_out[0], res_out[1]);
    
    SaveTetraMeshToPLY(template_dir + string("/TetraShell.ply"), nodes_out, nodes_out, faces_out, tetra_out, res_tetra[0], res_tetra[1], nb_faces_out, GRID_RES, 0.1f);
    
            
    delete []vertices;
    delete []normals_face;
    delete []faces;
    delete []skin_weights;
    delete []skeleton;
    delete []kinTree_Table;
    delete []Joints_T;
    delete []Joints;
    
    //delete []vertices_out;
    //delete []normals_out;
    delete []faces_out;
    delete []surface_out;
    
    delete []nodes_out;
    delete []tetra_out;
    delete []res_tetra;
    
    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete []tsdf[i][j];
        }
        delete []tsdf[i];
    }
    delete []tsdf;
}

void EmbedScanIntoOuterShell(string filename, string filename_pose) {
    cout << "==============Embed the scan into the user specific Outer shell==================" << endl;
    
    /**
            0. Load the outer shell
     */
    //TetraMesh *TetMesh = new TetraMesh(output_mesh_folder_name + string("TshapeCoarseTetraD.ply"));
    TetraMesh *TetMesh = new TetraMesh(output_mesh_folder_name + string("FittedTetraShell.ply"));
    
    /**
            1. Load the  3D scan of the sequence (it should be a centered ply file)
     */
    float *vertices;
    float *normals_face;
    int *faces;
    int *res = LoadPLY_Mesh(folder_name + filename, &vertices, &normals_face, &faces);
    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
    
    /**
            2. Transform the mesh into the T-shape
     */
    // Load skinning weights
    float *skin_weights = new float[24*res[0]];
    
    ifstream SWfile (folder_name+"weights_smpl.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }

    SWfile.read((char *) skin_weights, 24*res[0]*sizeof(float));
    SWfile.close();
    
    // load kinematic table
    ifstream KTfile (folder_name+"kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int *kinTree_Table = new int[48];
    KTfile.read((char *) kinTree_Table, 48*sizeof(int));
    KTfile.close();
    
    // Load the Skeleton
    DualQuaternion *skeleton = new DualQuaternion[24];
    
    Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(folder_name+"Tshapecoarsejoints.bin", Joints_T, kinTree_Table);
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(folder_name+"Tshapecoarsejoints.bin", folder_name+filename_pose, Joints, kinTree_Table);
    
    float *skeletonPC = new float[3*24];
    for (int s = 0; s < 24; s++) {
        //cout << Joints[s] << endl;
        skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
        skeletonPC[3*s] = Joints[s](0,3);
        skeletonPC[3*s+1] = Joints[s](1,3);
        skeletonPC[3*s+2] = Joints[s](2,3);
    }
    
    SavePCasPLY(output_mesh_folder_name + string("skel_in.ply"), skeletonPC, 24);
    delete []skeletonPC;
    
    // Transform the mesh to the star pose
    SkinMeshDBS(vertices, skin_weights, skeleton, res[0]);
    
    UpdateNormalsFaces(vertices, normals_face, faces, res[1]);
    
    SaveMeshToPLY(output_mesh_folder_name + string("skin.ply"), vertices, vertices, faces, res[0], res[1]);
    
    /**
            3. Build the level-set in the outer-shell
     */
    float *vol_weights;
    float *tsdf = TetMesh->LevelSet(vertices, faces, normals_face, res[0], res[1]);
    
    /**
            4. Extract the 3D mesh
     */
    float *VerticesTet = new float[3*TetMesh->NBEdges()];
    float *SkinWeightsTet = new float[24*TetMesh->NBEdges()];
    int *FacesTet = new int[3*3*TetMesh->NBTets()];
    int *res_tet = TetMesh->MT(tsdf, vol_weights, VerticesTet, SkinWeightsTet, FacesTet);
        
    SaveColoredMeshToPLY(output_mesh_folder_name + string("EmbeddedMesh.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
    
    /**
            5. Re-pose the 3D mesh
     */
    
    //transferSkinWeights(VerticesTet, SkinWeightsTet, vertices, skin_weights, res_tet[0], res[0]);
    
    // Transform the mesh from the star pose back to the original pose
    DualQuaternion *skeletonBack = new DualQuaternion[24];
    for (int s = 0; s < 24; s++) {
        skeletonBack[s] = skeleton[s].Inv(skeleton[s]);
    }
    
    //SkinMeshDBS(vertices, skin_weights, skeletonBack, res[0]);
    //SaveMeshToPLY(output_mesh_folder_name + string("skinBack.ply"), vertices, faces, res[0], res[1]);
    
    SkinMeshDBS(VerticesTet, SkinWeightsTet, skeletonBack, res_tet[0]);
    SaveColoredMeshToPLY(output_mesh_folder_name + string("EskinBack.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
    delete []skeletonBack;
    
    
    
    delete TetMesh;
    delete []VerticesTet;
    delete []SkinWeightsTet;
    delete []FacesTet;
    
    delete []vertices;
    delete []normals_face;
    delete []faces;
    delete []skin_weights;
    delete []skeleton;
    delete []kinTree_Table;
    delete []Joints_T;
    delete []Joints;
    
    delete []tsdf;
    delete []vol_weights;
}

void CreateLevelSetTetraTSDF(string output_dir, string input_dir, string template_dir, int idx_file) {
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadPLY_Mesh(input_dir + "meshes_centered/mesh_" + to_string(idx_file) + ".ply", &vertices, &normals_face, &faces);
    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;

    /**
            2. Transform the mesh into the T-shape
     */
    // Load skinning weights
    float* skin_weights = new float[24 * res[0]];

    ifstream SWfile(template_dir + "weights_smpl.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }

    SWfile.read((char*)skin_weights, 24 * res[0] * sizeof(float));
    SWfile.close();

    // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];

    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.2f);//, 0.1f);
    //LoadTSkeleton(template_dir+"Tshapecoarsejoints.bin", Joints_T, kinTree_Table);                                                            // <=======================

    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    string idx_s = to_string(idx_file);
    idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
    LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_dir + "smplparams_centered/skeleton_" + to_string(idx_file) + ".bin", Joints, kinTree_Table); // <=======================
    //LoadSkeleton(template_dir+"Tshapecoarsejoints.bin", input_dir + "smplparams/pose_" + idx_s + ".bin", Joints, kinTree_Table);
    //LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_dir + "smplparams/pose_" + to_string(idx_file) + ".bin", Joints, kinTree_Table);// For 
    // <=========== Here is to load fitted skeleton
    //LoadSkeleton(template_dir+"Tshapecoarsejoints.bin", input_dir + "Fit/pose.bin", Joints, kinTree_Table);
    // <=========== Here is to load Vibe skeleton
    //LoadSkeletonVIBE(template_dir+"Tshapecoarsejoints.bin", input_dir+"smplparams_vibe/", Joints, kinTree_Table, idx_file);

    float* skeletonPC = new float[3 * 24];
    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
        skeletonPC[3 * s] = Joints[s](0, 3);
        skeletonPC[3 * s + 1] = Joints[s](1, 3);
        skeletonPC[3 * s + 2] = Joints[s](2, 3);
    }

#ifdef VERBOSE
    SavePCasPLY(output_dir + string("skel_in.ply"), skeletonPC, 24);
#endif

    delete[]skeletonPC;

    SkinMeshDBS(vertices, skin_weights, skeleton, res[0]);

    UpdateNormalsFaces(vertices, normals_face, faces, res[1]);

#ifdef VERBOSE
    SaveMeshToPLY(output_dir + string("skin.ply"), vertices, vertices, faces, res[0], res[1]);
#endif

     /**
             3.0. Load the outer shell
      */
    TetraMesh* TetMesh = new TetraMesh(template_dir, 0.022f); //0.011f); // GRID_RES); // <==================================================
    
    /**
        4. Build the level-set in the outer-shell
    */
    float* fine_sdf = TetMesh->LevelSet(vertices, faces, normals_face, res[0], res[1]);
    SaveLevelSet(output_dir + "fineLevelSet_" + to_string(idx_file) + ".bin", fine_sdf, TetMesh->NBNodes());
    cout << "The fine level set is generated" << endl;

    /**
            5. Extract the 3D mesh
     */
     //#ifdef VERBOSE
    float* VerticesTet = new float[3 * TetMesh->NBEdges()];
    float* SkinWeightsTet = new float[24 * TetMesh->NBEdges()];
    int* FacesTet = new int[3 * 3 * TetMesh->NBTets()];
    int* res_tet = TetMesh->MT(fine_sdf, &fine_sdf[TetMesh->NBNodes()], VerticesTet, SkinWeightsTet, FacesTet);

    SaveColoredMeshToPLY(output_dir + string("Output_Mesh_" + to_string(idx_file) + ".ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);


    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }

    SkinMeshDBS(VerticesTet, SkinWeightsTet, skeleton, res_tet[0]);
    SaveColoredMeshToPLY(output_dir + string("Skinned_Output_Mesh_" + to_string(idx_file) + ".ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);


    delete[]VerticesTet;
    delete[]SkinWeightsTet;
    delete[]FacesTet;
    delete[]res_tet;
    //#endif

    delete[] fine_sdf;
    delete[]vertices;
    delete[]normals_face;
    delete[]faces;
    delete[]skin_weights;
    delete[]skeleton;
    delete[]kinTree_Table;
    delete[]Joints_T;
    delete[]Joints;

    delete TetMesh;

    return;
}

void CreateLevelSet(string output_dir, string input_dir, string template_dir, int idx_file) {
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    float *vertices;
    float *normals_face;
    int *faces;
    int *res = LoadPLY_Mesh(input_dir + "meshes_centered/mesh_" + to_string(idx_file) + ".ply", &vertices, &normals_face, &faces);
    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
    
    /**
            2. Transform the mesh into the T-shape
     */
    // Load skinning weights
    float *skin_weights = new float[24*res[0]];
    
    ifstream SWfile (template_dir+"weights_smpl.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }

    SWfile.read((char *) skin_weights, 24*res[0]*sizeof(float));
    SWfile.close();
    
    // load kinematic table
    ifstream KTfile (template_dir+"kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int *kinTree_Table = new int[48];
    KTfile.read((char *) kinTree_Table, 48*sizeof(int));
    KTfile.close();
    
    // Load the Skeleton
    DualQuaternion *skeleton = new DualQuaternion[24];
    
    Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.2f);//, 0.1f);
    //LoadTSkeleton(template_dir+"Tshapecoarsejoints.bin", Joints_T, kinTree_Table);                                                            // <=======================
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    string idx_s = to_string(idx_file);
    idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
    LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_dir + "smplparams_centered/skeleton_" + to_string(idx_file) + ".bin", Joints, kinTree_Table); // <=======================
    //LoadSkeleton(template_dir+"Tshapecoarsejoints.bin", input_dir + "smplparams/pose_" + idx_s + ".bin", Joints, kinTree_Table);
    //LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_dir + "smplparams/pose_" + to_string(idx_file) + ".bin", Joints, kinTree_Table);// For 
    // <=========== Here is to load fitted skeleton
    //LoadSkeleton(template_dir+"Tshapecoarsejoints.bin", input_dir + "Fit/pose.bin", Joints, kinTree_Table);
    // <=========== Here is to load Vibe skeleton
    //LoadSkeletonVIBE(template_dir+"Tshapecoarsejoints.bin", input_dir+"smplparams_vibe/", Joints, kinTree_Table, idx_file);
    
    Eigen::Matrix4f Root = Joints[6].inverse();
    float *skeletonPC = new float[3*24];
    for (int s = 0; s < 24; s++) {
        //Joints[s] = Root * Joints[s];                                                                                                                          // <=======================
        //cout << Joints[s] << endl;
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
        //skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
        skeletonPC[3*s] = Joints[s](0,3);
        skeletonPC[3*s+1] = Joints[s](1,3);
        skeletonPC[3*s+2] = Joints[s](2,3);
    }
    
    #ifdef VERBOSE
        SavePCasPLY(output_dir + string("skel_in.ply"), skeletonPC, 24);
    #endif
    
    delete []skeletonPC;
    
    /** If root coordinates are available, center the mesh**/
    // load center joint
    /*ifstream Rootfile (input_dir + "Fit/root.bin", ios::binary);
    if (Rootfile.is_open()) {
        float *root = new float[3];
        Rootfile.read((char *) root, 3*sizeof(float));
        Rootfile.close();
        cout << "root joint: " << root[0] << ", " << root[1] << ", " << root[2] << endl;
        CenterMesh(vertices, root, res[0]);
    }*/
    
    // Transform the outershell to the mesh pose

#ifdef BASIC_TETRA
    /*SkinMeshDBS(vertices, skin_weights, skeleton, res[0]);
    
    UpdateNormalsFaces(vertices, normals_face, faces, res[1]);*/
#endif
    
    #ifdef VERBOSE
        SaveMeshToPLY(output_dir + string("skin.ply"), vertices, vertices, faces, res[0], res[1]);
    #endif
    
    /**
            3. Fit the outershell
     */
        /**
                3.0. Load the outer shell
         */
        TetraMesh* TetMesh = new TetraMesh(template_dir, 0.015f); //0.011f); // GRID_RES); // <==================================================
    
        // Transform the outershell to the mesh pose
        TetMesh->Skin(skeleton);
    
        #ifdef VERBOSE
            SaveTetraMeshToPLY(output_dir + string("SkinnedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());
        #endif
    
        /**
                3.1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
         */
        my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
        my_float3 center_grid = make_my_float3(0.0f,-0.1f,0.0f);
        //my_float3 center_grid = make_my_float3(0.0f,-0.3f,0.0f);
        
        float*** coarse_sdf = LevelSet_gpu(vertices, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES, 0.1f);
    
        /*float ***coarse_sdfB = SDFFromLevelSet(coarse_sdf, size_grid, center_grid, GRID_RES);*/
    
///////////////////////////
        /*float *vertices_tet_out;
        float *normals_tet_out;
        int *faces_tet_out;
        int *res_tet_out = MarchingCubes(coarse_sdf, &vertices_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES, 0.0f);
        // Save 3D mesh to the disk for visualization
        SaveMeshToPLY(output_mesh_folder_name + string("coarseB.ply"), vertices_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
        return;*/
///////////////////////////
  
    
        // save the level set to file
        //SaveLevelSet(output_dir + "coarseLevelSet_" + to_string(idx_file) + ".bin", coarse_sdf, size_grid);
        //cout << "The coarse level set is generated" << endl;
        
        #ifndef BASIC_TETRA
            cout << "Fitting to sdf grid" << endl;
            TetMesh->FitToTSDF(coarse_sdf, size_grid, center_grid, GRID_RES, 0.0f);//, 20, 1, 10, 0.01f, 0.1f);
        #endif
    
        #ifdef VERBOSE
            SaveTetraMeshToPLY(output_dir + string("FittedTetraShellB.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());
            
            SavePCasPLY(output_dir + string("FittedPC.ply"), TetMesh->Nodes(), TetMesh->NBNodes());
        #endif
        
        for (int i = 0; i < size_grid.x; i++) {
            for (int j = 0; j < size_grid.y; j++) {
                delete []coarse_sdf[i][j];
                //delete []coarse_sdfB[i][j];
            }
            delete []coarse_sdf[i];
            //delete []coarse_sdfB[i];
        }
        delete []coarse_sdf;
        //delete []coarse_sdfB;
    
    /**
            4. Build the level-set in the outer-shell
     */
    float *fine_sdf = TetMesh->LevelSet(vertices, faces, normals_face, res[0], res[1]);
    SaveLevelSet(output_dir + "fineLevelSet_" + to_string(idx_file) + ".bin", fine_sdf, TetMesh->NBNodes());
    cout << "The fine level set is generated" << endl;
    
    
    /**
            5. Extract the 3D mesh
     */
    //#ifdef VERBOSE
        float *VerticesTet = new float[3*TetMesh->NBEdges()];
        float *SkinWeightsTet = new float[24*TetMesh->NBEdges()];
        int *FacesTet = new int[3*3*TetMesh->NBTets()];
        int *res_tet = TetMesh->MT(fine_sdf, &fine_sdf[TetMesh->NBNodes()], VerticesTet, SkinWeightsTet, FacesTet);
            
        SaveColoredMeshToPLY(output_dir + string("Output_Mesh_" + to_string(idx_file) + ".ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
    
        delete []VerticesTet;
        delete []SkinWeightsTet;
        delete []FacesTet;
        delete []res_tet;
    //#endif
    
    delete[] fine_sdf;    
    delete []vertices;
    delete []normals_face;
    delete []faces;
    delete []skin_weights;
    delete []skeleton;
    delete []kinTree_Table;
    delete []Joints_T;
    delete []Joints;
    
    delete TetMesh;
    
    return;
}

void CreateCoarseLevelSet(string output_dir, string input_dir, string template_dir, int idx_file, bool ORIENT = false) {
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadPLY_Mesh(input_dir + "meshes_centered/mesh_" + to_string(idx_file) + ".ply", &vertices, &normals_face, &faces);
    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;

    if (ORIENT) {
        ifstream posefile(input_dir + "poses/template_poses.txt");
        string line;
        if (!posefile.is_open()) {
            cout << "Error could not open: " << input_dir + "poses/template_poses.txt" << endl;
            return;
        }
        else {
            getline(posefile, line);
            getline(posefile, line);
            for (int id = 0; id < idx_file; id++)
                getline(posefile, line);
        }

        getline(posefile, line);
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        float angles[3] = { std::stof(words[0]) , std::stof(words[2]) , std::stof(words[1]) };
        Eigen::Matrix3f Rot = rodrigues2matrix(angles);
        Eigen::Matrix4f Root;
        Root << Rot(0, 0), Rot(0, 1), Rot(0, 2), 0.0,
            Rot(1, 0), Rot(1, 1), Rot(1, 2), 0.0,
            Rot(2, 0), Rot(2, 1), Rot(2, 2), 0.0,
            0.0, 0.0, 0.0, 1.0;
        //Eigen::Matrix4f Root = Joints[6].inverse();

        for (int v = 0; v < res[0]; v++) {
            if (vertices[3 * v] == 0.0f && vertices[3 * v + 1] == 0.0f && vertices[3 * v + 2] == 0.0f)
                continue;
            Eigen::Vector4f vtx = Eigen::Vector4f(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2], 1.0f);
            vtx = Root * vtx;
            vertices[3 * v] = vtx(0);
            vertices[3 * v + 1] = vtx(1);
            vertices[3 * v + 2] = vtx(2);
        }

        posefile.close();
    }

#ifdef VERBOSE
    SaveMeshToPLY(output_dir + string("skin_" + to_string(idx_file) + ".ply"), vertices, vertices, faces, res[0], res[1]);
#endif

    /**
            3.1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f, -0.1f, 0.0f);
    //my_float3 center_grid = make_my_float3(0.0f,-0.3f,0.0f);

    float*** coarse_sdf = LevelSet_gpu(vertices, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES, 0.1f);


#ifdef VERBOSE
        float *vertices_tet_out;
        float *normals_tet_out;
        int *faces_tet_out;
        int *res_tet_out = MarchingCubes(coarse_sdf, &vertices_tet_out, &normals_tet_out, &faces_tet_out, size_grid, center_grid, GRID_RES, 0.0f);
        // Save 3D mesh to the disk for visualization
        SaveMeshToPLY(output_dir + string("coarse_" + to_string(idx_file) + ".ply"), vertices_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
        delete[] vertices_tet_out;
        delete[] normals_tet_out;
        delete[] faces_tet_out;
        delete[] res_tet_out;
#endif

    // save the level set to file
    SaveLevelSet(output_dir + "coarseLevelSet_" + to_string(idx_file) + ".bin", coarse_sdf, size_grid);
    cout << "The coarse level set is generated" << endl;

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete[]coarse_sdf[i][j];
        }
        delete[]coarse_sdf[i];
    }
    delete[]coarse_sdf;

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;

    return;
}

void FitGridToOuterShell() {
    cout << "==============Build the user specific Outer shell and Fit the grid to it==================" << endl;
    
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    //my_float3 center_grid = make_my_float3(0.0f,-0.1f,0.0f);
    my_float3 center_grid = make_my_float3(0.0f,1.0f,-0.5f);
    
    //float ***tsdf = CreateLevelSet();
    float ***tsdf = LoadLevelSet(output_mesh_folder_name + string("levelset.bin"), size_grid);
    cout << "The level set is generated" << endl;
    
    /****for test*/
    /*float *vertices_out;
    float *normals_out;
    int *faces_out;
    int *res_out = MarchingCubes(tsdf, &vertices_out, &normals_out, &faces_out, size_grid, center_grid, GRID_RES, 0.0f);
    
    // Save 3D mesh to the disk for visualization
    SaveMeshToPLY(output_mesh_folder_name + string("shell.ply"), vertices_out, normals_out, faces_out, res_out[1], res_out[0]);
    
    delete []vertices_out;
    delete []normals_out;
    delete []faces_out;*/
    
    /**
            4. Fit the canonical grid to the level set
     */
    
    CubicMesh *OuterShell = new CubicMesh(make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE), center_grid, 0.018);//GRID_RES);
    OuterShell->FitToTSDF(tsdf, size_grid, center_grid, GRID_RES);
    
    // Save 3D mesh to the disk for visualization
    SaveCubicMeshToPLY(output_mesh_folder_name + string("FittedShell.ply"), OuterShell->Voxels(), OuterShell->Dim());
                
    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete []tsdf[i][j];
        }
        delete []tsdf[i];
    }
    delete []tsdf;
}

void FitTetraOuterShell() {
    cout << "==============Build the user specific Outer shell and Fit the outershell to it==================" << endl;
    
    /**
            0. Load the outer shell
     */
    TetraMesh *TetMesh = new TetraMesh(output_mesh_folder_name + string("TetraShell.ply"), GRID_RES);
    
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f,-0.1f,0.0f);
    //my_float3 center_grid = make_my_float3(0.0f,1.0f,-0.5f);
    
    //float ***tsdf = CreateLevelSet();
    float ***tsdf = LoadLevelSet(output_mesh_folder_name + string("levelset.bin"), size_grid);
    cout << "The level set is generated" << endl;
    
    TetMesh->FitToTSDF(tsdf, size_grid, center_grid, GRID_RES);
    
    //SaveMeshToPLY(output_mesh_folder_name + string("FittedSurfaceShell.ply"), TetMesh->Vertices(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->NBVertices(), TetMesh->NBFaces());
    SaveTetraMeshToPLY(output_mesh_folder_name + string("FittedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());
    
    SavePCasPLY(output_mesh_folder_name + string("FittedPC.ply"), TetMesh->Nodes(), TetMesh->NBNodes());
    
    
    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete []tsdf[i][j];
        }
        delete []tsdf[i];
    }
    delete []tsdf;
    
    delete TetMesh;
}

void FitTetraOuterShellFromOut(string output_dir, string input_dir, string template_dir, int idx_file) {
    cout << "==============Build the user specific Outer shell and Fit the outershell to it==================" << endl;
    
    /**
            0. Load the outer shell
     */
    TetraMesh *TetMesh = new TetraMesh(template_dir, GRID_RES);

    float* vertices_in;
    float* normals_face_in;
    int* faces_in;
    unsigned char* color_in;
    //int* res_in = LoadPLY_Mesh(input_dir + "meshes_centered/mesh_" + to_string(idx_file) + ".ply", &vertices_in, &normals_face_in, &faces_in);
    std::stringstream in_idx_name;
    in_idx_name << std::setfill('0') << right << setw(4) << to_string(idx_file + 1);
    int* res_in = LoadPLY_Mesh_Color(input_dir + "canonical_meshes/mesh_" + in_idx_name.str() + ".ply", &vertices_in, &normals_face_in, &faces_in, &color_in);
    //cout << "The PLY file has " << res_in[0] << " vertices and " << res_in[1] << " faces" << endl;

    //repose process from posed to T_pose
    
    // Load skinning weights
    float* skin_weights = new float[24 * res_in[0]];

    ifstream SWfile(input_dir + "meshes_centered/weights_smpl.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }

    SWfile.read((char*)skin_weights, 24 * res_in[0] * sizeof(float));
    SWfile.close();

    // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];

    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.5f);

    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_dir + "smplparams_centered/skeleton_" + to_string(idx_file) +".bin", Joints, kinTree_Table);

    float* skeletonPC = new float[3 * 24];
    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
    }

    delete[]skeletonPC;

    SkinMeshDBS(vertices_in, skin_weights, skeleton, res_in[0]);
    UpdateNormalsFaces(vertices_in, normals_face_in, faces_in, res_in[1]);
    SaveMeshToPLY(input_dir + "Meshskin/meshskin_" + to_string(idx_file) + ".ply", vertices_in, vertices_in, faces_in, res_in[0], res_in[1]);
    

    // ====================== VERSION CUBE
    /*my_int3 size_grid = make_my_int3(512, 512, 512);
    my_float3 center_grid = make_my_float3(0.0f, -0.1f, 0.0f);
    float*** coarse_sdf = LevelSet_gpu(vertices_in, faces_in, normals_face_in, res_in[0], res_in[1], size_grid, center_grid, 0.005f, 0.0f);
    TetMesh->FitToTSDF(coarse_sdf, size_grid, center_grid, 0.005f, 0.0f);//, 20, 1, 10, 0.01f, 0.1f);

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete[]coarse_sdf[i][j];
        }
        delete[]coarse_sdf[i];
    }
    delete[]coarse_sdf;*/
    
    /**
            1. Load the Level Set
     */
    //TetMesh->LoadLevelSet(template_dir + string("levelset.bin"));
    /*float *vol_weights;
    float *sdf = new float[TetMesh->NBNodes()];
    LoadLevelSet(template_dir + string("fineLevelSet_0000.bin"), sdf, &vol_weights, TetMesh->NBNodes());
    cout << "The level set is generated" << endl;*/
    /*for (int n = 0; n < TetMesh->NBNodes(); n++) {
        cout << sdf[n] << endl;
    }*/

    /*float* VerticesTet = new float[3 * TetMesh->NBEdges()];
    float* SkinWeightsTet = new float[24 * TetMesh->NBEdges()];
    int* FacesTet = new int[3 * 3 * TetMesh->NBTets()];
    int* res_tet = TetMesh->MT(sdf, NULL, VerticesTet, SkinWeightsTet, FacesTet);

    SaveMeshToPLY(template_dir + string("Output_Mesh.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);*/
    
    /**
            2. Load the Template Mesh
     */
    float *vertices;
    float *normals;
    int *faces;
    int *res = TetMesh->GetSurface(&vertices, &normals, &faces);
    cout << "The PLY file has " << res[0] << " surfaces and " << res[1] << " faces" << endl;
    
    //SaveMeshToPLY(template_dir + string("skin.ply"), vertices, normals, faces, res[0], res[1]);
    
    //cout << "Finish save skin" << endl;


    // ====================== VERSION TETRA
    /**
            3. Fit the mesh to the TSDF
     */
    float* sdf = TetMesh->LevelSet(vertices_in, faces_in, normals_face_in, res_in[0], res_in[1]);
    fitLevelSet::FitToLevelSet(sdf, TetMesh->Nodes(), TetMesh->Tetras(), TetMesh->NBTets(), vertices, normals, faces, res[0], res[1]);   

    /*  #use color
    cout << "finish fitting & start getting color" << endl;
    unsigned char* result_color = GetCorrespondencesColor(color_in, vertices, normals, res[0], vertices_in, normals_face_in, faces_in, res_in[1]);
    cout << +result_color[0] << "," << +result_color[1] << "," << +result_color[2] << "," << endl;
    SaveMeshToPLYColor(output_dir + "FittedMesh_" + to_string(idx_file) +".ply" , vertices, vertices, faces, result_color, res[0], res[1]);
    */

    //SaveMeshToPLY(output_dir + "FittedMesh_" + to_string(idx_file) + ".ply", vertices, vertices, faces, res[0], res[1]);
    //cout << "Fitted mesh saved" << endl;

    delete[] sdf;

    /*
    cout << "from here new scripts" << endl;
    //Iwamoto-san data
    float* vertices_GT;
    float* normals_face_GT;
    float* uv_GT;
    int* faces_GT;
    int* faces_uv_GT;
    std::stringstream idx_obj;
    idx_obj << std::setfill('0') << right << setw(6) << to_string(idx_file+1);
    int* res_GT = LoadOBJ_Mesh_uv(input_dir + "OBJ_data/" + "Data1_" + idx_obj.str() + ".obj", &vertices_GT, &normals_face_GT, &uv_GT, &faces_GT, &faces_uv_GT);
    cout << "The PLY file has " << res_GT[0] << " vertices and " << res_GT[1] << " faces" << endl;
    // Load texture image
    // <==== Load texture image
    //cv::Mat I = cv::imread(path_scan + scan_name + idx_name + "/" + scan_name + idx_name + ".jpg", -1);
    std::stringstream idx_name;
    idx_name << std::setfill('0') << right << setw(5) << to_string(idx_file+1);
    cv::Mat I = cv::imread(input_dir + "OBJ_data/" + "tex." + idx_name.str() + ".png", -1);
    if (I.empty())
    {
        std::cout << "!!! Failed imread(): texture image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
        exit(3);
    }
    cout << "Texture size: " << I.rows << ", " << I.cols << endl;
    cout << "Pixel size: " << I.channels() << ", " << I.elemSize() << ", " << I.elemSize1() << ", " << I.step << endl;
    cout << "GetCorrespondences" << endl;
    //unsigned char* colors = GetCorrespondences(colors_GT, vertices, normals, res[0], vertices_GT, normals_face_GT, faces_GT, res_GT[1]);

    unsigned char* colors = GetCorrespondences_uv(I, uv_GT, vertices, normals, res[0], vertices_GT, normals_face_GT, faces_GT, faces_uv_GT, res_GT[1]);
    SaveMeshToPLYColor(output_dir + "FittedMesh_" + to_string(idx_file)+ ".ply", vertices, normals, faces, colors, res[0], res[1]);


    delete[] vertices;
    delete[] normals;
    delete[] faces;
    delete TetMesh;
    */
}

void SaveUVSkin_weight(float* weights, int Nb_Vertices_uvskin, string save_path) {
    ofstream weight_file(save_path, ios::binary);
    if (weight_file.is_open()) {
        weight_file.write((char*)weights, 24 * Nb_Vertices_uvskin * sizeof(float));
        weight_file.close();
    }
}

void FitTetraToCanonicalFromOut(string output_dir, string input_dir, string template_dir, int idx_file, bool volume_fit = false, bool use_fixed_weights = true) {
    cout << "==============Build the user specific Outer shell and Fit the outershell to it==================" << endl;
    string idx_s = to_string(idx_file);
    idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0'); //4
    string idx_s_pls1 = to_string(idx_file+1);
    idx_s_pls1.insert(idx_s_pls1.begin(), 4 - idx_s_pls1.length(), '0'); //4 _ +1
    //string idx_s2 = to_string(idx_file);
    //idx_s2.insert(idx_s2.begin(), 2 - idx_s2.length(), '0'); //2
    string idx_p = to_string(idx_file);
    idx_p.insert(idx_p.begin(), 3 - idx_p.length(), '0'); //3
    //string idx_p2_pls1 = to_string(idx_file+1);
    //idx_p2_pls1.insert(idx_p2_pls1.begin(), 2 - idx_p2_pls1.length(), '0'); //2
    string idx_obj = to_string(idx_file + 1);
    idx_obj.insert(idx_obj.begin(), 6 - idx_obj.length(), '0'); //5
    //string idx_obj2 = to_string(idx_file + 1);
    //idx_obj2.insert(idx_obj2.begin(), 2 - idx_obj2.length(), '0'); //2
    string idx_name = to_string(idx_file + 1);
    idx_name.insert(idx_name.begin(), 5 - idx_name.length(), '0');
    //string idx_name2 = to_string(idx_file + 1);
    //idx_name2.insert(idx_name2.begin(), 2 - idx_name2.length(), '0'); //2
    string idx_name3 = to_string(idx_file + 1);
    idx_name3.insert(idx_name3.begin(), 3 - idx_name3.length(), '0'); //3

    /**
            0. Load the outer shell
     */
    TetraMesh* TetMesh = new TetraMesh(template_dir, GRID_RES);

    /**
            1. Load the scan in canonical pose and its corresponding skin weights
     */
    cout << "==== Step 1.1 :   Load the scan in canonical pose ====" << endl;
    float* vertices_in;
    float* normals_face_in;
    int* faces_in;
    int* res_in = LoadPLY_Mesh(input_dir + "data/canonical_meshes/mesh_" + idx_s + ".ply", &vertices_in, &normals_face_in, &faces_in);
    //int* res_in = LoadPLY_Mesh(input_dir + "data/canonical_meshes/mesh_" + idx_s_pls1 + ".ply", &vertices_in, &normals_face_in, &faces_in);
    cout << "The PLY file has " << res_in[0] << " vertices and " << res_in[1] << " faces" << endl;

    cout << "==== Step 1.2 :  Load skinning weights ====" << endl;
    // Load skinning weights    
    float* skin_weights = new float[24 * res_in[0]];
    ifstream SWfile(input_dir + "data/canonical_meshes/skin_weights_" + idx_s + ".bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }
    SWfile.read((char*)skin_weights, 24 * res_in[0] * sizeof(float));
    SWfile.close();

    /**
            2. Fit the outer shell to the canonical mesh
    */
    cout << "==== Step 2.1 :  Fit the outer shell to the canonical mesh ====" << endl; float* vertices;
    // ====================== VERSION TETRA
    // Embed the 3D mesh into tetra sdf field

    float* normals;
    float* weights;
    int* faces;
    int* res;
    if (volume_fit) {
        my_int3 size_grid = make_my_int3(256, 256, 256);
        my_float3 center_grid = make_my_float3(0.0f, -0.1f, 0.0f);

        float*** tsdf = LevelSet_gpu(vertices_in, faces_in, normals_face_in, res_in[0], res_in[1], size_grid, center_grid, 0.01f, 0.0f);
        cout << "The level set is generated" << endl;

        TetMesh->FitToTSDF(tsdf, size_grid, center_grid, 0.01f);

        // Get surface vertices
        res = TetMesh->GetSurface(&vertices, &normals, &weights, &faces);

        for (int i = 0; i < size_grid.x; i++) {
            for (int j = 0; j < size_grid.y; j++) {
                delete[]tsdf[i][j];
            }
            delete[]tsdf[i];
        }
        delete[]tsdf;
    }
    else {
        // Get surface vertices
        res = TetMesh->GetSurface(&vertices, &normals, &weights, &faces);

        // Compute radial basis functions
        float* RBF = new float[16 * res[0]];
        int* RBF_id = new int[16 * res[0]];
        for (int n = 0; n < res[0]; n++) {
            for (int i = 0; i < 16; i++)
                RBF_id[16 * n + i] = -1;
        }
        fitLevelSet::ComputeRBF(RBF, RBF_id, vertices, faces, res[1], res[0]);
        cout << "The PLY file has " << res[0] << " surfaces and " << res[1] << " faces" << endl;

        float* sdf = TetMesh->LevelSet(vertices_in, faces_in, normals_face_in, res_in[0], res_in[1]);

        // Fit the 3D mesh to the tetra sdf field
        fitLevelSet::FitToLevelSet(sdf, TetMesh->Nodes(), TetMesh->Tetras(), RBF, RBF_id, TetMesh->NBTets(), vertices, normals, faces, res[0], res[1], 30, 10, 0.01f, 0.003f);

        delete[] RBF;
        delete[] RBF_id;

        delete[] sdf;
    }

    cout << "==== Step 2.2 : Assign skinning weights from T-posed scan ====" << endl;
    float* weights_c;
    if (use_fixed_weights) {
        // Load skinning weights    
        weights_c = new float[24 * res[0]];
        ifstream WCfile(output_dir + "fitted_skin_weights.bin", ios::binary);
        if (!WCfile.is_open()) {
            cout << "Could not load skin weights" << endl;
            WCfile.close();
            return;
        }
        WCfile.read((char*)weights_c, 24 * res[0] * sizeof(float));
        WCfile.close();
    }
    else {
        weights_c = GetCorrespondences(skin_weights, vertices, normals, res[0], vertices_in, normals_face_in, faces_in, res_in[1]);

        // Save the assigned weights
        ofstream WCfile(output_dir + "fitted_skin_weights_" + idx_s + ".bin", ios::binary);
        if (!WCfile.is_open()) {
            cout << "Could not open " << output_dir + "fitted_skin_weights_" + idx_s + ".bin" << endl;
            WCfile.close();
            return;
        }
        WCfile.write((char*)weights_c, 24 * res[0] * sizeof(float));
        WCfile.close();
    }

    /**
            3. Load pose data and re-pose the fitted outershell with corresponding weights
    */
    cout << "==== Step 3.1 : Load pose data ====" << endl;

    // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton    
    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(input_dir + "data/smplparams_centered/T_joints.bin", Joints_T, kinTree_Table, 0.5f);
    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(input_dir + "data/smplparams_centered/T_joints.bin", input_dir + "data/smplparams_centered/pose_" + idx_p + ".bin", Joints, kinTree_Table);


    Eigen::Matrix4f* skeleton = new Eigen::Matrix4f[24]();
    for (int s = 0; s < 24; s++) {
        //skeleton[s] = Joints[s] * InverseTransfo(Joints_T[s]);      //
        skeleton[s] = Joints_T[s] * InverseTransfo(Joints[s]);    //old_lbs new_lbs(inverse = true)
    }

    cout << "==== Step 3.2 : Skin the mesh ====" << endl;
    //  Save a T-pose copy
    float* vertices_cpy = new float[3 * res[0]];
    memcpy(vertices_cpy, vertices, 3 * res[0] * sizeof(float));
    float* normals_cpy = new float[3 * res[0]];
    memcpy(normals_cpy, normals, 3 * res[0] * sizeof(float));

    //SkinMeshLBS_old(vertices, weights, skeleton, res[0]);
    SkinMeshLBS(vertices, weights_c, skeleton, res[0], true);

    //update normal vectors
    UpdateNormals(vertices, normals, faces, res[0], res[1]);

    SaveMeshToPLY(output_dir + "PosedMesh_" + idx_s + ".ply", vertices, normals, faces, res[0], res[1]);

    /**
            4. Load texture image and original scan
    */
    cout << "==== Step 4.1 : Load scan ====" << endl;
    float* vertices_GT;
    float* normals_face_GT;
    float* uv_GT;
    int* faces_GT;
    int* faces_uv_GT;
    int* res_GT = LoadOBJ_Mesh_uv(input_dir + "data/centered_meshes/" + "mesh" + idx_s + ".obj", &vertices_GT, &normals_face_GT, &uv_GT, &faces_GT, &faces_uv_GT);
    cout << "The PLY file has " << res_GT[0] << " vertices and " << res_GT[2] << " faces" << endl;

    cout << "==== Step 4.2 : Load texture ====" << endl;
    // <==== Load texture image
    cv::Mat I = cv::imread(input_dir + "OBJ_data/" + "tex." + idx_name + ".png", -1);
    //cv::Mat I = cv::imread(input_dir + "textures/" + "Female_ballet_dancer_" + idx_name3 + ".jpg", -1);
    if (I.empty())
    {
        std::cout << "!!! Failed imread(): texture image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
        exit(3);
    }
    cout << "Texture size: " << I.rows << ", " << I.cols << endl;
    cout << "Pixel size: " << I.channels() << ", " << I.elemSize() << ", " << I.elemSize1() << ", " << I.step << endl;

    /**
            4. Get color value from uv interpolation
    */
    cout << "==== Step 5.1 : Read color ====" << endl;
    unsigned char* colors = GetCorrespondences_uv(I, uv_GT, vertices, normals, res[0], vertices_GT, normals_face_GT, faces_GT, faces_uv_GT, res_GT[2]);
    SaveMeshToPLY(output_dir + "FittedMesh_" + idx_s + ".ply", vertices_cpy, normals_cpy, colors, faces, res[0], res[1]);

    delete[] vertices_in;
    delete[] normals_face_in;
    delete[] faces_in;
    delete[] res_in;
    delete[] skin_weights;
    delete[] vertices;
    delete[] normals;
    delete[] weights;
    delete[] faces;
    delete[] res;
    delete[] kinTree_Table;
    delete[] Joints_T;
    delete[] Joints;
    delete[] skeleton;
    delete[] weights_c;

    cout << "==== DONE ====" << endl;

    delete[] vertices_GT;
    delete[] normals_face_GT;
    delete[] uv_GT;
    delete[] faces_GT;
    delete[] faces_uv_GT;
    delete[] res_GT;

    delete[] vertices_cpy;
    delete[] normals_cpy;
    delete[] colors;
}


float* GetUVskinWeight(string template_dir,string uvskin_path) {
    ///run only firsttime 
    //Load the outer shell
    TetraMesh* TetMesh = new TetraMesh(template_dir, GRID_RES);

    //Load the uvskin mesh
    float* vertices_UVSKIN;
    float* normals_face_UVSKIN;
    int* faces_UVSKIN;
    int* res_UVSKIN = LoadOBJ_Mesh(uvskin_path, &vertices_UVSKIN, &normals_face_UVSKIN, &faces_UVSKIN);

    cout << "Vertices : " << res_UVSKIN[0] << " Faces : " << res_UVSKIN[1] << endl;

    //Load the template mesh
    float* vertices_SMPL;
    float* normals_face_SMPL;
    int* faces_SMPL;
    int* res_SMPL = LoadPLY_Mesh(template_dir + "Template_skin.ply", &vertices_SMPL, &normals_face_SMPL, &faces_SMPL);
    cout << "Vertices : " << res_SMPL[0] << " Faces : " << res_SMPL[1] << endl;

    //Fit the mesh to the TSDF
    float* sdf = TetMesh->LevelSet(vertices_SMPL, faces_SMPL, normals_face_SMPL, res_SMPL[0], res_SMPL[1]);     //make sdf of template_mesh
    fitLevelSet::FitToLevelSet(sdf, TetMesh->Nodes(), TetMesh->Tetras(), TetMesh->NBTets(), vertices_UVSKIN, normals_face_UVSKIN, faces_UVSKIN, res_UVSKIN[0], res_UVSKIN[1]);    //Fit uvskin to template_mesh(sdf)
    SaveMeshToPLY(template_dir + "FittedMesh_uvskin.ply", vertices_UVSKIN, vertices_UVSKIN, faces_UVSKIN, res_UVSKIN[0], res_UVSKIN[1]);

    // Load skinning weights
    float* skin_weights = new float[24 * res_SMPL[0]];

    ifstream SWfile(template_dir + "TshapeWeights.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        //return -1;
    }

    SWfile.read((char*)skin_weights, 24 * res_SMPL[0] * sizeof(float));
    SWfile.close();

    // <=== find corresponding skinning weight from template to uvskin
    float* weights = GetCorrespondences(skin_weights, vertices_UVSKIN, normals_face_UVSKIN, res_UVSKIN[0], vertices_SMPL, normals_face_SMPL, faces_SMPL, res_SMPL[1]);
    return weights;
}

void Repose2Tshape( string output_dir , string input_dir ,string template_dir, float* weights, int idx_file) {
    std::stringstream idx_i;
    std::stringstream idx_j;
    idx_i << std::setfill('0') << right << setw(6) << to_string(idx_file); 
    idx_j << std::setfill('0') << right << setw(6) << to_string(idx_file+1); 

    //Load the fitted mesh
    float* vertices_in;
    float* normals_face_in;
    int* faces_in;
    int* res_in = LoadOBJ_Mesh_retopo(input_dir + "retopo/mesh_" + idx_j.str() + ".obj", &vertices_in, &normals_face_in, &faces_in);
    if (idx_file == 0) {
        SaveMeshToPLY("D:/Data/Human/Template-star-0.015/debug_retopo.ply", vertices_in, vertices_in, faces_in, res_in[0], res_in[1]);

    }

    // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];


    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.5f);



    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_dir + "smplparams_centered/skeleton_" + to_string(idx_file) + ".bin", Joints, kinTree_Table);


    float* skeletonPC = new float[3 * 24];
    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
    }

    delete[]skeletonPC;

    SkinMeshDBS(vertices_in, weights, skeleton, res_in[0]);
    UpdateNormalsFaces(vertices_in, normals_face_in, faces_in, res_in[1]);
    SaveMeshToPLY(output_dir + "mesh_" + idx_i.str() + ".ply", vertices_in, vertices_in, faces_in, res_in[0], res_in[1]);
}

void retopo_reconst(string output_dir, string input_dir, string template_dir, float* weights, int idx_file) {
    std::stringstream idx_i;
    std::stringstream idx_j;
    idx_i << std::setfill('0') << right << setw(6) << to_string(idx_file);
    idx_j << std::setfill('0') << right << setw(6) << to_string(idx_file + 1);

    //Load the fitted mesh
    float* vertices_in;
    float* normals_face_in;
    int* faces_in;
    //int* res_in = LoadOBJ_Mesh_retopo(input_dir + "retopo_tpose/mesh_" + idx_i.str() + ".obj", &vertices_in, &normals_face_in, &faces_in);
    int* res_in = LoadPLY_Mesh(input_dir + "retopo_tpose/mesh_" + idx_i.str() + ".ply", &vertices_in, &normals_face_in, &faces_in);


    // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];


    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.5f);



    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_dir + "smplparams_centered/skeleton_" + to_string(idx_file) + ".bin", Joints, kinTree_Table);


    float* skeletonPC = new float[3 * 24];
    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }

    delete[]skeletonPC;

    SkinMeshDBS(vertices_in, weights, skeleton, res_in[0]);
    UpdateNormalsFaces(vertices_in, normals_face_in, faces_in, res_in[1]);
    SaveMeshToPLY(output_dir + "mesh_" + idx_i.str() + ".ply", vertices_in, vertices_in, faces_in, res_in[0], res_in[1]);
}

void PoseFromT_pose(string output_dir, string input_mesh_basepath, string input_skeleton_basepath, string template_dir, float* weights, int idx_file , bool color_flg) {
        
    //Load the fitted mesh
    float* vertices_in;
    float* normals_face_in;
    int* faces_in;
    unsigned char* color_in;
    int* res_in;

    std::stringstream idx_i;
    idx_i << std::setfill('0') << right << setw(4) << to_string(idx_file);
    if (color_flg == true) {
        res_in = LoadPLY_Mesh_Color(input_mesh_basepath + to_string(idx_file) + ".ply", &vertices_in, &normals_face_in, &faces_in , &color_in);
        //res_in = LoadPLY_Mesh_Color(input_mesh_basepath + idx_i.str() + ".ply", &vertices_in, &normals_face_in, &faces_in, &color_in);    //GT
    }
    else {
        //res_in = LoadPLY_Mesh(input_mesh_basepath + to_string(idx_file) + ".ply", &vertices_in, &normals_face_in, &faces_in);
        res_in = LoadPLY_Mesh(input_mesh_basepath + idx_i.str() + ".ply", &vertices_in, &normals_face_in, &faces_in);   //GT
    }

    // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(input_skeleton_basepath + "T_joints.bin", Joints_T, kinTree_Table, 0.5f);
    //LoadStarSkeleton(input_skeleton_basepath + "T_joints.bin", Joints_T, kinTree_Table, 0.5f);    //New

    std::stringstream idx_p;
    idx_p << std::setfill('0') << right << setw(3) << to_string(idx_file);
    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    //LoadSkeleton(input_skeleton_basepath + "T_joints.bin", input_skeleton_basepath + to_string(idx_file) + ".bin", Joints, kinTree_Table);
    //LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", input_skeleton_basepath + idx_p.str() + ".bin", Joints, kinTree_Table);
    LoadSkeleton(input_skeleton_basepath + "T_joints.bin", input_skeleton_basepath + idx_p.str() + ".bin", Joints, kinTree_Table);  //New

    
    // Dual blend skining
    
    // Load the Skeleton
    /*
    DualQuaternion* skeleton = new DualQuaternion[24];
    float* skeletonPC = new float[3 * 24];
    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }
    delete[]skeletonPC;
    SkinMeshDBS(vertices_in, weights, skeleton, res_in[0]);
    */

    
    // Linear blend skining
    
    DualQuaternion skeleton[24];

    Eigen::Matrix4f* T_skel = new Eigen::Matrix4f[24]();
    for (int s = 0; s < 24; s++) {
       // T_skel[s] = Joints[s] * InverseTransfo(Joints_T[s]);       //old_lbs   
        T_skel[s] = Joints_T[s] * InverseTransfo(Joints[s]);          //new_lbs(inverse=True) 
    }
    //SkinMeshLBS_old(vertices_in, weights, T_skel, res_in[0]);
    SkinMeshLBS(vertices_in, weights, T_skel, res_in[0], true);
    
    
    UpdateNormalsFaces(vertices_in, normals_face_in, faces_in, res_in[1]);
    if (color_flg == true) {
        SaveMeshToPLY(output_dir + "mesh_" + to_string(idx_file) + ".ply", vertices_in, vertices_in, color_in, faces_in, res_in[0], res_in[1]);
    }
    else {
        SaveMeshToPLY(output_dir + "mesh_" + to_string(idx_file) + ".ply", vertices_in, vertices_in, faces_in, res_in[0], res_in[1]);
    }
}


void SmplPosechange(string template_dir) {
    //Load the template mesh
    float* vertices_SMPL;
    float* normals_face_SMPL;
    int* faces_SMPL;
    int* res_SMPL = LoadPLY_Mesh(template_dir + "Template_skin.ply", &vertices_SMPL, &normals_face_SMPL, &faces_SMPL);
    cout << "Vertices : " << res_SMPL[0] << " Faces : " << res_SMPL[1] << endl;

    // Load skinning weights
    float* skin_weights = new float[24 * res_SMPL[0]];

    //ifstream SWfile(template_dir + "TshapeWeights.bin", ios::binary);
    ifstream SWfile(template_dir + "weights_smpl.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }

    SWfile.read((char*)skin_weights, 24 * res_SMPL[0] * sizeof(float));
    SWfile.close();


    // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];

    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.5f);
    //LoadStarSkeleton(template_dir + "TshapeSMPLjoints.bin", Joints_T, kinTree_Table, 0.5f);

    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", "D:/Data/Human/ARTICULATED/I_jumping/smplparams_centered/skeleton_100.bin", Joints, kinTree_Table);
    //LoadSkeleton(template_dir + "TshapeSMPLjoints.bin", "D:/Data/Human/ARTICULATED/I_jumping/smplparams_centered/skeleton_22.bin", Joints, kinTree_Table);


    float* skeletonPC = new float[3 * 24];
    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }

    SkinMeshDBS(vertices_SMPL, skin_weights, skeleton, res_SMPL[0]);
    UpdateNormalsFaces(vertices_SMPL, normals_face_SMPL, faces_SMPL, res_SMPL[1]);
    SaveMeshToPLY(template_dir + "smpl_posed.ply", vertices_SMPL, vertices_SMPL, faces_SMPL, res_SMPL[0], res_SMPL[1]);

    cout << "Finish!!" << endl;
}



float* LoadUVSkin_weight(int Nb_Vertices_uvskin, string path) {
    // Load skinning weights
    float* weights = new float[24 * Nb_Vertices_uvskin ];

    ifstream weight_file(path, ios::binary);
    if (!weight_file.is_open()) {
        cout << "Could not load skin weights" << endl;
        cout << path << endl;
        weight_file.close();
        //return -1;
    }

    weight_file.read((char*)weights, 24 * Nb_Vertices_uvskin * sizeof(float));
    weight_file.close();

    return weights;
    }

void GetPosedResult(string output_dir, string input_dir , string smpl_dir , int file_num , string color_flg ) {
    float* weights;
    bool color_flg_bool;
    if (color_flg == "True") {
        color_flg_bool = true;
    }
    else {
        color_flg_bool = false;
    }

    //weights = GetUVskinWeight("D:/Data/Human/Template-star-0.015/", "D:/Project/Human/Pose2Texture/make_displacement/uvskin.obj");
    //SaveUVSkin_weight(weights, 35010, "D:/Data/Human/Template-star-0.015/uvskin_weight.bin");
    weights = LoadUVSkin_weight(35010, "F:/Template-star-0.015/uvskin_weight.bin");
    for (int i = 2; i < file_num; i++) {
    //for (int i = 110; i < 111; i++) {
        cout << "data : " + to_string(i) << endl;
        //PoseFromT_pose(output_dir, input_dir + "reconst_Tpose_", smpl_dir + "pose_", "F:/Template-star-0.015/", weights, i , color_flg_bool);
        //PoseFromT_pose(output_dir, input_dir + "FittedMesh_", smpl_dir + "skeleton_", "D:/Data/Human/Template-star-0.015/", weights, i , color_flg_bool); //GT
        PoseFromT_pose(output_dir, input_dir + "FittedMesh_", smpl_dir + "pose_", "F:/Template-star-0.015/", weights, i, color_flg_bool); //GT

    }
}

void GetPosedResult(string output_dir, string input_dir, string smpl_dir, int file_num, string color_flg ,string inputSW_dir) {
    float* weights;
    bool color_flg_bool;
    if (color_flg == "True") {
        color_flg_bool = true;
    }
    else {
        color_flg_bool = false;
    }


    //weights = GetUVskinWeight("D:/Data/Human/Template-star-0.015/", "D:/Project/Human/Pose2Texture/make_displacement/uvskin.obj");
    //SaveUVSkin_weight(weights, 35010, "D:/Data/Human/Template-star-0.015/uvskin_weight.bin");
    for (int i = 2; i < file_num; i++) {
        std::stringstream idx_i;
        cout << "data : " + to_string(i) << endl;
        idx_i << std::setfill('0') << right << setw(4) << to_string(i);
        weights = LoadUVSkin_weight(35010, inputSW_dir + "skin_weights_" + idx_i.str() + ".bin");
        PoseFromT_pose(output_dir, input_dir + "reconst_Tpose_", smpl_dir + "pose_", "D:/Data/Human/Template-star-0.015/", weights, i, color_flg_bool);
        //PoseFromT_pose(output_dir, input_dir + "FittedMesh_", smpl_dir + "skeleton_", "D:/Data/Human/Template-star-0.015/", weights, i , true);
    }
}


void BuildAdjacencies(string template_dir) {
    TetraMesh *TetMesh = new TetraMesh(template_dir, GRID_RES);
    
    int rates[4] = {4,2,2,2};
    TetMesh->ComputeAdjacencies(template_dir, rates, 5);
    
    delete TetMesh;
}

void BuildDataSet() {
    for (int i = 0; i < 250; i++) {
        CreateLevelSet("/Volumes/HUMANDATA/Articulated/D_march/VolumetricData/",
                       "/Users/diegothomas/Documents/Data/ARTICULATED/D_march/",
                       "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template/", i);
    }
    
    for (int i = 0; i < 175; i++) {
        CreateLevelSet("/Volumes/HUMANDATA/Articulated/D_bouncing/VolumetricData/",
                       "/Users/diegothomas/Documents/Data/ARTICULATED/D_bouncing/",
                       "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template/", i);
    }
    
    for (int i = 0; i < 175; i++) {
        CreateLevelSet("/Volumes/HUMANDATA/Articulated/I_crane/VolumetricData/",
                       "/Users/diegothomas/Documents/Data/ARTICULATED/I_crane/",
                       "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template/", i);
    }
    
    for (int i = 0; i < 150; i++) {
        CreateLevelSet("/Volumes/HUMANDATA/Articulated/I_jumping/VolumetricData/",
                       "/Users/diegothomas/Documents/Data/ARTICULATED/I_jumping/",
                       "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template/", i);
    }
}

void Generate3DModelTetraTSDF( string fine_tsdf_path, string skeleton_path, string template_dir, string output_dir = output_mesh_folder_name) {
    cout << "============= =Generate the 3D mesh from the level sets==================" << endl;
    /**
            0. Load the outer shell
     */
    TetraMesh* TetMesh = new TetraMesh(template_dir, 0.008f);//GRID_RES);


    /**
            0.1 Re-pose
     */
     // load kinematic table
    ifstream KTfile(template_dir + "/kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];

    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "/Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.2f);

    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "/Tshapecoarsejoints.bin", skeleton_path, Joints, kinTree_Table);

    Eigen::Matrix4f Root = Joints[6].inverse();                                                                                                                      // <=======================
    for (int s = 0; s < 24; s++) {
        //Joints[s] = Root * Joints[s];           // if use cape 
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }

    /*
            2. load the fine level set
     */
    float* tsdf_fine = LoadTetLevelSet(fine_tsdf_path, TetMesh->NBNodes());
    cout << "The level set is generated" << endl;
    /*float *tsdf_fine_GT = LoadTetLevelSet("/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/tmp/fineLevelSet_0.bin", TetMesh->NBNodes());

    for (int i = 0; i < TetMesh->NBNodes(); i++) {
        cout << "tsdf: " << tsdf_fine[i] << "<---->" << tsdf_fine_GT[i] << endl;
    }*/

    /**
            3. Extract the 3D mesh
     */
    float* VerticesTet = new float[3 * TetMesh->NBEdges()];
    float* SkinWeightsTet = new float[24 * TetMesh->NBEdges()];
    int* FacesTet = new int[3 * 3 * TetMesh->NBTets()];
    int* res_tet = TetMesh->MT(tsdf_fine, NULL, VerticesTet, SkinWeightsTet, FacesTet);

    //SaveMeshToPLY(output_dir + string("Output_Mesh.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);

    SkinMeshDBS(VerticesTet, SkinWeightsTet, skeleton, res_tet[0]);

    SaveColoredMeshToPLY(output_dir /*+ string("Output_Mesh_posed.ply")*/, VerticesTet, FacesTet, res_tet[0], res_tet[1]);

    delete TetMesh;
    delete[]VerticesTet;
    delete[]SkinWeightsTet;
    delete[]FacesTet;
    delete[]tsdf_fine;
}

void Generate3DModelSkinned(string fine_tsdf_path, string skeleton_path, string template_dir, string output_dir = output_mesh_folder_name) {
    cout << "==============Generate the 3D mesh from the level sets==================" << endl;
    /**
            0. Load the outer shell
     */
    TetraMesh* TetMesh = new TetraMesh(template_dir, 0.008f);//GRID_RES);
    cout << "oK" << endl;
    //Load the template mesh
    float* vertices_SMPL;
    float* normals_face_SMPL;
    int* faces_SMPL;
    int* res_SMPL = LoadPLY_Mesh(template_dir + "Template_skin.ply", &vertices_SMPL, &normals_face_SMPL, &faces_SMPL);
    cout << "The Template mesh has " << res_SMPL[0] << " vertices and " << res_SMPL[1] << " faces" << endl;

    // Load skinning weights
    float* skin_weights = new float[24 * res_SMPL[0]];

    ifstream SWfile(template_dir + "TshapeWeights.bin", ios::binary);
    if (!SWfile.is_open()) {
        cout << "Could not load skin weights" << endl;
        SWfile.close();
        return;
    }

    SWfile.read((char*)skin_weights, 24 * res_SMPL[0] * sizeof(float));
    SWfile.close();

    /**
            0.1 Re-pose
     */
     // load kinematic table
    ifstream KTfile(template_dir + "/kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];

    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadASkeleton(template_dir + "Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.2f);//, 0.1f);

    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "/Tshapecoarsejoints.bin", skeleton_path, Joints, kinTree_Table);

    Eigen::Matrix4f Root = Joints[6].inverse();                                                                                                                      // <=======================
    for (int s = 0; s < 24; s++) {
        //Joints[s] = Root * Joints[s];           // if use cape 
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }

    // Transform the template to the mesh pose
    SkinMeshDBS(vertices_SMPL, skin_weights, skeleton, res_SMPL[0]);
    UpdateNormalsFaces(vertices_SMPL, normals_face_SMPL, faces_SMPL, res_SMPL[1]);

#ifdef VERBOSE
    SaveMeshToPLY(output_dir + string("skin.ply"), vertices_SMPL, vertices_SMPL, faces_SMPL, res_SMPL[0], res_SMPL[1]);
#endif

    // Transform the outershell to the mesh pose
    TetMesh->Skin(skeleton);

#ifdef VERBOSE
    SaveTetraMeshToPLY(output_dir + string("SkinnedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());
#endif

    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f, -0.1f, 0.0f);
    //my_float3 center_grid = make_my_float3(0.0f,-0.3f,0.0f);

    float*** coarse_sdf = LevelSet_gpu(vertices_SMPL, faces_SMPL, normals_face_SMPL, res_SMPL[0], res_SMPL[1], size_grid, center_grid, GRID_RES, 0.11f);

    cout << "The coarse level set is generated" << endl;
    cout << "Fitting to sdf grid" << endl;

    TetMesh->FitToTSDF(coarse_sdf, size_grid, center_grid, GRID_RES, 0.0f);//, 20, 1, 10, 0.01f, 0.1f);

#ifdef VERBOSE
    SaveTetraMeshToPLY(output_dir + string("FittedTetraShellB.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());
#endif

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete[]coarse_sdf[i][j];
            //delete []coarse_sdfB[i][j];
        }
        delete[]coarse_sdf[i];
        //delete []coarse_sdfB[i];
    }
    delete[]coarse_sdf;

    /*
            2. load the fine level set
     */
    float* tsdf_fine = LoadTetLevelSet(fine_tsdf_path, TetMesh->NBNodes());
    cout << "The level set is generated" << endl;
    /*float *tsdf_fine_GT = LoadTetLevelSet("/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/tmp/fineLevelSet_0.bin", TetMesh->NBNodes());

    for (int i = 0; i < TetMesh->NBNodes(); i++) {
        cout << "tsdf: " << tsdf_fine[i] << "<---->" << tsdf_fine_GT[i] << endl;
    }*/

    /**
            3. Extract the 3D mesh
     */
    float* VerticesTet = new float[3 * TetMesh->NBEdges()];
    float* SkinWeightsTet = new float[24 * TetMesh->NBEdges()];
    int* FacesTet = new int[3 * 3 * TetMesh->NBTets()];
    int* res_tet = TetMesh->MT(tsdf_fine, NULL, VerticesTet, SkinWeightsTet, FacesTet);

    SaveColoredMeshToPLY(output_dir, VerticesTet, FacesTet, res_tet[0], res_tet[1]);

    delete TetMesh;
    delete[]VerticesTet;
    delete[]SkinWeightsTet;
    delete[]FacesTet;
    delete[]tsdf_fine;
}

void Generate3DModel(string coarse_tsdf_path, string fine_tsdf_path, string skeleton_path, string template_dir, string output_dir = output_mesh_folder_name) {
    cout << "=============  =Generate the 3D mesh from the level sets==================" << endl;
    cout << "New!New!" << endl;
    /**
            0. Load the outer shell
     */
    TetraMesh* TetMesh = new TetraMesh(template_dir, 0.015f);//GRID_RES);
    
    /**
            0.1 Re-pose the outershell
     */
    // load kinematic table
    ifstream KTfile (template_dir+"/kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int *kinTree_Table = new int[48];
    KTfile.read((char *) kinTree_Table, 48*sizeof(int));
    KTfile.close();
    
    // Load the Skeleton
    DualQuaternion *skeleton = new DualQuaternion[24];
    
    Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir+"/Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.2f);
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir+"/Tshapecoarsejoints.bin", skeleton_path, Joints, kinTree_Table);
    
    Eigen::Matrix4f Root = Joints[6].inverse();                                                                                                                      // <=======================
    for (int s = 0; s < 24; s++) {
        //Joints[s] = Root * Joints[s];           // if use cape 
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }
    
    TetMesh->Skin(skeleton);
    SaveTetraMeshToPLY(output_dir + string("SkinnedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());
    
    /**
            1. Load the coarse level set and fit the outer shell to it
     */
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_int3 size_grid_64 = make_my_int3(64, 64, 64);
    my_float3 center_grid = make_my_float3(0.0f,-0.1f,0.0f);
    
    float ***tsdf = LoadLevelSet(coarse_tsdf_path, size_grid_64);
    cout << "The level set is generated" << endl;

    float* vertices_tet_out;
    float* normals_tet_out;
    int* faces_tet_out;
    int* res_tet_out = MarchingCubes(tsdf, &vertices_tet_out, &normals_tet_out, &faces_tet_out, size_grid_64, center_grid, 0.04f, 0.0f);
    // Save 3D mesh to the disk for visualization
    SaveMeshToPLY(output_dir + string("coarse.ply"), vertices_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);
    delete[] vertices_tet_out;
    delete[] normals_tet_out;
    delete[] faces_tet_out;
    delete[] res_tet_out;
    
    cout << size_grid_64.x << endl;
    cout << size_grid_64.y << endl;
    for (int i = 0; i < size_grid_64.x; i++) {
        for (int j = 0; j < size_grid_64.y; j++) {
            delete[]tsdf[i][j];
        }
        
        delete[]tsdf[i];
    }
    delete[]tsdf;
    float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadPLY_Mesh(output_dir + "coarse.ply", &vertices, &normals_face, &faces);

    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;

    tsdf = LevelSet_gpu(vertices, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES, 0.0f);

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;

    
    TetMesh->FitToTSDF(tsdf, size_grid, center_grid, GRID_RES, 0.0f);

    SaveTetraMeshToPLY(output_dir + string("FittedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());
            
    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete []tsdf[i][j];
        }
        delete []tsdf[i];
    }
    delete []tsdf;
    
    /*
            2. load the fine level set
     */
    float *tsdf_fine = LoadTetLevelSet(fine_tsdf_path, TetMesh->NBNodes());
    cout << "The level set is generated" << endl;
    /*float *tsdf_fine_GT = LoadTetLevelSet("/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/tmp/fineLevelSet_0.bin", TetMesh->NBNodes());
    
    for (int i = 0; i < TetMesh->NBNodes(); i++) {
        cout << "tsdf: " << tsdf_fine[i] << "<---->" << tsdf_fine_GT[i] << endl;
    }*/
    
    /**
            3. Extract the 3D mesh
     */
    float *VerticesTet = new float[3*TetMesh->NBEdges()];
    float *SkinWeightsTet = new float[24*TetMesh->NBEdges()];
    int *FacesTet = new int[3*3*TetMesh->NBTets()];
    int *res_tet = TetMesh->MT(tsdf_fine, NULL, VerticesTet, SkinWeightsTet, FacesTet);
        
    SaveColoredMeshToPLY(output_dir + string("Output_Mesh.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
    
    delete TetMesh;
    delete []VerticesTet;
    delete []SkinWeightsTet;
    delete []FacesTet;
    delete []tsdf_fine;
}

void GenerateCoarse3DModel(string coarse_tsdf_path, string coarse_tsdfGT_path, string skeleton_path,string template_dir ,string output_dir = output_mesh_folder_name) {
    cout << "=============  =Generate the 3D mesh from the level sets==================" << endl;
    cout << "only coarse" << endl;
    /**
            0. Load the outer shell
     */
    TetraMesh* TetMesh = new TetraMesh(template_dir, 0.015f);//GRID_RES);

    /**
            0.1 Re-pose the outershell
     */
     // load kinematic table
    ifstream KTfile(template_dir + "/kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    // Load the Skeleton
    DualQuaternion* skeleton = new DualQuaternion[24];

    Eigen::Matrix4f* Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir + "/Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.2f);

    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "/Tshapecoarsejoints.bin", skeleton_path, Joints, kinTree_Table);

    Eigen::Matrix4f Root = Joints[6].inverse();                                                                                                                      // <=======================
    for (int s = 0; s < 24; s++) {
        //Joints[s] = Root * Joints[s];           // if use cape 
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    }

    TetMesh->Skin(skeleton);
    SaveTetraMeshToPLY(output_dir + string("SkinnedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());

    /**
            1. Load the coarse level set and fit the outer shell to it
     */
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_int3 size_grid_64 = make_my_int3(64, 64, 64);
    my_float3 center_grid = make_my_float3(0.0f, -0.1f, 0.0f);

    float*** tsdf = LoadLevelSet(coarse_tsdf_path, size_grid_64);
    float*** tsdfGT = LoadLevelSet(coarse_tsdfGT_path, size_grid_64);
    cout << "The level set is generated" << endl;

    float* vertices_tet_out;
    float* normals_tet_out;
    int* faces_tet_out;
    int* res_tet_out = MarchingCubes(tsdf, &vertices_tet_out, &normals_tet_out, &faces_tet_out, size_grid_64, center_grid, 0.04f, 0.0f);
    // Save 3D mesh to the disk for visualization
    SaveMeshToPLY(output_dir + string("coarse.ply"), vertices_tet_out, normals_tet_out, faces_tet_out, res_tet_out[0], res_tet_out[1]);

    int* res_tetGT_out = MarchingCubes(tsdfGT, &vertices_tet_out, &normals_tet_out, &faces_tet_out, size_grid_64, center_grid, 0.04f, 0.0f);
    SaveMeshToPLY(output_dir + string("coarseGT.ply"), vertices_tet_out, normals_tet_out, faces_tet_out, res_tetGT_out[0], res_tetGT_out[1]);
    delete[] vertices_tet_out;
    delete[] normals_tet_out;
    delete[] faces_tet_out;
    delete[] res_tet_out;

    cout << size_grid_64.x << endl;
    cout << size_grid_64.y << endl;
    for (int i = 0; i < size_grid_64.x; i++) {
        for (int j = 0; j < size_grid_64.y; j++) {
            delete[]tsdf[i][j];
        }

        delete[]tsdf[i];
    }

    for (int i = 0; i < size_grid_64.x; i++) {
        for (int j = 0; j < size_grid_64.y; j++) {
            delete[]tsdfGT[i][j];
        }
        delete[]tsdfGT[i];
    }
    delete[]tsdfGT;

    float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadPLY_Mesh(output_dir + "coarse.ply", &vertices, &normals_face, &faces);

    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;

    tsdf = LevelSet_gpu(vertices, faces, normals_face, res[0], res[1], size_grid, center_grid, GRID_RES, 0.0f);

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;


    TetMesh->FitToTSDF(tsdf, size_grid, center_grid, GRID_RES, 0.0f);

    SaveTetraMeshToPLY(output_dir + string("FittedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete[]tsdf[i][j];
        }
        delete[]tsdf[i];
    }
    delete[]tsdf;

    int* resGT = LoadPLY_Mesh(output_dir + "coarseGT.ply", &vertices, &normals_face, &faces);

    cout << "The PLY file has " << resGT[0] << " vertices and " << resGT[1] << " faces" << endl;

    tsdfGT = LevelSet_gpu(vertices, faces, normals_face, resGT[0], resGT[1], size_grid, center_grid, GRID_RES, 0.0f);

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;


    TetMesh->FitToTSDF(tsdfGT, size_grid, center_grid, GRID_RES, 0.0f);

    SaveTetraMeshToPLY(output_dir + string("FittedTetraShellGT.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces(), TetMesh->Res());

    for (int i = 0; i < size_grid.x; i++) {
        for (int j = 0; j < size_grid.y; j++) {
            delete[]tsdf[i][j];
        }
        delete[]tsdf[i];
    }
    delete[]tsdf;
}

void RepairBinSDF(string input_dir, string output_dir, string template_dir) {
    cout << "============== Repair ==================" << endl;
    
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    for (int idx_file = 0; idx_file < 165; idx_file++) {
        float ***tsdf = LoadLevelSet(input_dir + string("coarseLevelSet_" + to_string(idx_file) + ".bin"), size_grid);
        
        SaveLevelSet(output_dir + "coarseLevelSet_" + to_string(idx_file) + ".bin", tsdf, size_grid);
        
        for (int i = 0; i < size_grid.x; i++) {
            for (int j = 0; j < size_grid.y; j++) {
                delete []tsdf[i][j];
            }
            delete []tsdf[i];
        }
        delete []tsdf;
    }
    
    return;
    
    /**
            0. Load the outer shell
     */
    TetraMesh *TetMesh = new TetraMesh(template_dir + string("/TetraShell-B.ply"), GRID_RES);
    
    for (int idx_file = 2; idx_file < 125; idx_file++) {
        /*
                2. load the fine level set
         */
        float *tsdf_fine = LoadTetLevelSet(input_dir + "fineLevelSet_" + to_string(idx_file) + ".bin", TetMesh->NBNodes());
        
        SaveLevelSet(output_dir + "fineLevelSet_" + to_string(idx_file) + ".bin", tsdf_fine, TetMesh->NBNodes());
        delete []tsdf_fine;
    }
    
    
    delete TetMesh;
}

void ObjToPlyCentered(string input_filename, string output_filename, string skeleton_path) {
    float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadOBJ_Mesh(input_filename, &vertices, &normals_face, &faces);

    /*
     Extract center and orientation from skeleton data
     */

    float center[3] = { 0.0f, 0.0f, 0.0f };
    float orient[3] = { 0.0f, 0.0f, 0.0f };
    string line;
    ifstream plyfile(skeleton_path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline(plyfile, line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("trans")) == 0) {
                    center[0] = std::stof(words[2]);
                    center[1] = std::stof(words[3]);
                    center[2] = std::stof(words[4]);
                }
                else if (words[1].compare(string("orientation")) == 0) {
                    orient[0] = std::stof(words[2]);
                    orient[1] = std::stof(words[3]);
                    orient[2] = std::stof(words[4]);
                }
            }

            getline(plyfile, line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        plyfile.close();
    }
    else {
        plyfile.close();
        cout << "could not load file: " << skeleton_path << endl;
        return;
    }

    cout << "center ==> " << center[0] << ", " << center[1] << ", " << center[2] << endl;
    cout << "orient ==> " << orient[0] << ", " << orient[1] << ", " << orient[2] << endl;
    float3 e2 = make_float3(0.0f, 1.0f, 0.0f);
    float3 e3 = make_float3(orient[0], orient[1], orient[2]);
    e2 = e2 - e3*dot(e2, e3);
    e2 = e2 * (1.0f / sqrt(dot(e2,e2)));
    float3 e1 = cross(e2, e3);
    e1 = e1 * (1.0f / sqrt(dot(e1, e1)));

    Eigen::Matrix3f Rot_inv;
    Rot_inv << e1.x, e2.x, e3.x,
        e1.y, e2.y, e3.y,
        e1.z, e2.z, e3.z;
    Eigen::Matrix3f Rot = Rot_inv.inverse();

    my_float3 ctr = make_my_float3(center[0], center[1], center[2]);
    //ctr = ctr * Rot;
    for (int v = 0; v < res[0]; v++) {
        if (vertices[3 * v] == 0.0f && vertices[3 * v + 1] == 0.0f && vertices[3 * v + 2] == 0.0f)
            continue;
        my_float3 vtx = make_my_float3(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
        //vtx = vtx * Rot;
        vertices[3 * v] = vtx.x - ctr.x;
        vertices[3 * v + 1] = vtx.y - ctr.y;
        vertices[3 * v + 2] = vtx.z - ctr.z;
    }

    SaveColoredMeshToPLY(output_filename, vertices, faces, res[0], res[1]);

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;
    
}

void PlyToPlyOrient(string input_filename, string output_filename, string template_dir, string skeleton_path) {
    float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadPLY_Mesh(input_filename, &vertices, &normals_face, &faces);

    /*
     Extract center and orientation from skeleton data
     */

     // load kinematic table
    ifstream KTfile(template_dir + "kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int* kinTree_Table = new int[48];
    KTfile.read((char*)kinTree_Table, 48 * sizeof(int));
    KTfile.close();

    Eigen::Matrix4f* Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir + "Tshapecoarsejoints.bin", skeleton_path, Joints, kinTree_Table);

    Eigen::Matrix4f Root = Joints[6].inverse();
    for (int v = 0; v < res[0]; v++) {
        if (vertices[3 * v] == 0.0f && vertices[3 * v + 1] == 0.0f && vertices[3 * v + 2] == 0.0f)
            continue;
        Eigen::Vector4f vtx = Eigen::Vector4f(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2], 1.0f);
        vtx = Root * vtx;
        vertices[3 * v] = vtx(0);
        vertices[3 * v + 1] = vtx(1);
        vertices[3 * v + 2] = vtx(2);
    }

    SaveColoredMeshToPLY(output_filename, vertices, faces, res[0], res[1]);

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;

}

void ColorizeMesh(string input_filename, string output_filename) {
    float* vertices;
    float* normals_face;
    int* faces;
    int* res = LoadPLY_Mesh(input_filename, &vertices, &normals_face, &faces);

    SaveColoredMeshToPLY(output_filename, vertices, faces, res[0], res[1]);

    delete[]vertices;
    delete[]normals_face;
    delete[]faces;

}

void VIBEPoseToSkeleton(string input_path, string output_path, string template_dir) {
    // load kinematic table
    ifstream KTfile (template_dir+"kintree.bin", ios::binary);
    if (!KTfile.is_open()) {
        cout << "Could not load kintree" << endl;
        KTfile.close();
        return;
    }
    int *kinTree_Table = new int[48];
    KTfile.read((char *) kinTree_Table, 48*sizeof(int));
    KTfile.close();
    
    // Load the Skeleton
    DualQuaternion *skeleton = new DualQuaternion[24];
    
    Eigen::Matrix4f *Joints_T = new Eigen::Matrix4f[24]();
    LoadStarSkeleton(template_dir+"Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.1f);
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    LoadSkeletonVIBE(template_dir+"Tshapecoarsejoints.bin", input_path+"smplparams_vibe/", Joints, kinTree_Table, 0);
    
    float *skeletonPC = new float[3*24];
    for (int s = 0; s < 24; s++) {
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
        //skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
        skeletonPC[3*s] = Joints[s](0,3);
        skeletonPC[3*s+1] = Joints[s](1,3);
        skeletonPC[3*s+2] = Joints[s](2,3);
    }
    
    SavePCasPLY(output_path + string("skeleton.ply"), skeletonPC, 24);
    delete []skeletonPC;
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
    
    if(cmdOptionExists(argv, argv+argc, "-h"))
    {
        cout << "HELP" << endl;
        cout << "Use the different modes of teh program using command -mode [mode]" << endl;
        cout << "[mode] = ObjToPlyCentered -> Generate a sequence of centered .ply files from a sequence of .obj files. The .obj files must be in meshes folder and results will be saves in meshes_centered folder" << endl;
        cout << "[mode] = CreateLevelSet -> Generate the coarse and fine level set from one or a sequence of centered .ply files with skeleton data" << endl;
        cout << "[mode] = GenerateOuterShell -> Generate the template outershell" << endl;
        cout << "[mode] = Generate3DMesh -> Generate the 3D mesh for the given level set" << endl;
        return 1;
    }

    // This will pick the best possible CUDA capable device
    int devID = findCudaDevice();
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    size_t free, total, initMem;
    cudaMemGetInfo(&free, &total);
    cout << "GPU " << devID << " memory: free=" << free << ", total=" << total << endl;

    //BuildAdjacencies("D:/Data/Human/Template-star-0.008/");
    
    
    for (int i = 1; i < 210; i++) {
        cout << "data : " + to_string(i) << endl;
        FitTetraToCanonicalFromOut("F:/HUAWEI/Iwamoto/data/FittedMesh_ply_new/"
            ,"F:/HUAWEI/Iwamoto/"
            , "F:/Template-star-0.015/"
            , i);
    }
    return 0;
    
 
    //retopo i_jumping
    /*
    float* weights = GetUVskinWeight("D:/Data/Human/Template-star-0.015/", "D:/Data/Human/ARTICULATED/I_jumping/retopo/mesh_000000.obj");
    for (int i = 0; i < 150; i++) {
        cout << "data : " + to_string(i) << endl;
    Repose2Tshape("D:/Data/Human/ARTICULATED/I_jumping/retopo_tpose/", "D:/Data/Human/ARTICULATED/I_jumping/", "D:/Data/Human/Template-star-0.015/", weights , i);
    retopo_reconst("D:/Data/Human/ARTICULATED/I_jumping/retopo_reconst/", "D:/Data/Human/ARTICULATED/I_jumping/", "D:/Data/Human/Template-star-0.015/", weights, i);
    }
    */

    //our i_jumping
    
    //float* weights = GetUVskinWeight("D:/Data/Human/Template-star-0.015/", "D:/Project/Human/Pose2Texture/make_displacement/uvskin.obj");
    
    //SmplPosechange("D:/Data/Human/Template-star-0.015/");
    //return 0;
    
    if(cmdOptionExists(argv, argv+argc, "-mode"))
    {
        std::string str1 (getCmdOption(argv, argv + argc, "-mode"));

        if (str1.compare(string("PLYToPlyOrient")) == 0) {
            cout << "========= Mode is ObjToPlyCentered =======" << endl;
            if (cmdOptionExists(argv, argv + argc, "-inputpath")) {
                std::string data_dir(getCmdOption(argv, argv + argc, "-inputpath"));
                if (cmdOptionExists(argv, argv + argc, "-size")) {
                    std::string nb_meshes(getCmdOption(argv, argv + argc, "-size"));
                    for (int i = 0; i < stoi(nb_meshes); i++) {
                        string idx_s = to_string(i);
                        idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
                        PlyToPlyOrient(data_dir + "/meshes_centered/mesh_" + to_string(i) + ".ply", 
                                        data_dir + "/meshes_centered/mesh_" + to_string(i) + ".ply",
                                        "F:/Template/",
                                        data_dir + "smplparams/pose_" + idx_s + ".bin");
                    }
                }
                else {
                    cout << "set nb of meshes to convert by using command -size [nb_meshes]" << endl;
                }
            }
            else {
                cout << "set directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }
        
        if (str1.compare(string("ObjToPlyCentered")) == 0) {
            cout << "========= Mode is ObjToPlyCentered =======" << endl;
            if(cmdOptionExists(argv, argv+argc, "-inputpath")) {
                std::string data_dir (getCmdOption(argv, argv + argc, "-inputpath"));
                if(cmdOptionExists(argv, argv+argc, "-size")) {
                    std::string nb_meshes (getCmdOption(argv, argv + argc, "-size"));
                    for (int i = 0; i < stoi(nb_meshes); i++) {
                        string idx_s = to_string(i);
                        idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
                        ObjToPlyCentered(data_dir + "/meshes/mesh_" + idx_s + ".obj", data_dir + "/meshes_centered/mesh_" + to_string(i) + ".ply", data_dir + "openpose_blender/skeleton_" + idx_s + ".ply");
                    }
                } else {
                    cout << "set nb of meshes to convert by using command -size [nb_meshes]" << endl;
                }
            } else {
                cout << "set directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }
        
        if (str1.compare(string("CreateLevelSet")) == 0) {
            cout << "========= Mode is CreateLevelSet =======" << endl;
            if(cmdOptionExists(argv, argv+argc, "-inputpath")) {
                std::string input_dir (getCmdOption(argv, argv + argc, "-inputpath"));
                if(cmdOptionExists(argv, argv+argc, "-outputpath")) {
                    std::string output_dir (getCmdOption(argv, argv + argc, "-outputpath"));
                    if(cmdOptionExists(argv, argv+argc, "-templatepath")) {
                        std::string template_dir (getCmdOption(argv, argv + argc, "-templatepath"));
                        if(cmdOptionExists(argv, argv+argc, "-size")) {
                            std::string nb_meshes (getCmdOption(argv, argv + argc, "-size"));
                            for (int i = 0; i < stoi(nb_meshes); i++) {
                                CreateLevelSet(output_dir, input_dir, template_dir, i);
                            }
                        } else {
                            CreateLevelSet(output_dir, input_dir, template_dir, 1);
                        }
                    } else {
                        cout << "set template directory where template models are using command -templatepath [path]" << endl;
                    }
                } else {
                    cout << "set output directory where results are saves using command -outputpath [path]" << endl;
                }
            } else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }

        if (str1.compare(string("CreateCoarseLevelSet")) == 0) {
            cout << "========= Mode is CreateCoarseLevelSet =======" << endl;
            if (cmdOptionExists(argv, argv + argc, "-inputpath")) {
                std::string input_dir(getCmdOption(argv, argv + argc, "-inputpath"));
                if (cmdOptionExists(argv, argv + argc, "-outputpath")) {
                    std::string output_dir(getCmdOption(argv, argv + argc, "-outputpath"));
                    if (cmdOptionExists(argv, argv + argc, "-templatepath")) {
                        std::string template_dir(getCmdOption(argv, argv + argc, "-templatepath"));
                        if (cmdOptionExists(argv, argv + argc, "-size")) {
                            std::string nb_meshes(getCmdOption(argv, argv + argc, "-size"));
                            for (int i = 0; i < stoi(nb_meshes); i++) {
                                CreateCoarseLevelSet(output_dir, input_dir, template_dir, i);
                            }
                        }
                        else {
                            CreateCoarseLevelSet(output_dir, input_dir, template_dir, 1);
                        }
                    }
                    else {
                        cout << "set template directory where template models are using command -templatepath [path]" << endl;
                    }
                }
                else {
                    cout << "set output directory where results are saves using command -outputpath [path]" << endl;
                }
            }
            else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }
        
        if (str1.compare(string("GenerateOuterShell")) == 0) {
            cout << "========= Mode is GenerateOuterShell =======" << endl;
            if(cmdOptionExists(argv, argv+argc, "-templatepath")) {
                std::string template_dir (getCmdOption(argv, argv + argc, "-templatepath"));
                if(cmdOptionExists(argv, argv+argc, "-grid_size")) {
                    std::string gridsize_s (getCmdOption(argv, argv + argc, "-grid_size"));
                    GRID_SIZE = stoi(gridsize_s);
                }
                if(cmdOptionExists(argv, argv+argc, "-grid_res")) {
                    std::string gridres_s (getCmdOption(argv, argv + argc, "-grid_res"));
                    GRID_RES = stof(gridres_s);
                }
                GenerateOuterShell(template_dir);
            } else {
                cout << "set template directory where template models are using command -templatepath [path]" << endl;
            }
        }
        
        if (str1.compare(string("Generate3DMesh")) == 0) {
            cout << "========= Mode is Generate3DMesh =======" << endl;
            if(cmdOptionExists(argv, argv+argc, "-inputpath")) {
                std::string input_dir (getCmdOption(argv, argv + argc, "-inputpath"));
                if(cmdOptionExists(argv, argv+argc, "-templatepath")) {
                    std::string template_dir (getCmdOption(argv, argv + argc, "-templatepath"));
                    if(cmdOptionExists(argv, argv+argc, "-outputpath")) {
                        std::string output_dir (getCmdOption(argv, argv + argc, "-outputpath"));
                        Generate3DModel(input_dir + "coarse_tsdf.bin", input_dir + "fine_tsdf.bin", input_dir + "skeleton.bin", template_dir, output_dir);
                    } else {
                        cout << "set output directory using command -outputpath [path]" << endl;
                    }
                } else {
                    cout << "set template directory where template models are using command -templatepath [path]" << endl;
                }
            } else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }

        if (str1.compare(string("Generate3DMeshSkinned")) == 0) {
            cout << "========= Mode is Generate3DMeshSkinned =======" << endl;
            if (cmdOptionExists(argv, argv + argc, "-inputpath")) {
                std::string input_dir(getCmdOption(argv, argv + argc, "-inputpath"));
                if (cmdOptionExists(argv, argv + argc, "-templatepath")) {
                    std::string template_dir(getCmdOption(argv, argv + argc, "-templatepath"));
                    if (cmdOptionExists(argv, argv + argc, "-outputpath")) {
                        std::string output_dir(getCmdOption(argv, argv + argc, "-outputpath"));
                        if (cmdOptionExists(argv, argv + argc, "-size")) {
                            std::string nb_meshes(getCmdOption(argv, argv + argc, "-size"));
                            for (int i = 0; i < stoi(nb_meshes); i++) {
                                string idx_s = to_string(i);
                                idx_s.insert(idx_s.begin(), 6 - idx_s.length(), '0');
                                Generate3DModelSkinned(input_dir + "fine_tsdf" + idx_s + ".bin", input_dir + "skeleton" + idx_s + ".bin", template_dir, output_dir + "PredMesh" + idx_s +".ply");
                            }
                        }
                        else {
                            string idx_s = to_string(0);
                            idx_s.insert(idx_s.begin(), 6 - idx_s.length(), '0');
                            Generate3DModelSkinned(input_dir + "fine_tsdf" + idx_s +".bin", input_dir + "skeleton" + idx_s +".bin", template_dir, output_dir + "PredMesh" + idx_s +".ply");
                        }
                    }
                    else {
                        cout << "set output directory using command -outputpath [path]" << endl;
                    }
                }
                else {
                    cout << "set template directory where template models are using command -templatepath [path]" << endl;
                }
            }
            else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }

        if (str1.compare(string("Generate3DMeshTetraTSDF")) == 0) {
            cout << "========= Mode is Generate3DMesh =======" << endl;
            if (cmdOptionExists(argv, argv + argc, "-inputpath")) {
                std::string input_dir(getCmdOption(argv, argv + argc, "-inputpath"));
                if (cmdOptionExists(argv, argv + argc, "-templatepath")) {
                    std::string template_dir(getCmdOption(argv, argv + argc, "-templatepath"));
                    if (cmdOptionExists(argv, argv + argc, "-outputpath")) {
                        std::string output_dir(getCmdOption(argv, argv + argc, "-outputpath"));
                        if (cmdOptionExists(argv, argv + argc, "-size")) {
                            std::string nb_meshes(getCmdOption(argv, argv + argc, "-size"));
                            for (int i = 0; i < stoi(nb_meshes); i++) {
                                string idx_s = to_string(i);
                                idx_s.insert(idx_s.begin(), 6 - idx_s.length(), '0');
                                Generate3DModelTetraTSDF(input_dir + "fine_tsdf" + idx_s + ".bin", input_dir + "skeleton" + idx_s + ".bin", template_dir, output_dir + "PredMesh" + idx_s + ".ply");
                            }
                        }
                        else {
                            string idx_s = to_string(0);
                            idx_s.insert(idx_s.begin(), 6 - idx_s.length(), '0');
                            Generate3DModelTetraTSDF(input_dir + "fine_tsdf" + idx_s + ".bin", input_dir + "skeleton" + idx_s + ".bin", template_dir, output_dir + "PredMesh" + idx_s + ".ply");
                        }
                    }
                    else {
                        cout << "set output directory using command -outputpath [path]" << endl;
                    }
                }
                else {
                    cout << "set template directory where template models are using command -templatepath [path]" << endl;
                }
            }
            else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }

        if (str1.compare(string("GenerateCoarse3DMesh")) == 0) {
            cout << "========= Mode is GenerateCoarse3DMesh =======" << endl;
            if (cmdOptionExists(argv, argv + argc, "-inputpath")) {
                std::string input_dir(getCmdOption(argv, argv + argc, "-inputpath"));
                if (cmdOptionExists(argv, argv + argc, "-templatepath")) {
                    std::string template_dir(getCmdOption(argv, argv + argc, "-templatepath"));
                    if (cmdOptionExists(argv, argv + argc, "-outputpath")) {
                        std::string output_dir(getCmdOption(argv, argv + argc, "-outputpath"));
                        GenerateCoarse3DModel(input_dir + "coarse_tsdf.bin", input_dir + "coarse_tsdfGT.bin", input_dir + "skeleton.bin", template_dir, output_dir);
                    }
                    else {
                        cout << "set output directory using command -outputpath [path]" << endl;
                    }
                } else {
                    cout << "set template directory where template models are using command -templatepath [path]" << endl;
                }
            }
            else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }

        if (str1.compare(string("GetPosedResult")) == 0) {
            cout << "========= Mode is GetPosedResult =======" << endl;
            if (cmdOptionExists(argv, argv + argc, "-inputpath")) {
                std::string input_dir(getCmdOption(argv, argv + argc, "-inputpath"));
                if (cmdOptionExists(argv, argv + argc, "-smplpath")) {
                    std::string smpl_dir(getCmdOption(argv, argv + argc, "-smplpath"));
                    if (cmdOptionExists(argv, argv + argc, "-outputpath")) {
                        std::string output_dir(getCmdOption(argv, argv + argc, "-outputpath"));
                        if (cmdOptionExists(argv, argv + argc, "-file_num")) {
                            std::string file_num(getCmdOption(argv, argv + argc, "-file_num"));
                            if (cmdOptionExists(argv, argv + argc, "-color_flg")) {
                                std::string color_flg(getCmdOption(argv, argv + argc, "-color_flg"));
                                if (cmdOptionExists(argv, argv + argc, "-individual_sw_flg")) {
                                    std::string individual_sw_flg(getCmdOption(argv, argv + argc, "-individual_sw_flg"));
                                    if (individual_sw_flg == "True") {
                                        if (cmdOptionExists(argv, argv + argc, "-inputSWpath")) {
                                            std::string inputSWpath(getCmdOption(argv, argv + argc, "-inputSWpath"));
                                            GetPosedResult(output_dir, input_dir, smpl_dir, stoi(file_num), color_flg, inputSWpath);
                                        }
                                    }
                                    else {
                                        GetPosedResult(output_dir, input_dir, smpl_dir, stoi(file_num), color_flg);
                                    }
                                        
                                }
                            }
                        }
                    }
                    else {
                        cout << "set output directory using command -outputpath [path]" << endl;
                    }
                }
                else {
                    cout << "set template directory where template models are using command -templatepath [path]" << endl;
                }
            }
            else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }

        return 0;
    }

    string data_dir = "F:/Articulated/D_march/";
    //string data_dir = "F:/Articulated/T_swing/";
    //string data_dir = "F:/CAPE/00122/shortlong_ballerina_spin/";
    //string data_dir = "F:/4D_data_example/";

    //CreateLevelSetTetraTSDF(data_dir + "TSDF-TetraTSDF/", data_dir, "F:/Template/", 0);
    
    /*for (int i = 0; i < 250; i++) {
        CreateLevelSetTetraTSDF(data_dir + "TSDF-TetraTSDF/", data_dir, "F:/Template/", i);
    }*/


    /*data_dir = "F:/CAPE/03331/shortshort_chicken_wings/";
    for (int i = 0; i < 132; i++) {
        string idx_s = to_string(i);
        idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
        ObjToPlyCentered(data_dir + "meshes/mesh_" + idx_s + ".obj", data_dir + "meshes_centered/mesh_" + to_string(i) + ".ply", data_dir + "openpose_blender/skeleton_" + idx_s + ".ply");
        PlyToPlyOrient(data_dir + "meshes_centered/mesh_" + to_string(i) + ".ply",
            data_dir + "meshes_centered/mesh_" + to_string(i) + ".ply",
            "F:/Template/",
            data_dir + "smplparams/pose_" + idx_s + ".bin");
    }

    for (int i = 0; i < 132; i++) {
        CreateLevelSet(data_dir + "TSDF/", data_dir, "F:/Template/", i);
    }*/





    /*for (int i = 0; i < 250; i++) {
        CreateLevelSet("F:/Articulated/I_squat/TSDF/", "F:/Articulated/I_squat/", "F:/Template/", i);
    }*/
    
   /* RepairBinSDF("/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/TSDF/", "/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/tmp/", "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template");*/
    
    /*VIBEPoseToSkeleton("/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/", "/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/skeleton_vibe/", "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template/");*/
    
    /*CreateLevelSet(output_mesh_folder_name,
        "D:/Data/Human/ARTICULATED/D_march/",
        "D:/Data/Human/Template/", 0);*/
    /*CreateLevelSet(output_mesh_folder_name, "F:/Articulated/D_march/",
                       "F:/Template/", 0);*/

    /*for (int i = 0; i < 250; i++) {
        ColorizeMesh("F:/Articulated/D_march/meshes_centered/mesh_" + to_string(i) + ".ply", "F:/Articulated/D_march/colored_meshes/mesh_" + to_string(i) + ".ply");
    }*/

    /*for (int i = 81; i < 284; i++) {
        string idx_s = to_string(i);
        idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
        CreateLevelSet("F:/CAPE/00032/longshort_flying_eagle/TSDF/", "F:/CAPE/00032/longshort_flying_eagle/", "F:/Template/", i);
        /*PlyToPlyOrient("F:/CAPE/03331/shortshort_chicken_wings/meshes_centered/mesh_" + to_string(i) + ".ply",
            "F:/CAPE/03331/shortshort_chicken_wings/meshes_centered/mesh_" + to_string(i) + ".ply",
            "F:/Template/", "F:/CAPE/03331/shortshort_chicken_wings/smplparams/pose_" + idx_s + ".bin");*/
    //}
    
    //TestArticulated();
    
    // A. Generate the outer shell from a 3D scan
    //GenerateOuterShell();
    //BuildAdjacencies("F:/Template/");
    
    // B. Fit the outer shell to the scan
    //FitGridToOuterShell();
    //FitTetraOuterShell();
    
    // C. Embedd the 3D scan into the fitted outershell
    //EmbedScanIntoOuterShell(string("meshes_centered/mesh_0.ply"), string("smplparams_centered/skeleton_0.bin"));
    
    // D. Build Dataset
    //BuildDataSet();
    
    // E. Generate results
    /*GenerateCoarse3DModel("C:/Users/thomas/Documents/Projects/Output/test/coarse_tsdf.bin",
        "C:/Users/thomas/Documents/Projects/Output/test/coarse.ply");*/

    /*Generate3DModel("C:/Users/thomas/Documents/Projects/Output/test/coarse_tsdf.bin", 
        "C:/Users/thomas/Documents/Projects/Output/test/fine_tsdf.bin", 
        "C:/Users/thomas/Documents/Projects/Output/test/skeleton.bin", 
        "F:/Template/", "C:/Users/thomas/Documents/Projects/Output/test/");
    */


	return 0 ;
}

