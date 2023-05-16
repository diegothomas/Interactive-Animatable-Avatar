#include "include/Utilities.h"
#include "include/MarchingCubes.h"
#include "include/LevelSet.h"
#include "include/MeshUtils.h"
#include "include/TetraMesh.h"
#include "include/CubicMesh.h"
#include "include/Fit.h"

using namespace std;

#ifndef _APPLE_
SYSTEMTIME current_time, last_time;
#else
struct timeval current_time, last_time;
#endif

#ifdef _APPLE_
    string folder_name = "/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/";
    //string folder_name = "/Users/diegothomas/Documents/Data/ARTICULATED/D_march/";
    string output_mesh_folder_name = "/Users/diegothomas/Documents/Projects/HUAWEI/Output/";
#else
    //string folder_name = "D:/Data/KinectV2/WACV2021/all_data/KinectV2/2020_04_12_01/";
    string folder_name = "D:/Data/KinectV2/2020_02_26_10_27_13/camera_0/"; // data for teaser image
    //string folder_name = "D:/Data/KinectV2/WACV2021/Fig-6-a/"; // data for teaser image
    string output_mesh_folder_name = "D:/Projects/Human/HUAWEI/Output";
#endif

int GRID_SIZE = 128;
float GRID_RES = 0.022f;
//int GRID_SIZE = 64;
//float GRID_RES = 0.035f;

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
    LoadStarSkeleton(template_dir+"/Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.1f);
        
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
    SkinMesh(vertices, skin_weights, skeleton, res[0]);
    
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
    
    SaveTetraMeshToPLY(template_dir + string("/TetraShell.ply"), nodes_out, nodes_out, faces_out, tetra_out, res_tetra[0], res_tetra[1], nb_faces_out);
    
            
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
    SkinMesh(vertices, skin_weights, skeleton, res[0]);
    
    UpdateNormalsFaces(vertices, normals_face, faces, res[1]);
    
    SaveMeshToPLY(output_mesh_folder_name + string("skin.ply"), vertices, vertices, faces, res[0], res[1]);
    
    /**
            3. Build the level-set in the outer-shell
     */
    float *vol_weights;
    float *tsdf = TetMesh->LevelSet(&vol_weights, vertices, skin_weights, faces, normals_face, res[1]);
    
    /**
            4. Extract the 3D mesh
     */
    float *VerticesTet = new float[3*TetMesh->NBEdges()];
    float *SkinWeightsTet = new float[24*TetMesh->NBEdges()];
    int *FacesTet = new int[3*3*TetMesh->NBTets()];
    int *res_tet = TetMesh->MT(tsdf, vol_weights, VerticesTet, SkinWeightsTet, FacesTet);
        
    SaveMeshToPLY(output_mesh_folder_name + string("EmbeddedMesh.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
    
    /**
            5. Re-pose the 3D mesh
     */
    
    //transferSkinWeights(VerticesTet, SkinWeightsTet, vertices, skin_weights, res_tet[0], res[0]);
    
    // Transform the mesh from the star pose back to the original pose
    DualQuaternion *skeletonBack = new DualQuaternion[24];
    for (int s = 0; s < 24; s++) {
        skeletonBack[s] = skeleton[s].Inv(skeleton[s]);
    }
    
    //SkinMesh(vertices, skin_weights, skeletonBack, res[0]);
    //SaveMeshToPLY(output_mesh_folder_name + string("skinBack.ply"), vertices, faces, res[0], res[1]);
    
    SkinMesh(VerticesTet, SkinWeightsTet, skeletonBack, res_tet[0]);
    SaveMeshToPLY(output_mesh_folder_name + string("EskinBack.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
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

void CreateLevelSet(string output_dir, string input_dir, string template_dir, int idx_file) {
    string idx_s = to_string(idx_file);
    idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
    /**
            1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
     */
    float *vertices;
    float *normals_face;
    int *faces;
    int *res = LoadPLY_Mesh(input_dir + "meshes_centered/mesh_" + idx_s + ".ply", &vertices, &normals_face, &faces);
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
    LoadStarSkeleton(template_dir+"Tshapecoarsejoints.bin", Joints_T, kinTree_Table);//, 0.1f);
    //LoadTSkeleton(template_dir+"Tshapecoarsejoints.bin", Joints_T, kinTree_Table);
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    //LoadSkeleton(template_dir+"Tshapecoarsejoints.bin", input_dir + "smplparams_centered/skeleton_" + idx_s + ".bin", Joints, kinTree_Table);
    LoadSkeleton(template_dir+"Tshapecoarsejoints.bin", input_dir + "smplparams/pose_" + idx_s + ".bin", Joints, kinTree_Table);
    // <=========== Here is to load fitted skeleton
    //LoadSkeleton(template_dir+"Tshapecoarsejoints.bin", input_dir + "Fit/pose.bin", Joints, kinTree_Table);
    // <=========== Here is to load Vibe skeleton
    //LoadSkeletonVIBE(template_dir+"Tshapecoarsejoints.bin", input_dir+"smplparams_vibe/", Joints, kinTree_Table, idx_file);
    
    float *skeletonPC = new float[3*24];
    for (int s = 0; s < 24; s++) {
        //cout << Joints[s] << endl;
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
        //skeleton[s] = DualQuaternion(Joints_T[s] * InverseTransfo(Joints[s]));
        skeletonPC[3*s] = Joints[s](0,3);
        skeletonPC[3*s+1] = Joints[s](1,3);
        skeletonPC[3*s+2] = Joints[s](2,3);
    }
    
    #ifdef VERBOSE
        SavePCasPLY(output_mesh_folder_name + string("skel_in.ply"), skeletonPC, 24);
    #endif
    
    delete []skeletonPC;
    
    // Transform the outershell to the mesh pose
    /*SkinMesh(vertices, skin_weights, skeleton, res[0]);
    
    UpdateNormalsFaces(vertices, normals_face, faces, res[1]);*/
    
    #ifdef VERBOSE
        SaveMeshToPLY(output_mesh_folder_name + string("skin.ply"), vertices, vertices, faces, res[0], res[1]);
    #endif
    
    /**
            3. Fit the outershell
     */
        /**
                3.0. Load the outer shell
         */
        TetraMesh *TetMesh = new TetraMesh(template_dir, GRID_RES);
    
        // Transform the outershell to the mesh pose
        TetMesh->Skin(skeleton);
    
        #ifdef VERBOSE
            SaveTetraMeshToPLY(output_mesh_folder_name + string("SkinnedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces());
        #endif
    
        /**
                3.1. Load the 1st 3D scan of the sequence (it should be a centered ply file)
         */
        my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
        //my_float3 center_grid = make_my_float3(0.0f,-0.1f,0.0f);
        my_float3 center_grid = make_my_float3(0.0f,-0.3f,0.0f);
        
        float ***coarse_sdf = LevelSet(vertices, faces, normals_face, res[1], size_grid, center_grid, GRID_RES, 0.0f);
    
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
        SaveLevelSet(output_dir + "coarseLevelSet_" + idx_s + ".bin", coarse_sdf, size_grid);
        cout << "The coarse level set is generated" << endl;
        
        #ifndef BASIC_TETRA
            cout << "Fitting to sdf grid" << endl;
            TetMesh->FitToTSDF(coarse_sdf, size_grid, center_grid, GRID_RES, 0.1f);//, 20, 1, 10, 0.01f, 0.1f);
        #endif
    
        #ifdef VERBOSE
            SaveTetraMeshToPLY(output_mesh_folder_name + string("FittedTetraShellB.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces());
            
            SavePCasPLY(output_mesh_folder_name + string("FittedPC.ply"), TetMesh->Nodes(), TetMesh->NBNodes());
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
    float *vol_weights;
    float *fine_sdf = TetMesh->LevelSet(&vol_weights, vertices, skin_weights, faces, normals_face, res[1]);
    SaveLevelSet(output_dir + "fineLevelSet_" + idx_s + ".bin", fine_sdf, vol_weights, TetMesh->NBNodes());
    cout << "The fine level set is generated" << endl;
    
    
    /**
            5. Extract the 3D mesh
     */
    //#ifdef VERBOSE
        float *VerticesTet = new float[3*TetMesh->NBEdges()];
        float *SkinWeightsTet = new float[24*TetMesh->NBEdges()];
        int *FacesTet = new int[3*3*TetMesh->NBTets()];
        int *res_tet = TetMesh->MT(fine_sdf, &fine_sdf[TetMesh->NBNodes()], VerticesTet, SkinWeightsTet, FacesTet);
            
        SaveMeshToPLY(output_dir + string("Output_Mesh_" + idx_s + ".ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
    
        delete []VerticesTet;
        delete []SkinWeightsTet;
        delete []FacesTet;
        delete []res_tet;
    //#endif
    
    delete[] vol_weights;
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
    SaveTetraMeshToPLY(output_mesh_folder_name + string("FittedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces());
    
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

void FitTetraOuterShellFromOut() {
    cout << "==============Build the user specific Outer shell and Fit the outershell to it==================" << endl;
    
    /**
            0. Load the outer shell
     */
    TetraMesh *TetMesh = new TetraMesh(output_mesh_folder_name + string("TetraShell.ply"), GRID_RES);
    
    /**
            1. Load the Level Set
     */
    //TetMesh->LoadLevelSet(output_mesh_folder_name + string("levelset.bin"));
    float *vol_weights;
    float *sdf = new float[TetMesh->NBNodes()];
    LoadLevelSet(output_mesh_folder_name + string("sdf.bin"), sdf, &vol_weights, TetMesh->NBNodes());
    cout << "The level set is generated" << endl;
    /*for (int n = 0; n < TetMesh->NBNodes(); n++) {
        cout << sdf[n] << endl;
    }*/
    
    /**
            2. Load the Template Mesh
     */
    float *vertices;
    float *normals;
    int *faces;
    int *res = TetMesh->GetSurface(&vertices, &normals, &faces);
    cout << "The PLY file has " << res[0] << " vertices and " << res[1] << " faces" << endl;
    
    SaveMeshToPLY(output_mesh_folder_name + string("skin.ply"), vertices, normals, faces, res[0], res[1]);
    
    /**
            3. Fit the mesh to the TSDF
     */
    fitLevelSet::FitToLevelSet(sdf, TetMesh->Nodes(), TetMesh->Tetras(), TetMesh->NBTets(), vertices, normals, faces, res[0], res[1]);
    
    SaveMeshToPLY(output_mesh_folder_name + string("FittedMesh.ply"), vertices, vertices, faces, res[0], res[1]);
        
    delete[] sdf;
    delete[] vertices;
    delete[] normals;
    delete[] faces;
    delete TetMesh;
}

void BuildAdjacencies() {
    TetraMesh *TetMesh = new TetraMesh(output_mesh_folder_name + string("/Template/TetraShell-B.ply"), GRID_RES);
    
    int rates[4] = {4,2,2,2};
    TetMesh->ComputeAdjacencies(output_mesh_folder_name, rates, 9);
    
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

void Generate3DModel(string coarse_tsdf_path, string fine_tsdf_path, string skeleton_path, string template_dir, string output_dir = output_mesh_folder_name) {
    cout << "==============Generate the 3D mesh from the level sets==================" << endl;
    
    /**
            0. Load the outer shell
     */
    TetraMesh *TetMesh = new TetraMesh(template_dir, GRID_RES);
    
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
    LoadStarSkeleton(template_dir+"/Tshapecoarsejoints.bin", Joints_T, kinTree_Table, 0.1f);
        
    Eigen::Matrix4f *Joints = new Eigen::Matrix4f[24]();
    LoadSkeleton(template_dir+"/Tshapecoarsejoints.bin", skeleton_path, Joints, kinTree_Table);
    
    for (int s = 0; s < 24; s++)
        skeleton[s] = DualQuaternion(Joints[s] * InverseTransfo(Joints_T[s]));
    
    TetMesh->Skin(skeleton);
    
    
    SaveTetraMeshToPLY(output_dir + string("SkinnedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces());
    
    /**
            1. Load the coarse level set and fit the outer shell to it
     */
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f,-0.3f,0.0f);
    
    float ***tsdf = LoadLevelSet(coarse_tsdf_path, size_grid);
    cout << "The level set is generated" << endl;
    
    float *vertices_coarse;
    float *normals_coarse;
    int *faces_coarse;
    int *res_coarse = MarchingCubes(tsdf, &vertices_coarse, &normals_coarse, &faces_coarse, size_grid, center_grid, GRID_RES, 0.0f);
    // Save 3D mesh to the disk for visualization
    SaveMeshToPLY(output_dir + string("coarse.ply"), vertices_coarse, normals_coarse, faces_coarse, res_coarse[0], res_coarse[1]);
    
    delete[] vertices_coarse;
    delete[] normals_coarse;
    delete[] faces_coarse;
    delete[] res_coarse;
    
    TetMesh->FitToTSDF(tsdf, size_grid, center_grid, GRID_RES, 0.1f);
    
    SaveTetraMeshToPLY(output_dir + string("FittedTetraShell.ply"), TetMesh->Nodes(), TetMesh->Normals(), TetMesh->Faces(), TetMesh->Tetras(), TetMesh->NBNodes(), TetMesh->NBTets(), TetMesh->NBFaces());
        
    
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
        
    SaveMeshToPLY(output_dir + string("Output_Mesh.ply"), VerticesTet, FacesTet, res_tet[0], res_tet[1]);
    
    delete TetMesh;
    delete []VerticesTet;
    delete []SkinWeightsTet;
    delete []FacesTet;
    delete []tsdf_fine;
}

void GenerateCoarseModel(string coarse_tsdf_path, string save_path) {
    cout << "==============Generate the 3D mesh from the coarse level sets==================" << endl;

    /**
            1. Load the coarse level set and fit the outer shell to it
     */
    my_int3 size_grid = make_my_int3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    my_float3 center_grid = make_my_float3(0.0f, -0.3f, 0.0f);

    float*** tsdf = LoadLevelSet(coarse_tsdf_path, size_grid);
    cout << "The level set is generated" << endl;

    float* vertices_coarse;
    float* normals_coarse;
    int* faces_coarse;
    int* res_coarse = MarchingCubes(tsdf, &vertices_coarse, &normals_coarse, &faces_coarse, size_grid, center_grid, GRID_RES, 0.0f);
    // Save 3D mesh to the disk for visualization
    SaveMeshToPLY(save_path + string("coarse.ply"), vertices_coarse, normals_coarse, faces_coarse, res_coarse[0], res_coarse[1]);

    delete[] vertices_coarse;
    delete[] normals_coarse;
    delete[] faces_coarse;
    delete[] res_coarse;

   
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
        
        SaveLevelSet(output_dir + "fineLevelSet_" + to_string(idx_file) + ".bin", tsdf_fine, NULL, TetMesh->NBNodes());
        delete []tsdf_fine;
    }
    
    
    delete TetMesh;
}

void ObjToPlyCentered(string input_filename, string output_filename, string skeleton_path) {
    float *vertices;
    float *normals_face;
    int *faces;
    int *res = LoadOBJ_Mesh(input_filename, &vertices, &normals_face, &faces);
    
    /*
     Extract center and orientation from skeleton data
     */
    
    float center[3] = {0.0f, 0.0f, 0.0f};
    float orient[3] = {0.0f, 0.0f, 0.0f};
    string line;
    ifstream plyfile (skeleton_path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline (plyfile,line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        
        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("trans")) == 0) {
                    center[0] = std::stof(words[2]);
                    center[1] = std::stof(words[3]);
                    center[2] = std::stof(words[4]);
                } else if (words[1].compare(string("orientation")) == 0) {
                    orient[0] = std::stof(words[2]);
                    orient[1] = std::stof(words[3]);
                    orient[2] = std::stof(words[4]);
                }
            }
            
            getline (plyfile,line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        plyfile.close();
    } else {
        plyfile.close();
        cout << "could not load file: " << skeleton_path << endl;
        return;
    }
    
    cout << "center ==> " << center[0] << ", " << center[1] << ", " << center[2] << endl;
    cout << "orient ==> " << orient[0] << ", " << orient[1] << ", " << orient[2] << endl;
    
    my_float3 e2 = make_my_float3(0.0f, 1.0f, 0.0f);
    my_float3 e3 = make_my_float3(orient[0], orient[1], orient[2]);
    e3 = e3 - dot(e2,e3);
    e3 = e3 * (1.0f / norm(e3));
    my_float3 e1 = cross(e2, e3);
    e1 = e1 * (1.0f / norm(e1));
    
    Eigen::Matrix3f Rot_inv;
    Rot_inv << e1.x, e2.x, e3.x,
            e1.y, e2.y, e3.y,
            e1.z, e2.z, e3.z;
    //cout << Rot_inv << endl;
    Eigen::Matrix3f Rot = Rot_inv.inverse();
    
    /*int nb_v = 0;
    
    for (int v = 0; v < res[0]; v++) {
        if (vertices[3*v] == 0.0f && vertices[3*v+1] == 0.0f && vertices[3*v+2] == 0.0f)
            continue;
        center[0] = center[0] + vertices[3*v];
        center[1] = center[1] + vertices[3*v+1];
        center[2] = center[2] + vertices[3*v+2];
        nb_v++;
    }
    center[0] = center[0]/float(nb_v);
    center[1] = center[1]/float(nb_v);
    center[2] = center[2]/float(nb_v);*/
    
    for (int v = 0; v < res[0]; v++) {
        if (vertices[3*v] == 0.0f && vertices[3*v+1] == 0.0f && vertices[3*v+2] == 0.0f)
            continue;
        my_float3 vtx = make_my_float3(vertices[3*v], vertices[3*v+1], vertices[3*v+2]);
        vtx = vtx * Rot;
        vertices[3*v] = vtx.x - center[0];
        vertices[3*v+1] = vtx.y - center[1];
        vertices[3*v+2] = vtx.z - center[2];
    }
    
    SaveMeshToPLY(output_filename, vertices, faces, res[0], res[1]);
    
    delete []vertices;
    delete []normals_face;
    delete []faces;
    
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
        cout << "[mode] = GenerateCoarse -> Generate the 3D mesh for the given coarse level set" << endl;
        return 1;
    }
    
    
    if(cmdOptionExists(argv, argv+argc, "-mode"))
    {
        std::string str1 (getCmdOption(argv, argv + argc, "-mode"));
        
        if (str1.compare(string("ObjToPlyCentered")) == 0) {
            cout << "========= Mode is ObjToPlyCentered =======" << endl;
            if(cmdOptionExists(argv, argv+argc, "-inputpath")) {
                std::string data_dir (getCmdOption(argv, argv + argc, "-inputpath"));
                if(cmdOptionExists(argv, argv+argc, "-size")) {
                    std::string nb_meshes (getCmdOption(argv, argv + argc, "-size"));
                    for (int i = 0; i < stoi(nb_meshes); i++) {
                        string idx_s = to_string(i);
                        idx_s.insert(idx_s.begin(), 4 - idx_s.length(), '0');
                        ObjToPlyCentered(data_dir + "/meshes/mesh_" + idx_s + ".obj", data_dir+"/meshes_centered/mesh_" + idx_s + ".ply", data_dir+"openpose_blender/skeleton_" + idx_s + ".ply");
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
        if (str1.compare(string("GenerateCoarse")) == 0) {
            cout << "========= Mode is GenerateCoarse =======" << endl;
            if (cmdOptionExists(argv, argv + argc, "-inputpath")) {
                std::string input_dir(getCmdOption(argv, argv + argc, "-inputpath"));
                if (cmdOptionExists(argv, argv + argc, "-savepath")) {
                    std::string save_dir(getCmdOption(argv, argv + argc, "-savepath"));
                    GenerateCoarseModel(input_dir, save_dir);
                }
                else {
                    cout << "set save directory using command -savepath [path]" << endl;
                }
            }
            else {
                cout << "set input directory where meshes folder is using command -inputpath [path]" << endl;
            }
        }
        
        return 0;
    }
    
    ObjToPlyCentered("/Users/diegothomas/Documents/Data/ARTICULATED/D_squat/meshes/mesh_0000.obj", "/Users/diegothomas/Documents/Data/ARTICULATED/D_squat/meshes_centered/mesh_0000.ply", "/Users/diegothomas/Documents/Data/ARTICULATED/D_squat/openpose_blender/mesh_0000/Skeleton.ply");
    
   /* RepairBinSDF("/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/TSDF/", "/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/tmp/", "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template");*/
    
    /*VIBEPoseToSkeleton("/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/", "/Users/diegothomas/Documents/Projects/Data/ARTICULATED/D_march/skeleton_vibe/", "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template/");*/
    
    /*for (int i = 0; i < 250; i++) {
        CreateLevelSet("/Users/diegothomas/Documents/Data/ARTICULATED/D_squat/TSDF/",
                           "/Users/diegothomas/Documents/Data/ARTICULATED/D_squat/",
                           "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template/", i);
    }*/
    
    //TestArticulated();
    
    // A. Generate the outer shell from a 3D scan
    //GenerateOuterShell();
    //BuildAdjacencies();
    
    // B. Fit the outer shell to the scan
    //FitGridToOuterShell();
    //FitTetraOuterShell();
    
    // C. Embedd the 3D scan into the fitted outershell
    //EmbedScanIntoOuterShell(string("meshes_centered/mesh_0.ply"), string("smplparams_centered/skeleton_0.bin"));
    
    // D. Build Dataset
    //BuildDataSet();
    
    // E. Generate results
    /*Generate3DModel("/Users/diegothomas/Documents/Projects/HUAWEI/Output/coarse.bin", "/Users/diegothomas/Documents/Projects/HUAWEI/Output/fine.bin", "/Users/diegothomas/Documents/Projects/HUAWEI/Output/skeleton.bin", "/Users/diegothomas/Documents/Projects/HUAWEI/Output/Template");*/
    
	return 0 ;
}

