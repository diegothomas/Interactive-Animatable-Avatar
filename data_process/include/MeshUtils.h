//
//  MeshUtils.h
//  DEEPANIM
//
//  Created by Diego Thomas on 2021/01/18.
//

#ifndef MeshUtils_h
#define MeshUtils_h

#include "include/Utilities.h"

void SaveColoredPC(string path, float *PC, float* Colors, int nbPoints){
    ofstream pcfile (path);
    if (pcfile.is_open()){
        for(int i=0; i< nbPoints; i++){
            pcfile << PC[3*i] << " " << PC[3*i+1] << " " << PC[3*i+2] <<" " << Colors[3*i] << " " << Colors[3*i+1] << " " << Colors[3*i+2]  << endl;
        }
        pcfile.close();
    }
}

void SavePCasPLY(string path, float *PC, int nbPoints) {
    ofstream plyfile (path);
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbPoints << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << PC[3*i] << " " << PC[3*i+1] << " " << PC[3*i+2] << endl;
        }
    }

      plyfile.close();
}

void SaveMeshToPLY(string path, float *Vertices, float *Normals, int *Faces, int nbPoints, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbPoints << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "property float nx" << endl;
        plyfile << "property float ny" << endl;
        plyfile << "property float nz" << endl;
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << Vertices[3*i] << " " << Vertices[3*i+1] << " " << Vertices[3*i+2] << " " << Normals[3*i] << " " << Normals[3*i+1] << " " << Normals[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
    }

      plyfile.close();
}

void SaveMeshToPLY(string path, float *Vertices, int *Faces, int nbPoints, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbPoints << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << Vertices[3*i] << " " << Vertices[3*i+1] << " " << Vertices[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
    }

      plyfile.close();
}

void SaveTetraMeshToPLY(string path, float *Nodes, float *Normals, int *Faces, int *Tetra, int nbNodes, int nbTetra, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbNodes << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "property float nx" << endl;
        plyfile << "property float ny" << endl;
        plyfile << "property float nz" << endl;
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        /*plyfile << "element face " << 4*nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;*/
        plyfile << "element voxel " << nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbNodes; i++) {
            plyfile << Nodes[3*i] << " " << Nodes[3*i+1] << " " << Nodes[3*i+2] << " " << Normals[3*i] << " " << Normals[3*i+1] << " " << Normals[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
        
        for (int f = 0; f < nbTetra; f++) {
            plyfile << 4 << " " << Tetra[4*f] << " " << Tetra[4*f+1] << " " << Tetra[4*f+2] << " " << Tetra[4*f+3] << endl;
        }
        //cout << "voxels written" << endl;
    }

      plyfile.close();
}

void SaveTetraAndMeshToPLY(string path, float *Vertices, float *Normals, int *Faces, float *Nodes, int *Surface, int *Tetra, int nbNodes, int nbSurface, int nbTetra, int nbPoints, int nbFaces) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbPoints << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "property float nx" << endl;
        plyfile << "property float ny" << endl;
        plyfile << "property float nz" << endl;
        plyfile << "element face " << nbFaces << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        /*plyfile << "element face " << 4*nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;*/
        plyfile << "element nodes " << nbNodes << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "element voxel " << nbTetra << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "element surface " << nbSurface << endl;
        plyfile << "property list uchar int vertex_index" << endl;
        plyfile << "end_header" << endl;
        //property float nx\nproperty float ny\nproperty float nz\nelement face \(nbFaces) \nproperty list uchar int vertex_index \nend_header\n" << endl;

        for (int i = 0; i < nbPoints; i++) {
            plyfile << Vertices[3*i] << " " << Vertices[3*i+1] << " " << Vertices[3*i+2] << " " << Normals[3*i] << " " << Normals[3*i+1] << " " << Normals[3*i+2] << endl;
        }
        //cout << "points written" << endl;
        
        for (int f = 0; f < nbFaces; f++) {
            plyfile << 3 << " " << Faces[3*f] << " " << Faces[3*f+1] << " " << Faces[3*f+2] << endl;
        }
        //cout << "faces written" << endl;
                
        for (int i = 0; i < nbNodes; i++) {
            plyfile << Nodes[3*i] << " " << Nodes[3*i+1] << " " << Nodes[3*i+2] << endl;
        }
        //cout << "nodes written" << endl;
        
        /*for (int v = 0; v < nbTetra; v++) {
            plyfile << 3 << " " << Tetra[4*v] << " " << Tetra[4*v+1] << " " << Tetra[4*v+2] << endl;
            plyfile << 3 << " " << Tetra[4*v] << " " << Tetra[4*v+1] << " " << Tetra[4*v+3] << endl;
            plyfile << 3 << " " << Tetra[4*v] << " " << Tetra[4*v+2] << " " << Tetra[4*v+3] << endl;
            plyfile << 3 << " " << Tetra[4*v+1] << " " << Tetra[4*v+2] << " " << Tetra[4*v+3] << endl;
        }
        cout << "faces written" << endl;*/
        
        for (int f = 0; f < nbTetra; f++) {
            plyfile << 4 << " " << Tetra[4*f] << " " << Tetra[4*f+1] << " " << Tetra[4*f+2] << " " << Tetra[4*f+3] << endl;
        }
        //cout << "voxels written" << endl;
        
        for (int s = 0; s < nbSurface; s++) {
            plyfile << 2 << " " << Surface[2*s] << " " << Surface[2*s+1] << endl;
        }
        //cout << "surface written" << endl;
    }

      plyfile.close();
}


void SaveTetraSurfaceToPLY(string path, float *Vertices, int *Surface, int nbSurface) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << nbSurface << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "end_header" << endl;

        for (int i = 0; i < nbSurface; i++) {
            plyfile << Vertices[3*Surface[i]] << " " << Vertices[3*Surface[i]+1] << " " << Vertices[3*Surface[i]+2] << endl;
        }
        //cout << "points written" << endl;
    }

    plyfile.close();
}

void SaveCubicMeshToPLY(string path, my_float3 ***Voxels, my_int3 dim) {
    ofstream plyfile (path);
    cout<<"Saving the mesh to ply"<<endl;
    if (plyfile.is_open()) {
        plyfile << "ply" << endl;
        plyfile << "format ascii 1.0" << endl;
        plyfile << "comment ply file created by Diego Thomas" << endl;
        plyfile << "element vertex " << dim.x*dim.y*6 << endl;
        plyfile << "property float x" << endl;
        plyfile << "property float y" << endl;
        plyfile << "property float z" << endl;
        plyfile << "end_header" << endl;
        
        //face 1
        for (int i = 0; i < dim.x; i++) {
            for (int j = 0; j < dim.y; j++) {
                plyfile << Voxels[i][j][0].x << " " << Voxels[i][j][0].y << " " << Voxels[i][j][0].z << endl;
            }
        }
        
        //face 2
        for (int i = 0; i < dim.x; i++) {
            for (int j = 0; j < dim.y; j++) {
                plyfile << Voxels[i][j][dim.z-1].x << " " << Voxels[i][j][dim.z-1].y << " " << Voxels[i][j][dim.z-1].z << endl;
            }
        }
        
        //face 3
        for (int i = 0; i < dim.x; i++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[i][0][k].x << " " << Voxels[i][0][k].y << " " << Voxels[i][0][k].z << endl;
            }
        }
        
        //face 4
        for (int i = 0; i < dim.x; i++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[i][dim.y-1][k].x << " " << Voxels[i][dim.y-1][k].y << " " << Voxels[i][dim.y-1][k].z << endl;
            }
        }
        
        //face 5
        for (int j = 0; j < dim.y; j++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[0][j][k].x << " " << Voxels[0][j][k].y << " " << Voxels[0][j][k].z << endl;
            }
        }
        
        //face 6
        for (int j = 0; j < dim.y; j++) {
            for (int k = 0; k < dim.z; k++) {
                plyfile << Voxels[dim.x-1][j][k].x << " " << Voxels[dim.x-1][j][k].y << " " << Voxels[dim.x-1][j][k].z << endl;
            }
        }
        
        
        /*for (int i = 1; i < dim.x-1; i++) {
            for (int j = 1; j < dim.y-1; j++) {
                for (int k = 1; k < dim.z-1; k++) {
                    plyfile << Voxels[i][j][k].x << " " << Voxels[i][j][k].y << " " << Voxels[i][j][k].z << endl;
                }
            }
        }*/

       //cout << "points written" << endl;
    }

    plyfile.close();
}

void SaveLevelSet(string path, float ***tsdf, my_int3 dim) {
    
    auto file = std::fstream(path, std::ios::out | std::ios::binary);
    int size = dim.x*dim.y*dim.z;
    file.write((char*)&size, sizeof(int));
    
    for (int i = 0; i < dim.x; i++) {
        for (int j = 0; j < dim.y; j++) {
            //file.seekg((i*dim.y + j)*dim.z*sizeof(float));
            file.write((char*)tsdf[i][j], dim.z*sizeof(float));
        }
    }
    
    file.close();
    
}

void SaveLevelSet(string path, float *tsdf, float *weights, int nb_nodes) {
    cout << "nb nodes: " << nb_nodes << endl;
    auto file = std::fstream(path, std::ios::out | std::ios::binary);
    file.write((char*)&nb_nodes, sizeof(int));
    file.write((char*)tsdf, nb_nodes*sizeof(float));
    //file.write((char*)weights, 24*nb_nodes*sizeof(float));
    file.close();
    
}

float *LoadTetLevelSet(string path, int nb_nodes) {
    
    auto file = std::fstream(path, std::ios::in | std::ios::binary);
    
    int size_sdf;
    file.read((char*)&size_sdf, sizeof(int));
    cout << "tsdf sie: " << size_sdf << endl;
    
    float *tsdf = new float[nb_nodes];
    file.read((char*)tsdf, nb_nodes*sizeof(float));
    file.close();
        
    return tsdf;
}

float ***LoadLevelSet(string path, my_int3 dim) {
    
    auto file = std::fstream(path, std::ios::in | std::ios::binary);
    
    int size_sdf;
    file.read((char*)&size_sdf, sizeof(int));
    
    float ***tsdf = new float **[dim.x];
    for (int i = 0; i < dim.x; i++) {
        tsdf[i] = new float *[dim.y];
        for (int j = 0; j < dim.y; j++) {
            tsdf[i][j] = new float[dim.z];
            //file.seekg((i*dim.y + j)*dim.z*sizeof(float));
            file.read((char*)tsdf[i][j], dim.z*sizeof(float));
        }
    }
    
    file.close();
    
    
    return tsdf;
}


int LoadPLY(string path, float **Vertices) {
      string line;
    ifstream plyfile (path);
    if (plyfile.is_open()) {

        // read Header
        getline (plyfile,line); // PLY
        getline (plyfile,line); // format ascii 1.0
        getline (plyfile,line); // comment ply file
        getline (plyfile,line); // element vertex ??

        // get number of vertices
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        int nb_vertices = std::stoi(words[2]);
        cout << "number of vertices: " << nb_vertices << endl;

        while (line != string("end_header")) {
            getline (plyfile,line);
        }

        /*getline (plyfile,line); // property float x
        getline (plyfile,line); // property float y
        getline (plyfile,line); // property float z
        getline (plyfile,line); // property float nx
        getline (plyfile,line); // property float ny
        getline (plyfile,line); // property float nz
        getline (plyfile,line); // element face 0
        getline (plyfile,line); // property list uchar int vertex_indices
        getline (plyfile,line); // end_header*/

        // Allocate data to store vertices
        *Vertices = new float[3*nb_vertices]();
        float *Buff = *Vertices;
        //int i = 0;

        for (int i = 0; i < nb_vertices; i++)
          //while ( getline (plyfile,line) )
        {
            getline (plyfile,line);
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff[3*i] = std::stof(words[0]);
            Buff[3*i+1] = std::stof(words[1]);
            Buff[3*i+2] = std::stof(words[2]);
            //i += 3;
        }

          plyfile.close();
        return nb_vertices;
    }

      plyfile.close();
    cout << "could not load file: " << path << endl;
    return -1;
}

// The normals correspond to the normals of the faces
int *LoadPLY_Mesh(string path, float **Vertices, float **Normals, int **Faces){
    int *result = new int[2];
    result[0] = -1; result[1] = -1;

    string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline (plyfile,line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        
        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("vertex")) == 0) {
                    result[0] = std::stoi(words[2]);
                } else if (words[1].compare(string("face")) == 0) {
                    result[1] = std::stoi(words[2]);
                }
            }
            
            getline (plyfile,line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        
        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        
        *Normals = new float[3*result[1]]();
        float *Buff_nmls = *Normals;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;

        for (int i = 0; i < result[0]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_vtx[3*i] = std::stof(words[0]);
            Buff_vtx[3*i+1] = std::stof(words[1]);
            Buff_vtx[3*i+2] = std::stof(words[2]);
            //cout << Buff_vtx[3*i] << ", " << Buff_vtx[3*i+1] << ", " << Buff_vtx[3*i+2] << endl;
            
        }

        for (int i = 0; i < result[1]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 3) {
                cout << "Error non triangle faces" << endl;
            }
            
            Buff_faces[3*i] = std::stoi(words[1]);
            Buff_faces[3*i+1] = std::stoi(words[2]);
            Buff_faces[3*i+2] = std::stoi(words[3]);
        }
        
        plyfile.close();
        
        cout << "Computing normals" << endl;
        
        for (int f = 0; f < result[1]; f++) {
            // compute face normal
            my_float3 p1 = make_my_float3(Buff_vtx[3*Buff_faces[3*f]], Buff_vtx[3*Buff_faces[3*f]+1], Buff_vtx[3*Buff_faces[3*f]+2]);
            my_float3 p2 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+1]], Buff_vtx[3*Buff_faces[3*f+1]+1], Buff_vtx[3*Buff_faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+2]], Buff_vtx[3*Buff_faces[3*f+2]+1], Buff_vtx[3*Buff_faces[3*f+2]+2]);
            
            my_float3 nml = cross(p2-p1, p3-p1);
            float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
            
            Buff_nmls[3*f] = nml.x/mag;
            Buff_nmls[3*f+1] = nml.y/mag;
            Buff_nmls[3*f+2] = nml.z/mag;
        }
        
        return result;
    }

    plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

int *LoadPLY_PyMesh(string path, float **Vertices, int **Faces, int **Voxels){
    int *result = new int[3];
    result[0] = -1; result[1] = -1; result[2] = -1;

      string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        // get number of vertices
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        result[0] = std::stoi(words[2]);
        cout << "number of vertices: " << result[0] << endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        // get number of faces
        std::istringstream iss2(line);
        std::vector<std::string> words2((std::istream_iterator<std::string>(iss2)), std::istream_iterator<std::string>());
        result[1]  = std::stoi(words2[2]);
        cout << "number of faces: " << result[1]  << endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        std::istringstream iss3(line);
        std::vector<std::string> words3((std::istream_iterator<std::string>(iss3)), std::istream_iterator<std::string>());
        result[2] = std::stoi(words3[2]);
        cout << "number of voxels: " << result[2] << endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        cout << line << endl;

        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
        
        *Voxels = new int[4*result[2]]();
        int *Buff_vox = *Voxels;

        for (int i = 0; i < result[0]; i++) {
            double x,y,z;
            plyfile.read((char *) &x, 8);
            plyfile.read((char *) &y, 8);
            plyfile.read((char *) &z, 8);
            Buff_vtx[3*i] = float(x);
            Buff_vtx[3*i+1] = float(y);
            Buff_vtx[3*i+2] = float(z);
        }

        for (int i = 0; i < result[1]; i++) {
            unsigned char a;
            int b;
            plyfile.read((char *) &a, 1);

            if (int(a) != 3) {
                cout << "Error non triangle faces" << endl;
            }

            plyfile.read((char *) &b, 4);
            Buff_faces[3*i] = b;
            plyfile.read((char *) &b, 4);
            Buff_faces[3*i+1] = b;
            plyfile.read((char *) &b, 4);
            Buff_faces[3*i+2] = b;
        }

        for (int i = 0; i < result[2]; i++) {
            unsigned char a;
            int b;
            plyfile.read((char *) &a, 1);

            if (int(a) != 4) {
                cout << "Error non tetrahedral voxel" << endl;
            }

            plyfile.read((char *) &b, 4);
            Buff_vox[4*i] = b;
            plyfile.read((char *) &b, 4);
            Buff_vox[4*i+1] = b;
            plyfile.read((char *) &b, 4);
            Buff_vox[4*i+2] = b;
            plyfile.read((char *) &b, 4);
            Buff_vox[4*i+3] = b;
        }

        plyfile.close();
        return result;
    }

      plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

int *LoadOBJ_Mesh(string path, float **Vertices, float **Normals, int **Faces){
    int *result = new int[2];
    result[0] = -1; result[1] = -1;

    string line;
    ifstream objfile (path, ios::binary);
    if (objfile.is_open()) {
        vector<float> vertices;
        vector<int> faces;
        
        while(getline (objfile,line)) {
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (words[0].compare(string("v")) == 0) {
                vertices.push_back(std::stof(words[1]));
                vertices.push_back(std::stof(words[2]));
                vertices.push_back(std::stof(words[3]));
            }
            
            if (words[0].compare(string("f")) == 0) {
                faces.push_back(std::stoi(words[1])-1);
                faces.push_back(std::stoi(words[2])-1);
                faces.push_back(std::stoi(words[3])-1);
            }
        }
        
        objfile.close();
        
        result[0] = vertices.size()/3;
        result[1] = faces.size()/3;
        
        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        memcpy(Buff_vtx, vertices.data(), 3*result[0]*sizeof(float));
        
        *Normals = new float[3*result[1]]();
        float *Buff_nmls = *Normals;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
        memcpy(Buff_faces, faces.data(), 3*result[1]*sizeof(int));
     
        cout << "Computing normals" << endl;
        
        for (int f = 0; f < result[1]; f++) {
            // compute face normal
            //cout << Buff_faces[3*f] << ", " << Buff_faces[3*f+1] << ", " << Buff_faces[3*f+2] << endl;
            my_float3 p1 = make_my_float3(Buff_vtx[3*Buff_faces[3*f]], Buff_vtx[3*Buff_faces[3*f]+1], Buff_vtx[3*Buff_faces[3*f]+2]);
            my_float3 p2 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+1]], Buff_vtx[3*Buff_faces[3*f+1]+1], Buff_vtx[3*Buff_faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(Buff_vtx[3*Buff_faces[3*f+2]], Buff_vtx[3*Buff_faces[3*f+2]+1], Buff_vtx[3*Buff_faces[3*f+2]+2]);
            
            my_float3 nml = cross(p2-p1, p3-p1);
            float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
            
            Buff_nmls[3*f] = nml.x/mag;
            Buff_nmls[3*f+1] = nml.y/mag;
            Buff_nmls[3*f+2] = nml.z/mag;
        }
        
        return result;
    }

    objfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

void LoadSkeletonVIBE(string template_skeleton, string filename_pose, Eigen::Matrix4f *Joints, int *KinTree_Table, int index) {
    
    float *Joints_T = new float[24*3]();
    ifstream Tfile (template_skeleton, ios::binary);
    if (!Tfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Tfile.close();
        return;
    }
    Tfile.read((char *) Joints_T, 24*3*sizeof(float));
    Tfile.close();
    
    /*
     ########### Search for the pose that face the camera the most
     */
    float *Angles = new float[24*3]();
    for (int j = 0; j < 3*24; j++)
        Angles[j] = 0.0f;
    float *Angles_tmp = new float[24*3]();
    float min_angle = 1.0e32;
    int best_cam = 0;
    for (int c = 1; c < 9; c++) {
        ifstream sklfile (filename_pose + "skeleton_cam_" + to_string(c) + "_" + to_string(index) + ".bin");
        if (!sklfile.is_open()) {
            cout << "Could not load skeleton of the model" << endl;
            sklfile.close();
            return;
        }
        sklfile.read((char *) Angles_tmp, 24*3*sizeof(float));
        sklfile.close();
        
        for (int j = 0; j < 3*24; j++)
            Angles[j] = Angles[j] + Angles_tmp[j];
        
        if (Angles[0] < min_angle) {
            min_angle = Angles[0];
            best_cam = c;
        }
    }
    
    delete[] Angles_tmp;
    
    for (int j = 0; j < 3*24; j++)
        Angles[j] = Angles[j]/8.0f;
    
    /*cout << "best cam: " << best_cam << endl;
    
    ifstream sklfile (filename_pose + "skeleton_cam_" + to_string(best_cam) + "_" + to_string(index) + ".bin");
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Angles, 24*3*sizeof(float));
    sklfile.close();*/
    
    /*for (int j = 0; j < 24; j++) {
        cout << "angle " << j << " : " << Angles[3*j] << ", " << Angles[3*j+1] << ", " << Angles[3*j+2] << endl;
    }*/
    
    //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[0]);
    //Eigen::Matrix4f rotation = euler2matrix(PI-Angles[0], Angles[1], Angles[2]);
    Eigen::Matrix4f rotation = euler2matrix(0.0f, 0.0f, 0.0f);
                
    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    //Eigen::Matrix4f jnt0_loc = _Pose.inverse() * Shift * Kinematics[0];
    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints[0] = jnt0_loc;

    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix

        //Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);
        Eigen::Matrix4f rotation = euler2matrix(Angles[3*j], Angles[3*j+1], Angles[3*j+2]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[3*KinTree_Table[2*j+1]] - Joints_T[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[3*KinTree_Table[2*j+1]+1] - Joints_T[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[3*KinTree_Table[2*j+1]+2] - Joints_T[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        //Eigen::Matrix4f jnt_loc = _Pose.inverse() * Shift * Kinematics[KinTree_Table[2*j+1]];
        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints[KinTree_Table[2*j+1]] = jnt_loc;
    }

    delete []KinTree_Table;
    delete []Kinematics;
    delete []Angles;
    delete []Joints_T;
}

void LoadSkeletonTEMPLATE(string template_skeleton, string filename_pose, Eigen::Matrix4f *Joints) {
    
    float *Template_rotations = new float[27*3]();
    ifstream sklfile (template_skeleton + "Rotations.bin", ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Template_rotations, 27*3*sizeof(float));
    sklfile.close();
    
    float *Template_translations = new float[27*3]();
    ifstream Tfile (template_skeleton + "Translations.bin", ios::binary);
    if (!Tfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Tfile.close();
        return;
    }
    Tfile.read((char *) Template_translations, 27*3*sizeof(float));
    Tfile.close();
        
    int *KinTree_Table = new int[27*2]();
    ifstream Kfile (template_skeleton + "KinematicTable.bin", ios::binary);
    if (!Kfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Kfile.close();
        return;
    }
    Kfile.read((char *) KinTree_Table, 27*2*sizeof(int));
    Kfile.close();
        
    //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[0]);
    Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[0]/180.0f, PI*Template_rotations[1]/180.0f, PI*Template_rotations[2]/180.0f);

    Joints[0] << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[2],
                    0.0f, 0.0f, 0.0f, 1.0f;
    cout << Joints[0] << endl;
    
    // build the template skeleton
    for (int j = 1; j < 27; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[3*j]);
        Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[3*j]/180.0f, PI*Template_rotations[3*j+1]/180.0f, PI*Template_rotations[3*j+2]/180.0f);
        //cout << Template_rotations[3*j] << ", " << Template_rotations[3*j+1] << ", " << Template_rotations[3*j+2] << endl;
        //cout << rotation << endl;

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[3*j],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[3*j+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[3*j+2],
                    0.0f, 0.0f, 0.0f, 1.0f;
        
        //cout << KinTree_Table[2*j] << ", " << KinTree_Table[2*j+1] << endl;
        
        Joints[KinTree_Table[2*j+1]] = Joints[KinTree_Table[2*j]] * Transfo;
        //cout << Joints[KinTree_Table[2*j+1]]  << endl;
    }

    delete []Template_rotations;
    delete []Template_translations;
    delete []KinTree_Table;
}

void LoadSkeletonARTICULATED(string template_skeleton, string filename_pose, Eigen::Matrix4f *Joints, int index) {
    
    float *Template_rotations = new float[27*3]();
    ifstream sklfileT (template_skeleton + "Rotations.bin", ios::binary);
    if (!sklfileT.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfileT.close();
        return;
    }
    sklfileT.read((char *) Template_rotations, 27*3*sizeof(float));
    sklfileT.close();
    
    float *Template_translations = new float[27*3]();
    ifstream Tfile (template_skeleton + "Translations.bin", ios::binary);
    if (!Tfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Tfile.close();
        return;
    }
    Tfile.read((char *) Template_translations, 27*3*sizeof(float));
    Tfile.close();
    
    ifstream sklfile (filename_pose);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    
    string line;
    getline (sklfile,line);
    getline (sklfile,line);
    getline (sklfile,line);
    for (int i = 0; i < index; i++)
        getline (sklfile,line);
    
    //cout << line << endl;
    std::istringstream iss(line);
    std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
    
    float *Rotations = new float[27*3]();
    Rotations[0] = stof(words[0]); Rotations[1] = stof(words[1]); Rotations[2] = stof(words[2]); // root
    Rotations[3] = stof(words[3]); Rotations[4] = stof(words[4]); Rotations[5] = stof(words[5]); // waist
    Rotations[6] = stof(words[6]); Rotations[7] = -stof(words[8]); Rotations[8] = stof(words[7]); // neck
    Rotations[9] = 0.0f; Rotations[10] = 0.0f; Rotations[11] = 0.0f; // head
    Rotations[12] = 0.0f; Rotations[13] = 0.0f; Rotations[14] = 0.0f; // head right
    Rotations[15] = 0.0f; Rotations[16] = 0.0f; Rotations[17] = 0.0f; // head left
    Rotations[18] = 0.0f; Rotations[19] = 0.0f; Rotations[20] = 0.0f; // head front
    Rotations[21] = 0.0f; Rotations[22] = 0.0f; Rotations[23] = 0.0f; // head back
    Rotations[24] = 0.0f; Rotations[25] = 0.0f; Rotations[26] = 0.0f; // head up
    Rotations[27] = stof(words[9]); Rotations[28] = stof(words[10]); Rotations[29] = 0.0f; // left sternum
    Rotations[30] = stof(words[11]); Rotations[31] = stof(words[12]); Rotations[32] = stof(words[13]); // left shoulder
    Rotations[33] = 0.0f; Rotations[34] = stof(words[14]); Rotations[35] = stof(words[15]); // left elbow
    Rotations[36] = -stof(words[16]); Rotations[37] = stof(words[17]); Rotations[38] = 0.0f; // left wrist
    Rotations[39] = 0.0f; Rotations[40] = 0.0f; Rotations[41] = 0.0f; // left hand
    Rotations[42] = stof(words[18]); Rotations[43] = stof(words[19]); Rotations[44] = 0.0f; // right sternum
    Rotations[45] = stof(words[20]); Rotations[46] = -stof(words[21]); Rotations[47] = -stof(words[22]); // right shoulder
    Rotations[48] = 0.0f; Rotations[49] = stof(words[23]); Rotations[50] = -stof(words[24]); // right elbow
    Rotations[51] = stof(words[25]); Rotations[52] = 0.0f; Rotations[53] = stof(words[26]); // right wrist
    Rotations[54] = 0.0f; Rotations[55] = 0.0f; Rotations[56] = 0.0f; // right hand
    Rotations[57] = stof(words[27]); Rotations[58] = stof(words[28]); Rotations[59] = stof(words[29]); // left hip
    Rotations[60] = stof(words[30]); Rotations[61] = 0.0f; Rotations[62] = stof(words[31]); // left knee
    Rotations[63] = stof(words[32]); Rotations[64] = 0.0f; Rotations[65] = 0.0f; // left ankle
    Rotations[66] = 0.0f; Rotations[67] = 0.0f; Rotations[68] = 0.0f; // left toe
    Rotations[69] = stof(words[33]); Rotations[70] = stof(words[34]); Rotations[71] = stof(words[35]); // right hip
    Rotations[72] = stof(words[36]); Rotations[73] = 0.0f; Rotations[74] = stof(words[37]); // right knee
    Rotations[75] = stof(words[38]); Rotations[76] = 0.0f; Rotations[77] = 0.0f; // right ankle
    Rotations[78] = 0.0f; Rotations[79] = 0.0f; Rotations[80] = 0.0f; // right toe
        
    Template_translations[0] = stof(words[39]); Template_translations[1] = stof(words[40]); Template_translations[2] = stof(words[41]);
    //Template_translations[0] = 0.0f; Template_translations[1] = 0.0f; Template_translations[2] = 0.0f;
        
    sklfile.close();
        
    int *KinTree_Table = new int[27*2]();
    ifstream Kfile (template_skeleton + "KinematicTable.bin", ios::binary);
    if (!Kfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        Kfile.close();
        return;
    }
    Kfile.read((char *) KinTree_Table, 27*2*sizeof(int));
    Kfile.close();
        
    //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[0]);
    Eigen::Matrix4f rotationP = euler2matrix(Rotations[0], Rotations[1], Rotations[2]);
    cout << Rotations[0] << ", " << Rotations[1] << ", " << Rotations[2] << endl;
    
    Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[0]/180.0f, PI*Template_rotations[1]/180.0f, PI*Template_rotations[2]/180.0f);
    cout << Template_rotations[0] << ", " << Template_rotations[1] << ", " << Template_rotations[2] << endl;

    Joints[0] << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[2],
                    0.0f, 0.0f, 0.0f, 1.0f;
    Joints[0] = Joints[0]*rotationP;
    /*Joints[0](0,3) = stof(words[39]);
    Joints[0](1,3) = stof(words[40]);
    Joints[0](2,3) = stof(words[41]);*/
    cout << Joints[0] << endl;
    
    // build the template skeleton
    for (int j = 1; j < 27; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        //Eigen::Matrix3f rotation = rodrigues2matrix(&Template_rotations[3*j]);
        Eigen::Matrix4f rotationP = euler2matrix(Rotations[3*j], Rotations[3*j+1], Rotations[3*j+2]);
        Eigen::Matrix4f rotation = euler2matrix(PI*Template_rotations[3*j]/180.0f, PI*Template_rotations[3*j+1]/180.0f, PI*Template_rotations[3*j+2]/180.0f);
        cout << Rotations[3*j] << ", " << Rotations[3*j+1] << ", " << Rotations[3*j+2] << endl;
        //cout << rotation << endl;

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Template_translations[3*j],
                    rotation(1,0), rotation(1,1), rotation(1,2), Template_translations[3*j+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Template_translations[3*j+2],
                    0.0f, 0.0f, 0.0f, 1.0f;
        
        //cout << KinTree_Table[2*j] << ", " << KinTree_Table[2*j+1] << endl;
        
        Joints[KinTree_Table[2*j+1]] = Joints[KinTree_Table[2*j]] * Transfo * rotationP;
        //cout << Joints[KinTree_Table[2*j+1]]  << endl;
    }

    //delete []Template_rotations;
    //delete []Template_translations;
    delete []KinTree_Table;
}

void LoadSkeleton(string filename_joints, string filename_pose, Eigen::Matrix4f *Joints, int *KinTree_Table) {
    float *Joints_T = new float[24*3]();
    ifstream sklfile (filename_joints, ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Joints_T, 24*3*sizeof(float));
    sklfile.close();
    
    ifstream psefile (filename_pose, ios::binary);
    if (!psefile.is_open()) {
        cout << "Could not load pose: " << filename_pose  << endl;
        psefile.close();
        return;
    }
    float *Angles = new float[3*24]();
    psefile.read((char *) Angles, 24*3*sizeof(float));
    psefile.close();
    
    /*for (int i = 0; i < 24; i++) {
        cout << "i: " << Angles[3*i] << ", " << Angles[3*i+1] << ", " << Angles[3*i+2] << endl;
    }*/

    //Eigen::Matrix3f rotation = _Rectify * rodrigues2matrix(&Angles[0]);
    Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[0]);

    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    //Eigen::Matrix4f jnt0_loc = _Pose.inverse() * Shift * Kinematics[0];
    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints[0] = jnt0_loc;

    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix

        Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints_T[3*KinTree_Table[2*j+1]] - Joints_T[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints_T[3*KinTree_Table[2*j+1]+1] - Joints_T[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints_T[3*KinTree_Table[2*j+1]+2] - Joints_T[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        //Eigen::Matrix4f jnt_loc = _Pose.inverse() * Shift * Kinematics[KinTree_Table[2*j+1]];
        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints[KinTree_Table[2*j+1]] = jnt_loc;
    }
    
    Eigen::Matrix3f center;
    center(0) = (Joints[3](0,3) + Joints[9](0,3))/2.0f;
    center(1) = (Joints[3](1,3) + Joints[9](1,3))/2.0f;
    center(2) = (Joints[3](2,3) + Joints[9](2,3))/2.0f;
    
    for (int j = 0; j < 24; j++) {
        Joints[j](0,3) = Joints[j](0,3) - center(0);
        Joints[j](1,3) = Joints[j](1,3) - center(1);
        Joints[j](2,3) = Joints[j](2,3) - center(2);
    }

    delete []Kinematics;
    delete []Angles;
    delete []Joints_T;
}

void LoadStarSkeleton(string filename, Eigen::Matrix4f *Joints_T, int *KinTree_Table, float angle_star = 0.5f) {
    float *Joints = new float[24*3]();
    ifstream sklfile (filename, ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Joints, 24*3*sizeof(float));
    sklfile.close();

    for (int i = 0; i < 24; i++) {
        Joints_T[i] << 1.0, 0.0, 0.0, Joints[3*i],
                         0.0, 1.0, 0.0, Joints[3*i+1],
                         0.0, 0.0, 1.0, Joints[3*i+2],
                          0.0, 0.0, 0.0, 1.0;
    }
    

    // Take the A pose
    float *Angles = new float[3*24]();

    for (int i = 0; i < 3*24; i++) {
        Angles[i] = 0.0f;
    }

    Angles[5] = angle_star;
    Angles[8] = -angle_star;
    
    /*for (int i = 0; i < 24; i++) {
        cout << "i: " << Angles[3*i] << ", " << Angles[3*i+1] << ", " << Angles[3*i+2] << endl;
    }*/

    Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[0]);

    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints_T[0] = jnt0_loc;


    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints[3*KinTree_Table[2*j+1]] - Joints[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[3*KinTree_Table[2*j+1]+1] - Joints[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[3*KinTree_Table[2*j+1]+2] - Joints[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints_T[KinTree_Table[2*j+1]] = jnt_loc;
    }
    return;
    
    // Compute joints orientations
    for (int i = 1; i < 7; i++) {
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 7; i < 12; i++) {
        if (i == 9)
            continue;
        float a = Joints_T[i](0,2);
        float b = Joints_T[i](1,2);
        float c = Joints_T[i](2,2);
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,2) = Joints_T[i](0,1);
        Joints_T[i](1,2) = Joints_T[i](1,1);
        Joints_T[i](2,2) = Joints_T[i](2,1);
        
        Joints_T[i](0,1) = a;
        Joints_T[i](1,1) = b;
        Joints_T[i](2,1) = c;
    }
    
    for (int i = 3; i < 9; i++) {
        if (i == 1 || i == 2 || i == 4 || i == 5 || i == 7 || i == 8 || i == 10 || i == 11)
            continue;
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 13; i < 24; i++) {
        if (i == 15)
            continue;
        if (i == 13 || i == 16 || i == 18 || i == 20 || i == 22) {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = -Joints_T[i](0,1);
            Joints_T[i](1,0) = -Joints_T[i](1,1);
            Joints_T[i](2,0) = -Joints_T[i](2,1);
            
            Joints_T[i](0,1) = a;
            Joints_T[i](1,1) = b;
            Joints_T[i](2,1) = c;
        } else {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = Joints_T[i](0,1);
            Joints_T[i](1,0) = Joints_T[i](1,1);
            Joints_T[i](2,0) = Joints_T[i](2,1);
            
            Joints_T[i](0,1) = -a;
            Joints_T[i](1,1) = -b;
            Joints_T[i](2,1) = -c;
        }
    }
    
    delete []Angles;
    delete []Kinematics;
    delete []Joints;
}

void LoadTSkeleton(string filename, Eigen::Matrix4f *Joints_T, int *KinTree_Table) {
    float *Joints = new float[24*3]();
    ifstream sklfile (filename, ios::binary);
    if (!sklfile.is_open()) {
        cout << "Could not load skeleton of the model" << endl;
        sklfile.close();
        return;
    }
    sklfile.read((char *) Joints, 24*3*sizeof(float));
    sklfile.close();

    for (int i = 0; i < 24; i++) {
        Joints_T[i] << 1.0, 0.0, 0.0, Joints[3*i],
                         0.0, 1.0, 0.0, Joints[3*i+1],
                         0.0, 0.0, 1.0, Joints[3*i+2],
                          0.0, 0.0, 0.0, 1.0;
    }
    

    // Take the A pose
    float *Angles = new float[3*24]();

    for (int i = 0; i < 3*24; i++) {
        Angles[i] = 0.0f;
    }

    Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[0]);

    Eigen::Matrix4f *Kinematics = new Eigen::Matrix4f[24]();
    Kinematics[0] << rotation(0,0), rotation(0,1), rotation(0,2), Joints[0],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[2],
                    0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::Matrix4f jnt0_loc = Kinematics[0];
    Joints_T[0] = jnt0_loc;


    for (int j = 1; j < 24; j++) {
        // transform current bone with angle following the kinematic table
        // Get rotation matrix
        Eigen::Matrix3f rotation = rodrigues2matrix(&Angles[3*j]);

        Eigen::Matrix4f Transfo;
        Transfo << rotation(0,0), rotation(0,1), rotation(0,2), Joints[3*KinTree_Table[2*j+1]] - Joints[3*KinTree_Table[2*j]],
                    rotation(1,0), rotation(1,1), rotation(1,2), Joints[3*KinTree_Table[2*j+1]+1] - Joints[3*KinTree_Table[2*j]+1],
                    rotation(2,0), rotation(2,1), rotation(2,2), Joints[3*KinTree_Table[2*j+1]+2] - Joints[3*KinTree_Table[2*j]+2],
                    0.0f, 0.0f, 0.0f, 1.0f;

        Kinematics[KinTree_Table[2*j+1]]= Kinematics[KinTree_Table[2*j]] * Transfo;

        Eigen::Matrix4f jnt_loc = Kinematics[KinTree_Table[2*j+1]];
        Joints_T[KinTree_Table[2*j+1]] = jnt_loc;
    }
    return;
    
    // Compute joints orientations
    for (int i = 1; i < 7; i++) {
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 7; i < 12; i++) {
        if (i == 9)
            continue;
        float a = Joints_T[i](0,2);
        float b = Joints_T[i](1,2);
        float c = Joints_T[i](2,2);
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,2) = Joints_T[i](0,1);
        Joints_T[i](1,2) = Joints_T[i](1,1);
        Joints_T[i](2,2) = Joints_T[i](2,1);
        
        Joints_T[i](0,1) = a;
        Joints_T[i](1,1) = b;
        Joints_T[i](2,1) = c;
    }
    
    for (int i = 3; i < 9; i++) {
        if (i == 1 || i == 2 || i == 4 || i == 5 || i == 7 || i == 8 || i == 10 || i == 11)
            continue;
        Joints_T[i](0,0) = -Joints_T[i](0,0);
        Joints_T[i](1,0) = -Joints_T[i](1,0);
        Joints_T[i](2,0) = -Joints_T[i](2,0);
        
        Joints_T[i](0,1) = -Joints_T[i](0,1);
        Joints_T[i](1,1) = -Joints_T[i](1,1);
        Joints_T[i](2,1) = -Joints_T[i](2,1);
    }
    
    for (int i = 13; i < 24; i++) {
        if (i == 15)
            continue;
        if (i == 13 || i == 16 || i == 18 || i == 20 || i == 22) {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = -Joints_T[i](0,1);
            Joints_T[i](1,0) = -Joints_T[i](1,1);
            Joints_T[i](2,0) = -Joints_T[i](2,1);
            
            Joints_T[i](0,1) = a;
            Joints_T[i](1,1) = b;
            Joints_T[i](2,1) = c;
        } else {
            float a = Joints_T[i](0,0);
            float b = Joints_T[i](1,0);
            float c = Joints_T[i](2,0);
            Joints_T[i](0,0) = Joints_T[i](0,1);
            Joints_T[i](1,0) = Joints_T[i](1,1);
            Joints_T[i](2,0) = Joints_T[i](2,1);
            
            Joints_T[i](0,1) = -a;
            Joints_T[i](1,1) = -b;
            Joints_T[i](2,1) = -c;
        }
    }
    
    delete []Angles;
    delete []Kinematics;
    delete []Joints;
}

int CleanFaces(float *Vertices, int *Faces, int nbFaces) {
    vector<int> merged_faces;
    for (int f = 0; f < nbFaces; f++) {
        if (Faces[3*f] == -1)
            continue;
        
        float dist1 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+1]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+1]]) +
                        (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+1]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+1]+1]) +
                        (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+1]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+1]+2]);
        
        float dist2 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+2]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f+2]]) +
                        (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+2]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f+2]+1]) +
                        (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+2]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f+2]+2]);
        
        float dist3 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f+2]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f+2]]) +
                        (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f+2]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f+2]+1]) +
                        (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f+2]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f+2]+2]);
        
        if (dist1 < 1.0e-8 || dist2 < 1.0e-8 || dist3 < 1.0e-8) {
            Faces[3*f] = -1;
            continue;
        }
        
        cout << f << " / " << nbFaces << endl;
        merged_faces.push_back(Faces[3*f]);
        merged_faces.push_back(Faces[3*f+1]);
        merged_faces.push_back(Faces[3*f+2]);
        
        for (int f2 = f+1; f2 < nbFaces; f2++) {
            if (Faces[3*f2] == -1)
                continue;
            
            float dist11 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2]]) +
                            (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2]+1]) +
                            (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2]+2]);
            
            float dist12 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+1]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+1]]) +
                            (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+1]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+1]+1]) +
                            (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+1]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+1]+2]);
            
            float dist13 = (Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+2]])*(Vertices[3*Faces[3*f]]-Vertices[3*Faces[3*f2+2]]) +
                            (Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+2]+1])*(Vertices[3*Faces[3*f]+1]-Vertices[3*Faces[3*f2+2]+1]) +
                            (Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+2]+2])*(Vertices[3*Faces[3*f]+2]-Vertices[3*Faces[3*f2+2]+2]);
            
            float dist21 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2]]) +
                            (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2]+1]) +
                            (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2]+2]);
            
            float dist22 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+1]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+1]]) +
                            (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+1]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+1]+1]) +
                            (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+1]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+1]+2]);
            
            float dist23 = (Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+2]])*(Vertices[3*Faces[3*f+1]]-Vertices[3*Faces[3*f2+2]]) +
                            (Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+2]+1])*(Vertices[3*Faces[3*f+1]+1]-Vertices[3*Faces[3*f2+2]+1]) +
                            (Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+2]+2])*(Vertices[3*Faces[3*f+1]+2]-Vertices[3*Faces[3*f2+2]+2]);
            
            float dist31 = (Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2]])*(Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2]]) +
                            (Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2]+1])*(Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2]+1]) +
                            (Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2]+2])*(Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2]+2]);
            
            float dist32 = (Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+1]])*(Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+1]]) +
                            (Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+1]+1])*(Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+1]+1]) +
                            (Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+1]+2])*(Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+1]+2]);
            
            float dist33 = (Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+2]])*(Vertices[3*Faces[3*f+2]]-Vertices[3*Faces[3*f2+2]]) +
                            (Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+2]+1])*(Vertices[3*Faces[3*f+2]+1]-Vertices[3*Faces[3*f2+2]+1]) +
                            (Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+2]+2])*(Vertices[3*Faces[3*f+2]+2]-Vertices[3*Faces[3*f2+2]+2]);
            
            
            if ((dist11 < 1.0e-8 || dist12 < 1.0e-8 || dist13 < 1.0e-8) &&
                (dist21 < 1.0e-8 || dist22 < 1.0e-8 || dist23 < 1.0e-8) &&
                (dist31 < 1.0e-8 || dist32 < 1.0e-8 || dist33 < 1.0e-8)) {
                cout << "F" << endl;
                cout << Vertices[3*Faces[3*f]] << ", " << Vertices[3*Faces[3*f]+1] << ", " << Vertices[3*Faces[3*f]+2] << endl;
                cout << Vertices[3*Faces[3*f+1]] << ", " << Vertices[3*Faces[3*f+1]+1] << ", " << Vertices[3*Faces[3*f+1]+2] << endl;
                cout << Vertices[3*Faces[3*f+2]] << ", " << Vertices[3*Faces[3*f+2]+1] << ", " << Vertices[3*Faces[3*f+2]+2] << endl;
                cout << "F2" << endl;
                cout << Vertices[3*Faces[3*f2]] << ", " << Vertices[3*Faces[3*f2]+1] << ", " << Vertices[3*Faces[3*f2]+2] << endl;
                cout << Vertices[3*Faces[3*f2+1]] << ", " << Vertices[3*Faces[3*f2+1]+1] << ", " << Vertices[3*Faces[3*f2+1]+2] << endl;
                cout << Vertices[3*Faces[3*f2+2]] << ", " << Vertices[3*Faces[3*f2+2]+1] << ", " << Vertices[3*Faces[3*f2+2]+2] << endl;
                Faces[3*f2] = -1;
            }
        }
    }
    memcpy(Faces, merged_faces.data(), merged_faces.size()*sizeof(int));
    int res = merged_faces.size()/3;
    merged_faces.clear();
    cout << nbFaces << " -> " << res << endl;
    return res;
}

float *ComputeNormalsFaces(float *Vertices, int *Faces, int nbFaces) {
    float *Normals = new float[3*nbFaces];
    
    for (int f = 0; f < nbFaces; f++) {
        // compute face normal
        my_float3 p1 = make_my_float3(Vertices[3*Faces[3*f]], Vertices[3*Faces[3*f]+1], Vertices[3*Faces[3*f]+2]);
        my_float3 p2 = make_my_float3(Vertices[3*Faces[3*f+1]], Vertices[3*Faces[3*f+1]+1], Vertices[3*Faces[3*f+1]+2]);
        my_float3 p3 = make_my_float3(Vertices[3*Faces[3*f+2]], Vertices[3*Faces[3*f+2]+1], Vertices[3*Faces[3*f+2]+2]);
        
        my_float3 nml = cross(p2-p1, p3-p1);
        float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
        
        if (mag > 0.0f) {
            Normals[3*f] = nml.x/mag;
            Normals[3*f+1] = nml.y/mag;
            Normals[3*f+2] = nml.z/mag;
        } else {
            Normals[3*f] = 0.0f;
            Normals[3*f+1] = 0.0f;
            Normals[3*f+2] = 0.0f;
        }
    }
    return Normals;
}

void UpdateNormalsFaces(float *Vertices, float *Normals, int *Faces, int nbFaces) {
    for (int f = 0; f < nbFaces; f++) {
        // compute face normal
        my_float3 p1 = make_my_float3(Vertices[3*Faces[3*f]], Vertices[3*Faces[3*f]+1], Vertices[3*Faces[3*f]+2]);
        my_float3 p2 = make_my_float3(Vertices[3*Faces[3*f+1]], Vertices[3*Faces[3*f+1]+1], Vertices[3*Faces[3*f+1]+2]);
        my_float3 p3 = make_my_float3(Vertices[3*Faces[3*f+2]], Vertices[3*Faces[3*f+2]+1], Vertices[3*Faces[3*f+2]+2]);
        
        my_float3 nml = cross(p2-p1, p3-p1);
        float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
        
        Normals[3*f] = nml.x/mag;
        Normals[3*f+1] = nml.y/mag;
        Normals[3*f+2] = nml.z/mag;
    }
}

void UpdateNormals(float *Vertices, float *Normals, int *Faces, int nbVertices, int nbFaces) {
    
    for (int f = 0; f < nbVertices; f++) {
        Normals[3*f] = 0.0f;
        Normals[3*f+1] = 0.0f;
        Normals[3*f+2] = 0.0f;
    }
    
    for (int f = 0; f < nbFaces; f++) {
        // compute face normal
        my_float3 p1 = make_my_float3(Vertices[3*Faces[3*f]], Vertices[3*Faces[3*f]+1], Vertices[3*Faces[3*f]+2]);
        my_float3 p2 = make_my_float3(Vertices[3*Faces[3*f+1]], Vertices[3*Faces[3*f+1]+1], Vertices[3*Faces[3*f+1]+2]);
        my_float3 p3 = make_my_float3(Vertices[3*Faces[3*f+2]], Vertices[3*Faces[3*f+2]+1], Vertices[3*Faces[3*f+2]+2]);
        
        my_float3 nml = cross(p2-p1, p3-p1);
        Normals[3*Faces[3*f]] = Normals[3*Faces[3*f]] + nml.x;
        Normals[3*Faces[3*f]+1] = Normals[3*Faces[3*f]+1] + nml.y;
        Normals[3*Faces[3*f]+2] = Normals[3*Faces[3*f]+2] + nml.z;
        
        Normals[3*Faces[3*f+1]] = Normals[3*Faces[3*f+1]] + nml.x;
        Normals[3*Faces[3*f+1]+1] = Normals[3*Faces[3*f+1]+1] + nml.y;
        Normals[3*Faces[3*f+1]+2] = Normals[3*Faces[3*f+1]+2] + nml.z;
        
        Normals[3*Faces[3*f+2]] = Normals[3*Faces[3*f+2]] + nml.x;
        Normals[3*Faces[3*f+2]+1] = Normals[3*Faces[3*f+2]+1] + nml.y;
        Normals[3*Faces[3*f+2]+2] = Normals[3*Faces[3*f+2]+2] + nml.z;
    }
    
    for (int f = 0; f < nbVertices; f++) {
        my_float3 nml = make_my_float3(Normals[3*f], Normals[3*f+1], Normals[3*f+2]);
        float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
        
        if (mag > 0.0f) {
            Normals[3*f] = Normals[3*f]/mag;
            Normals[3*f+1] = Normals[3*f+1]/mag;
            Normals[3*f+2] = Normals[3*f+2]/mag;
        }
    }
}

void SkinArticulatedMesh(float *Vertices, float *SkinWeights, DualQuaternion *Skeleton, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
            
        // Blend the transformations with iterative algorithm
        DualQuaternion Transfo = DualQuaternion(Quaternion(0.0,0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,0.0));
        for (int j = 0; j < 24; j++) {
            if (SkinWeights[24*v+j] > 0.0)
                Transfo = Transfo + (Skeleton[j] * SkinWeights[24*v+j]);
        }
                
        Transfo = Transfo.Normalize();
        DualQuaternion point = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), Quaternion(Vertices[3*v], Vertices[3*v+1], Vertices[3*v+2], 0.0f));
        point  = Transfo * point * Transfo.DualConjugate2();

        my_float3 vtx = point.Dual().Vector();
        Vertices[3*v] = vtx.x;
        Vertices[3*v+1] = vtx.y;
        Vertices[3*v+2] = vtx.z;
    }
}

void SkinMesh(float *Vertices, float *SkinWeights, DualQuaternion *Skeleton, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
            
        // Blend the transformations with iterative algorithm
        DualQuaternion Transfo = DualQuaternion(Quaternion(0.0,0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,0.0));
        for (int j = 0; j < 24; j++) {
            if (SkinWeights[24*v+j] > 0.0)
                Transfo = Transfo + (Skeleton[j] * SkinWeights[24*v+j]);
        }
                
        Transfo = Transfo.Normalize();
        DualQuaternion point = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), Quaternion(Vertices[3*v], Vertices[3*v+1], Vertices[3*v+2], 0.0f));
        point  = Transfo * point * Transfo.DualConjugate2();

        my_float3 vtx = point.Dual().Vector();
        Vertices[3*v] = vtx.x;
        Vertices[3*v+1] = vtx.y;
        Vertices[3*v+2] = vtx.z;
    }
}

void SkinMeshARTICULATED(float *Vertices, float *SkinWeights, DualQuaternion *Skeleton, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
            
        // Blend the transformations with iterative algorithm
        DualQuaternion Transfo = DualQuaternion(Quaternion(0.0,0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,0.0));
        for (int j = 0; j < 26; j++) {
            if (SkinWeights[26*v+j] > 0.0)
                Transfo = Transfo + (Skeleton[j] * SkinWeights[26*v+j]);
        }
                
        Transfo = Transfo.Normalize();
        DualQuaternion point = DualQuaternion(Quaternion(0.0,0.0,0.0,1.0), Quaternion(Vertices[3*v], Vertices[3*v+1], Vertices[3*v+2], 0.0f));
        point  = Transfo * point * Transfo.DualConjugate2();

        my_float3 vtx = point.Dual().Vector();
        Vertices[3*v] = vtx.x;
        Vertices[3*v+1] = vtx.y;
        Vertices[3*v+2] = vtx.z;
    }
}

void transferSkinWeights(float *VerticesA, float *SkinWeightsA, float *VerticesB, float *SkinWeightsB, int nbVerticesA, int nbVerticesB) {
    for (int v1 = 0; v1 < nbVerticesA; v1++) {
        my_float3 p1 = make_my_float3(VerticesA[3*v1], VerticesA[3*v1+1], VerticesA[3*v1+2]);
        float min_dist = 1.0e32;
        for (int v2 = 0; v2 < nbVerticesB; v2++) {
            my_float3 p2 = make_my_float3(VerticesB[3*v2], VerticesB[3*v2+1], VerticesB[3*v2+2]);
            float dist = norm(p1-p2);
            if (dist < min_dist) {
                min_dist = dist;
                for (int s = 0; s < 24; s++) {
                    SkinWeightsA[24*v1+s] = SkinWeightsB[24*v2+s];
                }
            }
        }
    }
}

// The normals correspond to the normals of the faces
int *LoadPLY_TetAndSurface(string path, float **Vertices, float **Normals, int **Faces, float **Nodes, int **Surface_edges, int **Tetra){
    int *result = new int[5];
    result[0] = -1; result[1] = -1; result[2] = -1; result[3] = -1; result[4] = -1;
    
    string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline (plyfile,line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        
        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("vertex")) == 0) {
                    result[0] = std::stoi(words[2]);
                    cout << "nb vertices: " << result[0] << endl;
                } else if (words[1].compare(string("face")) == 0) {
                    result[1] = std::stoi(words[2]);
                    cout << "nb face: " << result[1] << endl;
                } else if (words[1].compare(string("nodes")) == 0) {
                    result[2] = std::stoi(words[2]);
                    cout << "nb nodes: " << result[2] << endl;
                } else if (words[1].compare(string("surface")) == 0) {
                    result[3] = std::stoi(words[2]);
                    cout << "nb surface: " << result[3] << endl;
                } else if (words[1].compare(string("voxel")) == 0) {
                    result[4] = std::stoi(words[2]);
                    cout << "nb voxel: " << result[4] << endl;
                }
            }
            
            getline (plyfile,line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        
        // Allocate data
        *Vertices = new float[3*result[0]]();
        float *Buff_vtx = *Vertices;
        
        *Normals = new float[3*result[0]]();
        float *Buff_nml = *Normals;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
        
        *Nodes = new float[3*result[2]]();
        float *Buff_nodes = *Nodes;
        
        *Surface_edges = new int[2*result[3]]();
        int *Buff_surf = *Surface_edges;
                
        *Tetra = new int[4*result[4]]();
        int *Buff_tetra = *Tetra;

        for (int i = 0; i < result[0]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_vtx[3*i] = std::stof(words[0]);
            Buff_vtx[3*i+1] = std::stof(words[1]);
            Buff_vtx[3*i+2] = std::stof(words[2]);
            Buff_nml[3*i] = std::stof(words[3]);
            Buff_nml[3*i+1] = std::stof(words[4]);
            Buff_nml[3*i+2] = std::stof(words[5]);
        }        
        
        for (int i = 0; i < result[1]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 3) {
                cout << "Error non triangular face" << endl;
            }
            
            Buff_faces[3*i] = std::stof(words[1]);
            Buff_faces[3*i+1] = std::stof(words[2]);
            Buff_faces[3*i+2] = std::stof(words[3]);
        }
        
        for (int i = 0; i < result[2]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_nodes[3*i] = std::stof(words[0]);
            Buff_nodes[3*i+1] = std::stof(words[1]);
            Buff_nodes[3*i+2] = std::stof(words[2]);
        }
        

        for (int i = 0; i < result[4]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 4) {
                cout << "Error non tetrahedron" << endl;
            }
            
            Buff_tetra[4*i] = std::stoi(words[1]);
            Buff_tetra[4*i+1] = std::stoi(words[2]);
            Buff_tetra[4*i+2] = std::stoi(words[3]);
            Buff_tetra[4*i+3] = std::stoi(words[4]);
        }
        
        for (int i = 0; i < result[3]; i++) {
            getline (plyfile,line); // PLY
            //cout << line << endl;
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 2) {
                cout << "Error non edge" << endl;
            }
            
            //cout << i << ": " << words[1] << ", " << words[2] << endl;
            
            Buff_surf[2*i] = std::stoi(words[1]);
            Buff_surf[2*i+1] = std::stoi(words[2]);
        }
        
        plyfile.close();
        return result;
    }

    plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

int *LoadPLY_Tet(string path, float **Nodes, int **Faces, int **Tetra){
    int *result = new int[3];
    result[0] = -1; result[1] = -1; result[2] = -1;
    
    string line;
    ifstream plyfile (path, ios::binary);
    if (plyfile.is_open()) {
        //Read header
        getline (plyfile,line); // PLY
        //cout << line << endl;
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        
        while (words[0].compare(string("end_header")) != 0) {
            if (words[0].compare(string("element")) == 0) {
                if (words[1].compare(string("vertex")) == 0) {
                    result[0] = std::stoi(words[2]);
                    cout << "nb vertices: " << result[0] << endl;
                } else if (words[1].compare(string("face")) == 0) {
                    result[1] = std::stoi(words[2]);
                    cout << "nb face: " << result[1] << endl;
                } else if (words[1].compare(string("voxel")) == 0) {
                    result[2] = std::stoi(words[2]);
                    cout << "nb voxel: " << result[2] << endl;
                }
            }
            
            getline (plyfile,line); // PLY
            //cout << line << endl;
            iss = std::istringstream(line);
            words.clear();
            words = std::vector<std::string>((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        }
        
        // Allocate data
        *Nodes = new float[3*result[0]]();
        float *Buff_nodes = *Nodes;
        
        *Faces = new int[3*result[1]]();
        int *Buff_faces = *Faces;
                
        *Tetra = new int[4*result[2]]();
        int *Buff_tetra = *Tetra;

        for (int i = 0; i < result[0]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            Buff_nodes[3*i] = std::stof(words[0]);
            Buff_nodes[3*i+1] = std::stof(words[1]);
            Buff_nodes[3*i+2] = std::stof(words[2]);
        }
        
        for (int i = 0; i < result[1]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 3) {
                cout << "Error non triangular face" << endl;
            }
            
            Buff_faces[3*i] = std::stof(words[1]);
            Buff_faces[3*i+1] = std::stof(words[2]);
            Buff_faces[3*i+2] = std::stof(words[3]);
        }

        for (int i = 0; i < result[2]; i++) {
            getline (plyfile,line); // PLY
            std::istringstream iss(line);
            std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
            
            if (std::stoi(words[0]) != 4) {
                cout << "Error non tetrahedron" << endl;
            }
            
            Buff_tetra[4*i] = std::stoi(words[1]);
            Buff_tetra[4*i+1] = std::stoi(words[2]);
            Buff_tetra[4*i+2] = std::stoi(words[3]);
            Buff_tetra[4*i+3] = std::stoi(words[4]);
        }
        
        plyfile.close();
        return result;
    }

    plyfile.close();
    cout << "could not load file: " << path << endl;
    return result;
}

void LoadLevelSet(string path, float *tsdf, float **weights, int nb_nodes) {
    
    ifstream file (path, ios::binary);
    file.read((char*) tsdf, nb_nodes*sizeof(float));
    //file.write((char*)weights, 24*nb_nodes*sizeof(float));
    file.close();
    
}

void CenterMesh(float *Vertices, float *Root, int nbVertices) {
    for (int v = 0; v < nbVertices; v++) {
        if (Vertices[3*v] == 0.0f && Vertices[3*v+1] == 0.0f && Vertices[3*v+2] == 0.0f)
            continue;
        
        Vertices[3*v] = Vertices[3*v] - Root[0];
        Vertices[3*v+1] = Vertices[3*v+1] - Root[1];
        Vertices[3*v+2] = Vertices[3*v+2] - Root[2];
    }
}

#endif /* MeshUtils_h */
