import numpy as np
import random
import open3d as o3d
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fit SMPL body to vertices.')
    parser.add_argument('--path', type=str, default="D:/Human_data/4D_data/data/canonical_meshes/",
                        help='Path to the 3D skeleton that corresponds to the 3D scan.')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Flag to record intermediate results')
    parser.add_argument('--size', type=int, default=210,
                        help='nb of scans')
                        
    args = parser.parse_args()
    path = args.path
        
    for idx in range(0,args.size):
        #scan_path = path + "geometry/4dviews_nobi_retopo_" + str(idx).zfill(4) + ".obj"
        if not os.path.isfile(os.path.join(path , "PC_" + str(idx).zfill(4) + ".ply")):
            print("not find : " , os.path.join(path , "PC_" + str(idx).zfill(4) + ".ply"))
            continue
            
        pcd = o3d.io.read_point_cloud(os.path.join(path , "PC_" + str(idx).zfill(4) + ".ply"))
        print(pcd)
                                      
        print('run Poisson surface reconstruction')
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9)
        #vertices_to_remove = densities < np.quantile(densities, 0.01)
        #mesh.remove_vertices_by_mask(vertices_to_remove)
        print(mesh)
        o3d.io.write_triangle_mesh(os.path.join(path , "mesh_" + str(idx).zfill(4) + ".ply"), mesh, write_ascii=True)
    
