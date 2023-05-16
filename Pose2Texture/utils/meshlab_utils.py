import pymeshlab
import glob
from natsort import natsorted
import os
import tqdm
import polyscope as ps

def resampling_mesh(path, save_path , cellsize = None , is_percent = True):
    """
    In meshlab , Filters/Remeshing , .../Uniform mesh resampling
    In out case , we used this to increase vertices for renderpeople datasets
    """
    print("load_path : " , path)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path) 
    

    if is_percent == True:
        pt=pymeshlab.Percentage(cellsize)                 #define percentage object
    else:
        curr_mesh = ms.current_mesh()
        box = curr_mesh.bounding_box()
        diag = box.diagonal()
        pt=pymeshlab.Percentage((cellsize / diag) * 100)    #define percentage object

    #if abs is not None:

    print("resampling start . . .")
    ms.generate_resampled_uniform_mesh(cellsize = pt)
    print("resampling finish!")

    ms.save_current_mesh(save_path) 
    print("saved to : " , save_path)

def Surface_reconstruction_ball_pivoting(path, save_path , ballradius = 0 , clustering = 20 , creasethr = 90 , deletefaces = False):
    """
    In meshlab , Filters/Remeshing , .../Surface_reconstruction:ball_pivoting
    ballradius : Pivoting Ball radius (0 autoguess)   
    clustering : Clustering radius (% of ball radius) 
    deletefaces : Delete initial set of faces
    """

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path) 

    ballradius=pymeshlab.Percentage(ballradius)                 #define percentage object
    print("surface_reconstruction_ball_pivoting start . . .")
    ms.surface_reconstruction_ball_pivoting(ballradius = ballradius , clustering = clustering , creasethr = creasethr , deletefaces = deletefaces)
    print("surface_reconstruction_ball_pivoting finish!")

    ms.save_current_mesh(save_path , binary = False) 
    print("saved to : " , save_path)

def Surface_reconstruction_screened_Poisson(path, save_path , depth = 8 , fulldepth  = 5 , cgdepth = 0 , scale = 1.1 , samplespernode = 1.5 , pointweight = 4 , iters = 8):
    """
    In meshlab , Filters/Remeshing , .../Surface_reconstruction:screened_poisson
    """

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path) 

    print("Surface_reconstruction_screened_Poisson start . . .")
    ms.generate_surface_reconstruction_screened_poisson(depth = depth , fulldepth  = fulldepth , cgdepth = cgdepth , scale = scale, samplespernode = samplespernode , pointweight = pointweight , iters = iters)
    print("Surface_reconstruction_screened_Poisson finish!")

    ms.save_current_mesh(save_path , binary = False) 
    print("saved to : " , save_path)

def ICP(src_path, target_path , save_path ):
    """
    In meshlab , Filters/Remeshing , .../ICP Between Meshes 
    """

    ms = pymeshlab.MeshSet()
    print("src_path : " , src_path)
    ms.load_new_mesh(src_path) 

    print("target_path : " , target_path)
    ms.load_new_mesh(target_path) 

    #pymeshlab.MeshSet.apply_filter(ms , "compute_matrix_by_icp_between_meshes" , referencemesh = 1 , sourcemesh = 0 )
    ms.compute_matrix_by_icp_between_meshes(referencemesh = 1 , sourcemesh = 0 , samplenum  = 15000)
    print("compute_matrix_by_icp_between_meshes finish!")
    ms.set_current_mesh(0)
    ms.apply_matrix_freeze()
    ms.save_current_mesh(save_path , binary = False) 
    print("saved to : " , save_path)
    #ms.show_polyscope()

def snapshot(dir_path , save_dir_path ):        #FIXME:not working
    paths = natsorted(glob.glob(dir_path))
    for path in paths:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path) 
        save_path = os.path.join(save_dir_path , "test.png")
        ms.save_snapshot(imagebackgroundcolor = [50, 50, 50, 255] , imagewidth = 512, imageheights = 512, imagefilename = save_path)


if __name__ == "__main__":
    #how to use ICP
    """
    src_dir_path = r"Z:\Human\b20-kitamura\AvatarInTheShell\Before_EG_results\OurResult_BMBC_0725\1_4D_data\2_extrapolation\reconst_interp_on_mesh_ttt"
    target_dir   = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Iwamoto\data\data_org\centered_meshes" 
    src_paths    = natsorted(glob.glob(os.path.join(src_dir_path , r"mesh_[0-9][0-9][0-9][0-9].ply")))
    save_dir     = src_dir_path
    for src_path in tqdm.tqdm(src_paths):
        #basename = os.path.splitext(os.path.basename(src_path))[0]
        basename = os.path.splitext(os.path.basename(src_path))[0].split("_")[-1].zfill(4)
        save_path = os.path.join(save_dir , "ICPMesh_" + basename + ".ply")

        basename = os.path.splitext(os.path.basename(src_path))[0].split("_")[-1].zfill(4)
        target_path = os.path.join(target_dir , "mesh_" + basename + ".ply")
        #save_path = os.path.join(save_dir , "IniPosiPosedMesh_" + basename + ".ply")

        ICP(src_path , target_path , save_path) 
    """

    #how to use resampling_mesh
    #cell size
    #percentage = 0.230
    #percentage = 0.300
    abs = 0.006
    
    #paths = natsorted(glob.glob(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_POP\AITS\00215_poloshort\All_src_test_meshes\poloshort_tilt_twist_left\[0-9][0-9][0-9][0-9].obj"))
    #paths = natsorted(glob.glob(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_POP\AITS\03375_blazerlong\All_src_test_meshes\blazerlong_volleyball_trial2\smpl_[0-9][0-9][0-9][0-9].obj"))
    paths = natsorted(glob.glob(r"Z:\Human\b20-kitamura\AvatarInTheShell\master_thesis_results\Cape_POP\POP_results\blazerlong_volleyball_trial1\reconst_itp_depth8\mesh_[0-9][0-9][0-9][0-9].ply"))
    
    #paths = natsorted(glob.glob(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\debug\src_meshes\mesh_[0-9][0-9][0-9][0-9].ply"))
    #save_dir                  = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Iwamoto_0220\data\resampled_meshes"
    #save_dir = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_POP\AITS\03375_blazerlong\blazerlong_climb_trial1\retarget_motion\volleyball_trial2\data\resampled_smpls"
    save_dir = r"Z:\Human\b20-kitamura\AvatarInTheShell\master_thesis_results\Cape_POP\POP_results\blazerlong_volleyball_trial1\reconst_itp_depth8\resample_test"
    for path in tqdm.tqdm(paths):
        basename = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(save_dir , basename + ".obj")
        resampling_mesh(path , save_path , abs , is_percent = False)
        
    #how to use snapshot
    """
    dir_path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\3D_datasets\CAPE-33\data\centered_meshes\*.ply"
    save_path = "debug"
    snapshot(dir_path , save_path)
    """

    #how to use Surface_reconstruction with ball_pivoting
    """
    paths    = natsorted(glob.glob(r"D:\Project\Human\POP\data\cape\packed\03375_shortlong\train\*.ply"))
    save_dir = r"D:\Project\Human\POP\data\cape\packed\03375_shortlong\train\surface_reconst"
    for path in tqdm.tqdm(paths):
        basename = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(save_dir , basename + ".ply")
        Surface_reconstruction_ball_pivoting(path , save_path)
    """

    #how to use Surface_reconstruction with screened_Poisson
    """
    paths    = natsorted(glob.glob(r"D:\Project\Human\POP\data\cape\packed\03375_shortlong\train\*.ply"))
    save_dir = r"D:\Project\Human\POP\data\cape\packed\03375_shortlong\train\surface_reconst"
    for path in tqdm.tqdm(paths):
        basename = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(save_dir , basename + ".ply")
        Surface_reconstruction_screened_Poisson(path , save_path)
    """