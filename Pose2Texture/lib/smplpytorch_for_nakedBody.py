import os
import glob
import numpy as np
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from numba import jit,cuda , njit
import trimesh
from natsort import natsorted
import tqdm
import sys
import pandas as pd
import tempfile

if __name__ == "__main__":
    import skinning_utils
    from meshes_utils import load_ply , save_ply
else:
    from lib import skinning_utils


def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

@cuda.jit
def Dilatation_smpl_to_Outershell_gpu(Outershell , smpl_vtxs , bary_cod, correspond_face , vtx_num):
    i = cuda.grid(1)
    if i < vtx_num:
        for j in range(3):
            Outershell[i][j]  = bary_cod[i * 3    ] * smpl_vtxs[correspond_face[i * 3    ]][j] + \
                              + bary_cod[i * 3 + 1] * smpl_vtxs[correspond_face[i * 3 + 1]][j] + \
                              + bary_cod[i * 3 + 2] * smpl_vtxs[correspond_face[i * 3 + 2]][j]

class smplpytorch_processor():
    def __init__(self , model_path = None , gender = 'neutral'):
        self.smplpytorch_cuda = False
        self.model_path       = model_path
        self.gender           = gender

    def rescale_smpl_model(self , Tshapecoarsejoints , joint_T , kintree):
        '''
        Tshapecoarsejoints : smpl T-pose 
        joint_T            : Subject's T-pose
        kintree            : Parent-child ordered list
        '''
        
        Tshapecoarsejoints = Tshapecoarsejoints.reshape(-1,3)
        joints_smpl        = skinning_utils.LoadStarSkeleton(Tshapecoarsejoints, kintree, 0.0)
        
        joint_T = joint_T.reshape(-1,3)
        joint_T_processed  = skinning_utils.LoadStarSkeleton(joint_T, kintree, 0.0)  
        
        ###----------------Calculate translation(Joints(=smpl t_pose) -> Joints_T(=data_depend t_pose))-###
        skeleton_t = np.zeros([24,4,4] , np.float32)
        for i in range(24):
            skeleton_t[i] = np.dot(joints_smpl[i] , skinning_utils.InverseTransfo(joint_T_processed[i]))  


        smpl = pd.read_pickle(os.path.join(self.model_path , r"basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"))
        smpl_vtx = smpl["v_template"]
        vtx_num  = len(smpl_vtx)
        rescaled_vtx     = skinning_utils.SkinMeshLBS(smpl_vtx , smpl['weights'] , skeleton_t , vtx_num , True)
        smpl["v_template"] = rescaled_vtx

        self.tmp_save_dir  = tempfile.TemporaryDirectory()
        if self.gender == 'neutral':
            save_path = os.path.join(self.tmp_save_dir.name , "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
        elif self.gender == 'male':
            save_path = os.path.join(self.tmp_save_dir.name , "basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
        elif self.gender == 'female':
            save_path = os.path.join(self.tmp_save_dir.name , "basicModel_f_lbs_10_207_0_v1.0.0.pkl")
        pd.to_pickle(smpl , save_path)
        self.model_path  = self.tmp_save_dir.name

    def init_SMPL_layer(self):
        # Create the SMPL layer
        self.smpl_layer = SMPL_Layer(
            center_idx=None,                           
            gender=self.gender,
            model_root=os.path.abspath(self.model_path))
        
        if self.smplpytorch_cuda:
            self.smpl_layer.cuda()

    #root_path =  "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/test_motion_list/motion_lists/Dancing_1"
    def predict_smpl_by_smplpytorch(self , pose_params , shape_params = None):  #pose_path):
        batch_size = 1

        # Generate random pose and shape parameters
        pose_params = pose_params.reshape((1,72))
        pose_params = torch.from_numpy(pose_params.astype(np.float32))
        
        if shape_params is not None:
            shape_params = torch.from_numpy(shape_params[np.newaxis , :].astype(np.float32))
        else:
            shape_params = torch.zeros(batch_size, 10) 
        
                
        # Forward from the SMPL layer
        verts, Jtr = self.smpl_layer(pose_params, th_betas=shape_params)
    
        #debug
        """
        _ , _  , _ , smpl_face , _ , _ = load_ply("Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.015-0.05/Template_skin.ply")
        MESH = trimesh.base.Trimesh(vertices = verts[0].to('cpu').detach().numpy().copy()  , faces = smpl_face)
        MESH.export("./outer.obj")
        """
        return verts[0].cuda()
    
    def Dilatation_smpl_to_Outershell_gpu_caller(self , smpl_vtxs , bary_cod , correspond_face , vtx_num = 93894):
        threads_per_block = 64  #i.e) block dim
        blocks_per_grid = int(divUp(vtx_num,threads_per_block))     #i.e) grid dim  

        Outershell_vtxs                     = cuda.to_device(np.zeros((vtx_num , 3) , dtype= "float32"))
        smpl_vtxs                           = cuda.as_cuda_array(smpl_vtxs) 
              

        Dilatation_smpl_to_Outershell_gpu[blocks_per_grid,threads_per_block](Outershell_vtxs , smpl_vtxs , bary_cod, correspond_face , vtx_num)

        #debug
        """
        _ , _  , _ , smpl_face , _ , _ = load_ply("Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.015-0.05/Template_skin.ply")
        MESH = trimesh.base.Trimesh(vertices = Outershell_vtxs  , faces = smpl_face)
        MESH.export("./outer2.obj")
        """
        
        return Outershell_vtxs
        
if __name__ == "__main__":
    #paths = natsorted(glob.glob("D:/Project/Human/AITS/avatar-in-the-shell/not_important_utils/debug_pose/pose_*.bin"))
    paths = natsorted(glob.glob("Z:/Human/b20-kitamura/AvatarInTheShell_datasets/3D_datasets/CAPE-33/data/smplparams/pose_*.bin"))
    #paths = natsorted(glob.glob("Z:/Human/b20-kitamura/AvatarInTheShell_datasets/4D_datasets/MIT/MIT/data/smplparams/pose_*.bin"))
    #paths = natsorted(glob.glob("Z:/Human/b20-kitamura/AvatarInTheShell_datasets/3D_datasets/Female_ballet_dancer/data/smplparams/pose_*.bin"))
    #paths = natsorted(glob.glob("Z:/Human/b20-kitamura/AvatarInTheShell/Naked_pipeline_test_2022_11_10/test_dataset/smplparams/pose_*.bin"))
    #path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\test_motion_list\motion_lists\Dancing_1\seqs\00001.npz"
    
    SMPL_and_Outershell_Bary_path        = os.path.abspath("../../assets/heavy/SMPL_and_Outershell_Barycentric.bin")
    SMPL_and_Outershell_Corr_Face_path   = os.path.abspath("../../assets/heavy/SMPL_and_Outershell_CorrespondFace.bin")
    _ , _  , _ , smpl_face , _ , _       = load_ply("Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.015-0.05/Template_skin.ply")
    _ , _  , _ , Outershell_face , _ , Outershell_face_num = load_ply(r"D:\Project\Human\AITS\avatar-in-the-shell\assets\heavy\shellb_template.ply")
    #out_dir = "Z:/Human/b20-kitamura/AvatarInTheShell/Naked_pipeline_test_2022_11_10/test_dataset/GTmesh"
    #if os.path.exists(out_dir) == False:
    #    os.mkdir(out_dir)

    kintree              = np.fromfile("../../assets/kintree.bin" , np.int32)
    Tshapecoarsejoints   = np.fromfile("../../assets/Tshapecoarsejoints.bin" , np.float32)           #smpl t_pose (position)
    #T_joints            = np.fromfile(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\3D_datasets\Female_ballet_dancer\data\Canonical_Mesh_folder_2path\T_joints.bin" , np.float32)   
    #T_joints            = np.fromfile(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Iwamoto_0220\data\smplparams\T_joints.bin" , np.float32)   #data depend t_pose (position)
    T_joints            = np.fromfile(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_POP\AITS\03375_blazerlong\blazerlong_volleyball_trial1\data\smplparams_pop\T_joints.bin" , np.float32)   #data depend t_pose (position)
    #T_joints             = np.fromfile(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\MIT\MIT\data\smplparams\T_joints.bin" , np.float32)   #data depend t_pose (position)
    
    skinWeights         = skinning_utils.load_skinweight_bin(r"D:\Project\Human\AITS\avatar-in-the-shell\assets\heavy\weights_template.bin" , 6890)
    
    smplpytorch_processor_instance = smplpytorch_processor("D:/Project/Human/AITS/avatar-in-the-shell/assets/heavy")
    #smplpytorch_processor_instance = smplpytorch_processor("Z:/Human/b20-kitamura/AvatarInTheShell_datasets/SMPLmodel/SMPL_python_v.1.0.0/smpl/models" , 'male')
    #smplpytorch_processor_instance.rescale_smpl_model(Tshapecoarsejoints , T_joints , kintree )
    smplpytorch_processor_instance.init_SMPL_layer()

    bary_cod_cuda        = cuda.to_device(np.fromfile(SMPL_and_Outershell_Bary_path      , np.float32))   
    correspond_face_cuda = cuda.to_device(np.fromfile(SMPL_and_Outershell_Corr_Face_path , np.int32))   
    
    """
    for i , path in tqdm.tqdm(enumerate(paths)):
        #pose = np.load(path)['pose']
        pose = np.fromfile(path , np.float32)
        #pose = skinning_utils.fix_smpl_parts(pose, fix_hands = True , fix_foots = True , fix_wrists = False)
        smpl_verts           = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(pose)

        #debug(SMPL)
        MESH = trimesh.base.Trimesh(vertices = smpl_verts.to('cpu').detach().numpy().copy()  , faces = np.array(smpl_face))
        MESH.export(os.path.join("debug_smpl" , str(i).zfill(4) + "_smpl.obj"))
        sys.exit()
        
        Outershell_vtxs = smplpytorch_processor_instance.Dilatation_smpl_to_Outershell_gpu_caller(smpl_verts , bary_cod_cuda , correspond_face_cuda)

        #debug
        MESH = trimesh.base.Trimesh(vertices = Outershell_vtxs  , faces = np.array(Outershell_face))
        MESH.export(os.path.join(os.path.join("debug_smpl" ,str(i).zfill(4) + "_outershell.obj")))
    """

    ###
    """
    import time
    paths = natsorted(glob.glob(r"D:\Project\Human\snarf\data\compare_1128\DeformedSMPL\*.ply"))
    save_dir = "D:\Project\Human\snarf\data\compare_1128\DeformedSMPL"
    for i , path in tqdm.tqdm(enumerate(paths)):
        smpl_verts , nml , rgb , face , vtx_num , face_num = load_ply(path)
        time1 = time.time()
        Outershell_vtxs = smplpytorch_processor_instance.Dilatation_smpl_to_Outershell_gpu_caller(smpl_verts , bary_cod_cuda , correspond_face_cuda)
        time2 = time.time() 
        print("time2 : " , time2 - time1)
        #debug
        save_ply(os.path.join(save_dir  ,"mesh_" + str(i).zfill(4) + ".ply") , Outershell_vtxs.copy_to_host(), None , None , Outershell_face , 93894 , Outershell_face_num)
        #MESH = trimesh.base.Trimesh(vertices = Outershell_vtxs  , faces = np.array(Outershell_face))
        #MESH.export(os.path.join(save_dir ,"mesh_" + str(i).zfill(4) + ".ply"))
        time3 = time.time() 
        print("time3 : " ,time3 - time2)
    """

    #path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_SNARF\AITS\03375\blazerlong_volleyball_trial1\data\smplparams\pose_0157.binfitted_model.pkl"
    #path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Iwamoto_0220\data\smplparams_test\pose_0208.binfitted_model.pkl"
    path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_POP\AITS\03375_blazerlong\blazerlong_volleyball_trial1\data\smplparams_pop\pose_0050.binfitted_model.pkl"
    #path  = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_POP\AITS\03375_blazerlong\blazerlong_volleyball_trial1\data\smplparams_pop\pose_0073.binfitted_model.pkl"
    #path  = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Iwamoto\data\data_org\smplparams\pose_0000.binfitted_model.pkl"
    fitted_pkl = pd.read_pickle(path)
    pose  = np.array(fitted_pkl["pose"])
    beta  = np.array(fitted_pkl["betas"])
    trans = np.array(fitted_pkl['trans'])
    #print(pose.shape)
    #print(betas.shape)
    #print(trans.shape)

    pose = np.zeros([24,3])
    #beta = None

    #path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_SNARF\SNARF\03375\blazerlong_volleyball_trial1\blazerlong_volleyball_trial1.000157.npz"
    #path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_SNARF\SNARF\03375\blazerlong_volleyball_trial1\blazerlong_volleyball_trial1.000147.npz"
    #fitted_npz = np.load(path)
    #pose = fitted_npz["pose"]
    #beta = fitted_npz["betas"]
    #trans = fitted_npz["transl"]
    #print(tran)
    #pose_path = r"D:\Downloads\tmp_aaa\pose_0070.bin"
    #pose_path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Iwamoto\data\data_org\smplparams\pose_0170.bin"
    #pose_path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_SNARF\AITS\03375\blazerlong_volleyball_trial1\data\smplparams\pose_0157.bin"
    
    #pose_path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\test_motion_list\test_motion_list\motion_lists\gLO_sBM_cAll_d14_mLO1_ch05\seqs\pose_00160.bin"
    #pose = np.fromfile(pose_path , np.float32)
    #trans = np.fromfile(r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Iwamoto\data\data_org\smplparams\trans_0200.bin" , np.float32)
    #pose[0:3] = np.zeros(3 , np.float32)
    #smpl_verts             = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(pose , beta).to('cpu').detach().numpy().copy()
    smpl_verts              = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(pose).to('cpu').detach().numpy().copy()

    MESH = trimesh.base.Trimesh(vertices = smpl_verts  , faces = np.array(smpl_face))
    MESH.export("debug.obj")

    """
    for i in range(smpl_verts.shape[0]):
        smpl_verts[i] = smpl_verts[i] + trans

    MESH = trimesh.base.Trimesh(vertices = smpl_verts  , faces = np.array(smpl_face))
    MESH.export("debug_trans.obj")
    """