import numpy as np
import glob
from natsort import natsorted
import struct
import sys
from Mylib.meshes_utils import load_ply , save_ply , load_obj_retopo
import os
import tqdm

def load_smpl_npz(smpl_path): 
    smpl_dict = np.load(smpl_path)
    smpl_pose = smpl_dict['pose']
    smpl_trans = smpl_dict['transl']
    return smpl_pose , smpl_trans

def load_smpl_bin(smpl_path):
    f = open(smpl_path,"rb")
    skel_bin = f.read()
    smpl_list = []
    """for i in range(24):
        joint = []
        for j in range(3):
            joint.append(struct.unpack("f",skel_bin[4*(3*i+j):4*(3*i+j)+4])[0])
        smpl_list.append(joint)"""
    for i in range(24*3):
        smpl_list.append(struct.unpack("f",skel_bin[4*i:4*i+4])[0])
    smpl = np.array(smpl_list)
    smpl = smpl.astype(np.float32)
    return smpl

def load_smpl_bin_trans(smpl_path):
    f = open(smpl_path,"rb")
    skel_bin = f.read()
    trans_list = []
    """for i in range(24):
        joint = []
        for j in range(3):
            joint.append(struct.unpack("f",skel_bin[4*(3*i+j):4*(3*i+j)+4])[0])
        smpl_list.append(joint)"""
    for i in range(3):
        trans_list.append(struct.unpack("f",skel_bin[4*i:4*i+4])[0])
    trans = np.array(trans_list)
    trans = trans.astype(np.float32)
    return trans
#pose_folder_path = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\gLO_sBM_cAll_d14_mLO1_ch05\seqs"
#mesh_dir_path      = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\1_4D_data\3_all\reconst_interp_on_mesh_ttt"
#pose_folder_path = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\Joyful_Jump\seqs"
#mesh_dir_path      = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\2_I_crane\3_all\reconst_interp_on_mesh"
#pose_folder_path = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\Dancing_Twerk\seqs"
#mesh_dir_path      = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\3_CAPE_00159_shortshort\3_all\reconst_interp_on_mesh"
#pose_folder_path    = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\gLO_sBM_cAll_d14_mLO1_ch05\seqs"
#mesh_dir_path       = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\4_CAPE_03375_shortlong\3_all\reconst_interp_on_mesh"
#pose_folder_path = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\Dancing_1\seqs"
#mesh_dir_path      = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\5_Female_ballet_dancer\3_all\reconst_interp_on_mesh"
#pose_folder_path = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\Dancing_2\seqs"
#mesh_dir_path      = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\6_Male_ballet_dancer\3_all\reconst_interp_on_mesh"
#pose_folder_path = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\Standing_Taunt_Battlecry\seqs"
#mesh_dir_path      = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\7_Muscle_ref_bundle\3_all\reconst_interp_on_mesh"
pose_folder_path = r"D:\Data\Human\HUAWEI\test_motion_list\motion_lists\Taunt\seqs"
mesh_dir_path      = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\OurResult_BMBC_0725\8_Muscle_range_of_motion\3_all\reconst_interp_on_mesh"

#meshes_path = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\result_2path\cape\reconst_all\IPLMesh_*ply"
#save_path_headname = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\result_2path\cape\reconst_all\moved_mesh_"

meshes_path = os.path.join(mesh_dir_path , r"IPLMesh_*ply")
save_path_headname = os.path.join(mesh_dir_path , r"moved_mesh_") 


#pose_paths = natsorted(glob.glob(root_path))
mesh_paths = natsorted(glob.glob(meshes_path)) 

for path in mesh_paths:
    baseID = os.path.basename(path).split(".")[0].split("_")[-1]
    baseID5 = str(int(baseID)).zfill(5)
    baseID4 = str(int(baseID)).zfill(4)
    smpl_path = os.path.join(pose_folder_path , baseID5 + ".npz")
    #smpl_path = os.path.join(pose_folder_path ,"pose_" + baseID4 + ".bin")
    smpl_pose , smpl_trans = load_smpl_npz(smpl_path)
    #smpl_pose = load_smpl_bin(smpl_path)
    #trans_path = os.path.join(pose_folder_path ,"trans_" + baseID4 + ".bin")
    #smpl_trans = load_smpl_bin_trans(trans_path)
    vtx , nml , rgb , face , vtx_num , face_num = load_ply(path)
    #vtx , nml , txr , face , vtx2txr , vtx2nml , vtx_num , face_num = load_obj_retopo(path)
    new_vtx = []
    #for i in tqdm.tqdm(range(vtx_num)):
    for i in range(vtx_num):
        new_vtx_x = vtx[i][0] + smpl_trans[0]
        new_vtx_y = vtx[i][1] + smpl_trans[1]
        new_vtx_z = vtx[i][2] + smpl_trans[2]
        new_vtx.append([new_vtx_x,new_vtx_y,new_vtx_z])
    
    save_path = save_path_headname + baseID + ".ply"
    save_ply(save_path, new_vtx , None , None , face , vtx_num , face_num)
    
    print("src_path  : "  , path)
    print("save_path : " , save_path)
