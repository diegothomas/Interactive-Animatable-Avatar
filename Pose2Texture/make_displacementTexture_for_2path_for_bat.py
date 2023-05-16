from scipy import interpolate
import os
import glob
from natsort import natsorted
import numpy as np
import sys
from matplotlib import pyplot as plt
import tqdm
from gooey import Gooey
from gooey import GooeyParser
import cv2
from scipy.linalg import norm
import time
from scipy import ndimage
import shutil
import copy
from numba import jit,cuda , njit
import pandas as pd
import torch

from lib.meshes_utils import load_ply , load_obj ,save_obj ,save_ply
from lib import make_displacementTexture_utils , texture_avg_gpu , skinning_utils , smplpytorch_for_nakedBody

import math
PI = math.pi

@cuda.jit('void(float32[:,:,:] , float32[:,:,:,:] , int8[:,:,:] , float32[:,:] , int8)')      # h*w*3*k , h*w*k
def Mymedian_cuda(in_texture , out_texture , valid_texture, mask, kernel_size):
    h,w = cuda.grid(2)

    height = in_texture.shape[0]
    width = in_texture.shape[1]

    r = int(kernel_size/2)
    counter = 0
    if mask[h,w] == 1.0:
        for j in range(kernel_size):
            y = h + j - r
            for i in range(kernel_size):
                counter += 1
                x = w + i - r
                if y < 0 or y > height-1 or x < 0 or x > width -1:
                    out_texture[h,w][0][counter] = 0.0
                    out_texture[h,w][1][counter] = 0.0
                    out_texture[h,w][2][counter] = 0.0
                    valid_texture[h,w][counter] = 0
                else:
                    out_texture[h,w][0][counter] = in_texture[y,x][0]
                    out_texture[h,w][1][counter] = in_texture[y,x][1]
                    out_texture[h,w][2][counter] = in_texture[y,x][2]
                    if mask[y,x] == 1.0:
                        valid_texture[h,w][counter] = 1
                    else:
                        valid_texture[h,w][counter] = 0
                    

def Mymedian_gpu(out_texture , valid_texture , in_texture , mask, kernel_size):
    height = in_texture.shape[0]
    width  = in_texture.shape[0]
    threads_per_block = 16 , 16   #i.e) block dim
    blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim 
    
    if mask is not None:
        Mymedian_cuda[blocks_per_grid,threads_per_block](in_texture ,out_texture , valid_texture ,mask ,kernel_size)
    else:                                                   
        raise AssertionError("No mask")
    return out_texture , valid_texture


@cuda.jit('void(float32[:,:,:] , float32[:,:,:] , float32[:,:] , float32[:,:] , float32[:,:])')
def Mygaussian_cuda(in_texture , out_texture , mask, kernel ,sum):
    h,w = cuda.grid(2)

    #initialize
    out_texture[h,w][0] = 0.0
    out_texture[h,w][1] = 0.0
    out_texture[h,w][2] = 0.0
    sum[h,w]            = 0.0

    height = in_texture.shape[0]
    width = in_texture.shape[1]

    kernel_size = kernel.shape[0]

    r = int(kernel_size/2)
    counter = 0
    if mask[h,w] == 1.0:
        for j in range(kernel_size):
            y = h + j - r
            if y < 0 or y > height-1 :
                continue
            for i in range(kernel_size):
                x = w + i - r
                if x < 0 or x > width -1:
                    continue
                if mask[y,x] == 1.0:
                    out_texture[h,w][0] += in_texture[y,x][0] * kernel[j,i]
                    out_texture[h,w][1] += in_texture[y,x][1] * kernel[j,i]
                    out_texture[h,w][2] += in_texture[y,x][2] * kernel[j,i]

                    sum[h,w] += kernel[j,i]
                    counter += 1
        if counter > 0:
            out_texture[h,w][0] = out_texture[h,w][0] / sum[h,w] 
            out_texture[h,w][1] = out_texture[h,w][1] / sum[h,w]
            out_texture[h,w][2] = out_texture[h,w][2] / sum[h,w]
        else:
            out_texture[h,w][0] = in_texture[h,w][0]
            out_texture[h,w][1] = in_texture[h,w][1]
            out_texture[h,w][2] = in_texture[h,w][2]



###Gaussian
def Make_kernel(kernel_size = 3 , sigma = 1 , device = None):
    if kernel_size % 2 != 1:
        print("kernel size need to be set Odd!")
        sys.exit()
    #make kernel
    Kernel = np.zeros((kernel_size , kernel_size) , np.float32)
    gauss_sum = 0.0

    for x in range( - int(kernel_size / 2) , int(kernel_size / 2) + 1):
        for y in range( - int(kernel_size / 2) , int(kernel_size / 2 + 1)):
            Kernel[y + int(kernel_size / 2)][x + int(kernel_size / 2)] = np.exp(-((x*x + y*y) / (2 * sigma * sigma)))/(2 * PI * sigma * sigma)
            gauss_sum += Kernel[y + int(kernel_size / 2)][x + int(kernel_size / 2)] 

    Kernel = Kernel / gauss_sum

    r = int(kernel_size / 2)
    
    if device == None:
        return Kernel
    else:
        return cuda.to_device(Kernel)

def Mygaussian_gpu(out_texture ,in_texture , sum , mask, kernel):
    height = in_texture.shape[0]
    width  = in_texture.shape[0]
    threads_per_block = 16 , 16   #i.e) block dim
    blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim 
    
    if mask is not None:
        
        Mygaussian_cuda[blocks_per_grid,threads_per_block](in_texture ,out_texture ,mask ,kernel, sum)
    else:                                                   
        raise AssertionError("No mask")
    return out_texture

    
def display_local_coordinate_system_as_color_on_ply(posed_basis , vtx , face, vtx_num , face_num, save_folder_path , file_subtitle):
    f1_rgb = []
    f2_rgb = []
    f3_rgb = []
    for fs in posed_basis :
        f1_rgb.append([int(((fs[0][0]+1)/2)*255),int(((fs[0][1]+1)/2)*255),int(((fs[0][2]+1)/2)*255)])
        f2_rgb.append([int(((fs[1][0]+1)/2)*255),int(((fs[1][1]+1)/2)*255),int(((fs[1][2]+1)/2)*255)])
        f3_rgb.append([int(((fs[2][0]+1)/2)*255),int(((fs[2][1]+1)/2)*255),int(((fs[2][2]+1)/2)*255)])

    save_path_f1 = os.path.join(save_folder_path , file_subtitle + "_local_coordinate_system_f1.ply")
    save_path_f2 = os.path.join(save_folder_path , file_subtitle + "_local_coordinate_system_f2.ply")
    save_path_f3 = os.path.join(save_folder_path , file_subtitle + "_local_coordinate_system_f3.ply")

    save_ply(save_path_f1 ,vtx , None ,f1_rgb , face , vtx_num , face_num)
    save_ply(save_path_f2 ,vtx , None ,f2_rgb , face , vtx_num , face_num)
    save_ply(save_path_f3 ,vtx , None ,f3_rgb , face , vtx_num , face_num)

def display_local_coordinate_system_as_arrow_on_ply(posed_basis , vtx , face, vtx_num , face_num, save_folder_path , file_subtitle , sampling = 10):
    new_vtx  = []
    new_face = []
    new_rgb  = []
    j = 0
    for i , v in enumerate(vtx):
        if i % sampling == 0:
            new_vtx.append(v)
            new_vtx.append(v + posed_basis[i][0] * 0.01)
            new_vtx.append(v + posed_basis[i][0] * 0.01)
            new_vtx.append(v + posed_basis[i][1] * 0.01)
            new_vtx.append(v + posed_basis[i][1] * 0.01)
            new_vtx.append(v + posed_basis[i][2] * 0.01)
            new_vtx.append(v + posed_basis[i][2] * 0.01)
            new_rgb.append([128 , 128 , 128])
            new_rgb.append([255 , 128 , 128])
            new_rgb.append([255 , 128 , 128])
            new_rgb.append([128 , 255 , 128])
            new_rgb.append([128 , 255 , 128])
            new_rgb.append([128 , 128 , 255])
            new_rgb.append([128 , 128 , 255])
            new_face.append([j , j + 1 , j + 2])
            new_face.append([j , j + 3 , j + 4])
            new_face.append([j , j + 5 , j + 6])
            j += 7
        
    save_path = os.path.join(save_folder_path , file_subtitle + "_local_coordinate_system.ply")
    save_ply(save_path ,new_vtx , None ,new_rgb , new_face , len(new_vtx) , len(new_face))


def display_displacement_edge(fitted_vtx , Naked_vtx , diplace_color , save_path , sampling = 20):
    new_vtx  = []
    new_face = []
    new_rgb  = []
    j = 0

    for i , v1 in enumerate(fitted_vtx):
        if i%sampling != 0 or np.all(v1 == np.zeros(3)):
            continue
        new_vtx.append(v1)
        new_vtx.append(Naked_vtx[i])
        new_vtx.append(Naked_vtx[i])
        new_rgb.append(diplace_color[i,:3])
        new_rgb.append(diplace_color[i,:3])
        new_rgb.append(diplace_color[i,:3])
        new_face.append([j , j + 1 , j + 2])
        j += 3
    save_ply(save_path ,new_vtx , None ,new_rgb , new_face , len(new_vtx) , len(new_face))

def calc_standardization(texture,mu_disp,gamma_disp, nonzero_id , alpha ,shift = True):
    height = texture.shape[0]
    width = texture.shape[1]
    standarded = np.zeros([height,width,3])
    standarded[nonzero_id] = ((texture[nonzero_id] - mu_disp[nonzero_id]) / (alpha *gamma_disp[nonzero_id]))
    if shift:
        standarded[nonzero_id] += 0.5
        return standarded
    else:
        return standarded

def calc_normalization(texture,mu_disp,gamma, nonzero_id , alpha ,shift = True):
    height = texture.shape[0]
    width = texture.shape[1]
    standarded = np.zeros([height,width,3])
    standarded[nonzero_id] = ((texture[nonzero_id] - mu_disp[nonzero_id]) / (alpha * gamma))

    if shift:
        #standarded[nonzero_id] += 0.5
        standarded += 0.5
        return standarded
    else:
        return standarded

def img_standardization(texture_ax,mu_ax,gamma_ax,nonzero_id_ax,alpha,shift = True):
    height = texture_ax.shape[0]
    width = texture_ax.shape[1]
    standarded = np.zeros([height,width])

    standarded[nonzero_id_ax] = ((texture_ax[nonzero_id_ax] - mu_ax) / (alpha * gamma_ax))
    if shift:
        standarded[nonzero_id_ax] += 0.5
        return standarded
    else:
        return standarded

def cut_over0and1(texture):
    texture = np.where(texture>1, 1, texture) 
    texture = np.where(texture<0, 0, texture) 
    return texture

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

def load_testID(test_folder_path):
    f = open(test_folder_path,"r")
    test_folder = f.read()
    test_folder_list = test_folder.split("\n")
    return list(map(int,test_folder_list))
    
#@Gooey(tabbed_groups=True,default_size=(800, 1000))
def main():
  
    parser = GooeyParser(description='Regress texture from pose')

    main_group = parser.add_argument_group(
    "main Option", 
    "main option"
    )

    sub_group = parser.add_argument_group(
    "sub Options", 
    "Basically fixed"
    )
    
    sub_group.add_argument(
        '--asset_path',
        default= r"..\assets",
        type=str,
        help="")

    main_group.add_argument(
        '--root_path',
        default= None,
        #default = "D:\Data\Human\HUAWEI\MIT\MIT\data",
        type=str,
        help='Where logs and weights will be saved  , example = "D:\Data\Human\HUAWEI\MIT\MIT\data",')
    
    main_group.add_argument(
        '--dataset_type',
        default= 'ours',
        #default = "D:\Data\Human\HUAWEI\MIT\MIT\data",
        type=str,
        help="'ours' : (default). Use smpl pickle files containing pose, beta, and trans. This pickle file is used when we want to train and test with smpl parameters obtained from smpl fitting. \
              'cape' : Use only pose file.  When this selection is made, the smpl is created at the original location. Therefore, even without the translation parameter, make sure that the output result of smplpytorch matches the FittedMesh before using it. \
              'pop'  : same with ours , but folder name is smplparams_pop")

    main_group.add_argument(
        '--gender',
        default= None,
        #default = "D:\Data\Human\HUAWEI\MIT\MIT\data",
        type=str,
        help="'neutral' or 'male' or 'female' , Only applies to cape dataset. We use the Neutral model for our dataset." )
    
    main_group.add_argument(
        '--use_beta',
        default= None,
        type=str,
        help="'True → use beta param for smpl shape \
               False → use t_joints for smpl shape" )

    main_group.add_argument(
        '--save_average_texture',
        default= "True",
        type=str,
        help='"True" → save average texture , "False" → load average texture')
    
    main_group.add_argument(
        '--Training_type',
        default = None,
        type = str,
        help="select from True , False",
        widget='Dropdown',
        choices=["Interpolation","Extrapolation","All","Debug"]
        )

    main_group.add_argument(
        '--useID',
        default = "True",
        type = str,
        help="select from True , False",
        widget='Dropdown',
        choices=["True","False"]
        )

    main_group.add_argument(
        '--IDstart',
        default = 2,
        help="when use ID start for Extrapolation , Interpolation. If the number is default(-999), it is ignored.",
        type = int,
        )

    main_group.add_argument(
        '--IDend',
        default = -999,
        #default = 70,
        help="when use ID end for Extrapolation , Interpolation. If the number is default(-999), it is ignored.",
        type = int,
        )

    main_group.add_argument(
        '--IDstep',
        default = -999,
        #default = 5,
        help="when use ID steps for Extrapolation , Interpolation. If the number is default(-999), it is ignored.",
        type = int,
        )
        
    main_group.add_argument(
        '--trainID_folder_path',
        default = None,
        help="when use ID steps for Extrapolation , Interpolation. If the number is default(-999), it is ignored.",
        type = str,
        )

    main_group.add_argument(
        '--duplicate_smpl',
        default = "True",
        type = str,
        help="duplicate smpl pose to 3channels for 3d datasets(like female_dancer_datasets)",
        )

    main_group.add_argument(
        '--use_disp_flg',
        default= "True",
        type=str,
        help='Where logs and weights will be saved')

    main_group.add_argument(
        '--use_color_flg',
        default= "False",
        type=str,
        help='Where logs and weights will be saved')
    
    main_group.add_argument(
        '--ColorTexture_from_raw_flg',
        default= "True",
        type=str,
        help='If True, make color texture from raw')

    main_group.add_argument(
        '--color_raw_texture_path',
        default= None,
        #default = "D:\Data\Human\HUAWEI\MIT\MIT\OBJ_data",
        type=str,
        help='Where logs and weights will be saved  , example = "D:\Data\Human\HUAWEI\Iwamoto\OBJ_data",')
    
    sub_group.add_argument(
        '--norm_a',
        default= 4,
        type=int,
        help='')
   
    main_group.add_argument(
        '--mode',
        default= 1,
        type=int,
        help="Choose from \"0\" : original , or \"1\" : 2path(use original_local_coordinate_system) , or \"2\" : 2path(define new_local_coordinate system)")

    sub_group.add_argument(
        '--height',
        default= 768,
        type=int,
        help='Where logs and weights will be saved')

    sub_group.add_argument(
        '--width',
        default= 768,
        type=int,
        help='Where logs and weights will be saved')

    sub_group.add_argument(
        '--Orthonormalize',
        default= "False",
        type=str,
        help='Where logs and weights will be saved')
    
    sub_group.add_argument(
        '--Debug',
        default= "False",
        type=str,
        help='Where logs and weights will be saved')

    args = parser.parse_args()
    start = time.time()

    ###----------------Path&Parameter Setting----------------------------------------------###
    FittedMesh_pathlist          = natsorted(glob.glob(os.path.join(args.root_path,r"FittedMesh_pose\IniPosiPosedMesh_*.ply")))
    #PosedSurface_path_headname  = os.path.join(args.root_path,r"FittedMesh_pose\SkinnedSurface_")
    mode = args.mode
    dataset_type = args.dataset_type
    if dataset_type == "cape":
        pose_folder_path             = os.path.join(args.root_path ,r"smplparams_cape")       #for cape
        gender       = args.gender
    elif dataset_type == "pop":
        pose_folder_path             = os.path.join(args.root_path, r"smplparams_pop")       #for our smpl fitting parameter
        gender = 'neutral'
    elif dataset_type == "ours":
        pose_folder_path             = os.path.join(args.root_path, r"smplparams")       #for our smpl fitting parameter
        gender = 'neutral'

    smplparam_path_headname           = os.path.join(pose_folder_path , "pose_")

    #fit_path =  glob.glob(os.path.join(pose_folder_path , "pose_*.binfitted_model.pkl"))[0]
    #shutil.copy(fit_path,Canonical_Mesh_folder)

    if args.ColorTexture_from_raw_flg   == "True":
        ColorTexture_from_raw_flg = True
    elif args.ColorTexture_from_raw_flg == "False":
        ColorTexture_from_raw_flg = False 
    else:
        print("ColorTexture_from_raw_flg error")

    if args.use_color_flg   == "True":
        use_color_flg = True
    elif args.use_color_flg == "False":
        use_color_flg = False
    else:
        print("use_color_flg error")

    if args.Orthonormalize   == "False":
        Orthonormalize = False
    elif args.Orthonormalize == "True":
        Orthonormalize = True

    if args.use_beta   == "False":
        use_beta = False
    elif args.use_beta == "True":
        use_beta = True

    if args.Debug   == "False":
        Debug = False
    elif args.Debug == "True":
        Debug = True

    if ColorTexture_from_raw_flg and use_color_flg:
        color_raw_texture_headname  = os.path.join(args.color_raw_texture_path,r"tex.")
        uv_raw_mapping_headname     = os.path.join(args.root_path,r"FittedMesh_pose\uv_raw_mapping_")
    else:
        color_raw_texture_headname = None
        uv_raw_mapping_headname = None
    #PosedSurface_pathlist  = natsorted(glob.glob(os.path.join(args.root_path,r"FittedMesh_ply_pose\SkinnedSurface_*.ply")))
    #smplparam_pathlist          = natsorted(glob.glob(os.path.join(args.root_path,r"smplparams_centered\pose_*.bin")))

    if dataset_type == "cape":
        save_dir_root = os.path.join(args.root_path , "Texture_2path_" + args.Training_type + "_cape")
    elif dataset_type == "pop":
        save_dir_root = os.path.join(args.root_path , "Texture_2path_" + args.Training_type + "_pop")
    else:
        save_dir_root = os.path.join(args.root_path , "Texture_2path_" + args.Training_type)
    
    
    if os.path.exists(save_dir_root) == False:
        os.mkdir(save_dir_root)

    if args.save_average_texture == "False":
        if mode == 0:
            Canonical_Mesh_folder = os.path.join(args.root_path,"Canonical_Mesh_folder")
        elif mode == 1 or mode == 2:
            Canonical_Mesh_folder = os.path.join(args.root_path,"Canonical_Mesh_folder_2path")
    elif args.save_average_texture == "True":
            Canonical_Mesh_folder = save_dir_root

    Tshapecoarsejoints_path   = os.path.join(args.asset_path , "Tshapecoarsejoints.bin")

    if dataset_type != "cape":
        joint_t_path  = os.path.join(pose_folder_path,"T_joints.bin")
        shutil.copy(joint_t_path,Canonical_Mesh_folder)
    else:
        shutil.copy(Tshapecoarsejoints_path,os.path.join(Canonical_Mesh_folder , "T_joints.bin"))
        
    T_joints_path       = os.path.join(Canonical_Mesh_folder , "T_joints.bin")

    save_disp_dir  = os.path.join(save_dir_root , "texture_src")
    save_color_dir = os.path.join(save_dir_root , "texture_color")
    
    save_rgb_dir   = os.path.join(save_dir_root , "texture_rgb_src")
    
    shellTemplate_path = os.path.join(args.asset_path , "heavy" ,r"shellb_template.ply")

    #tetraSurface_path = os.path.join(args.asset_path , "shellb_template.ply")
    kintree_path       = os.path.join(args.asset_path , "kintree.bin")
    sw_shellb_surface_path    = os.path.join(args.asset_path , "heavy" , "weight_shellb_template.bin")
    sw_smpl_surface_path      = os.path.join(args.asset_path , "heavy" , "weights_template.bin")
    
    predictedNaked_posed_dir = os.path.join(save_dir_root , r"predictedNaked_posed")
    shellTemplate_posed_dir  = os.path.join(save_dir_root , r"Basis_debug")
    

    height = args.height
    width = args.width
    #alpha = args.norm_a
    data_cnt = 0      # 0  +1 : disp  , +2 : color 
    
    if args.use_disp_flg  == "True":
        data_cnt += 1            
    if args.use_color_flg == "True":
        data_cnt += 2   

    if os.path.exists(save_disp_dir)  == False:
        os.mkdir(save_disp_dir)

    if os.path.exists(save_color_dir) == False:
        os.mkdir(save_color_dir)

    if os.path.exists(predictedNaked_posed_dir) == False:
        os.mkdir(predictedNaked_posed_dir)

    if os.path.exists(shellTemplate_posed_dir)  == False:
        os.mkdir(shellTemplate_posed_dir)

    IDs = None
    if args.useID == "True":
        if args.IDstart != -999 and args.IDend != -999 and args.IDstep != -999:
            IDs = [i for i in range(args.IDstart , args.IDend, args.IDstep)]
        elif args.trainID_folder_path != None:
            IDs = load_testID(args.trainID_folder_path)

        for id in IDs:
            print(FittedMesh_pathlist[id])

    ###----------------Load UV coordinate and make faceIDTexture & BaryTexture--------------------###
    # faceIDTexture : Each pixel has id about "Which face surrounds that pixel" , BaryTexture :  Each pixel has interpolate weight from vtx1 ,vtx2 , vtx3 
    _ , _ , uv_txr , face , uv_faceID2uvIDset , _ , vtx_num , face_num = load_obj(os.path.join(args.asset_path , "heavy" , r"Shellb_expandUV_768.obj"))

    uv_txr = np.array(uv_txr)

    BaryTexture_path     = os.path.join(args.asset_path , "heavy/BaryTexture.npz")
    #save
    """
    print("start GetFaceIDandbaryTexture")
    FaceIDTexture , BaryTexture = make_displacementTexture_utils.GetFaceIDandbaryTexture(uv_txr,uv_faceID2uvIDset,face_num,height,width)  
    np.savez(BaryTexture_path , FaceIDTexture=FaceIDTexture , BaryTexture=BaryTexture)
    print("finish GetFaceIDandbaryTexture")
    """
    #load
    BaryTexture_npz = np.load(BaryTexture_path)
    FaceIDTexture = BaryTexture_npz["FaceIDTexture"]
    BaryTexture   = BaryTexture_npz["BaryTexture"]
    
    ###----------------Load Datas for LBS----------------------------------------------------------###
    kintree         = skinning_utils.load_kintree(kintree_path)
    Tshapecoarsejoints = skinning_utils.Load_joint(Tshapecoarsejoints_path)                        #smpl t_pose (position)
    joints_smpl     = skinning_utils.LoadStarSkeleton(Tshapecoarsejoints,kintree,0.5)  
    T_joints        = skinning_utils.Load_joint(T_joints_path)                       #data depend t_pose (position)     
    #joint_T         = skinning_utils.LoadStarSkeleton(T_joints,kintree,0.5)    
    joint_T         = skinning_utils.LoadStarSkeleton(T_joints,kintree,0.0)    
    skinWeights_shellb   = skinning_utils.load_skinweight_bin(sw_shellb_surface_path , vtx_num)         #93894 is hard coding
    skinWeights_smpl     = skinning_utils.load_skinweight_bin(sw_smpl_surface_path   , 6890)            #6890 is hard coding

    if mode == 1:
        smpl_model_dir_path     = os.path.join(args.asset_path , "heavy")
        smplpytorch_processor   = smplpytorch_for_nakedBody.smplpytorch_processor(smpl_model_dir_path , gender=gender)
        smplpytorch_processor.init_SMPL_layer()
        SMPL_and_Outershell_Bary_path       = os.path.abspath("../assets/heavy/SMPL_and_Outershell_Barycentric.bin")
        SMPL_and_Outershell_Corr_Face_path  = os.path.abspath("../assets/heavy/SMPL_and_Outershell_CorrespondFace.bin")
        SMPL_and_Outershell_Bary_cuda       = cuda.to_device(np.fromfile(SMPL_and_Outershell_Bary_path      , np.float32))   
        SMPL_and_Outershell_Corr_Face_cuda  = cuda.to_device(np.fromfile(SMPL_and_Outershell_Corr_Face_path , np.int32))   

        if use_beta == True:
            if dataset_type == "cape":
                beta          = None
            elif dataset_type == "ours" or dataset_type == "pop":
                smplparam_path = glob.glob(smplparam_path_headname + "*.binfitted_model.pkl")[0]
                beta          = np.array(pd.read_pickle(smplparam_path)["betas"])

            smpl_shape_verts   = smplpytorch_processor.predict_smpl_by_smplpytorch(np.zeros([24,3]) ,beta)
            shellTemplate_shaped_vtx  = smplpytorch_processor.Dilatation_smpl_to_Outershell_gpu_caller(smpl_shape_verts , SMPL_and_Outershell_Bary_cuda , SMPL_and_Outershell_Corr_Face_cuda)
        else:
            ###----------------Calculate translation(Joints(=smpl t_pose) -> Joints_T(=data_depend t_pose))-###
            smplpytorch_processor.rescale_smpl_model(Tshapecoarsejoints , T_joints , kintree)
            skeleton_t = np.zeros([24,4,4])
            for i in range(24):
                skeleton_t[i] = np.dot(joints_smpl[i] , skinning_utils.InverseTransfo(joint_T[i]))    

            shellTemplate_vtx  , _ , _ , _  , _  , _ = load_ply(shellTemplate_path)
            shellTemplate_shaped_vtx                                            = skinning_utils.SkinMeshLBS_gpu(shellTemplate_vtx , skinWeights_shellb , skeleton_t 
                                                                                                            , vtx_num , True)  

        shellTemplate_shaped_vtx_save_path  = os.path.join(Canonical_Mesh_folder , "shellTemplate_shaped_vtx.ply")
        save_ply(shellTemplate_shaped_vtx_save_path  , shellTemplate_shaped_vtx  , None , None , face , vtx_num , face_num)


    if mode == 0 or mode == 1:
        #----------------initialize basis-------------------------------------------------------#
        basis_list = []
        for i in tqdm.tqdm(range(vtx_num)):
            f1 = np.array([1,0,0] , dtype = "float32")
            f2 = np.array([0,1,0] , dtype = "float32")
            f3 = np.array([0,0,1] , dtype = "float32")
            fs = np.array([f1,f2,f3])
            basis_list.append(fs)
        basis_list = np.array(basis_list)
        basis_list_i = copy.deepcopy(basis_list)
    
        if Debug:
            #display_local_coordinate_system_as_color_on_ply(basis_list , shellTemplate_shaped_vtx , face , vtx_num , face_num , shellTemplate_posed_dir, "posed_Mesh_" + "init" + "_with_posed")
            display_local_coordinate_system_as_arrow_on_ply(basis_list , shellTemplate_shaped_vtx , face , vtx_num , face_num , shellTemplate_posed_dir, "init")
    texture_disp_list = []
    disp_list         = []
    dis_avg_list      = []
    color_list        = []
    uv_raw_list       = []
    texture_disp      = None
    base_id_list      = []

    for i, fittedmesh_path in enumerate(FittedMesh_pathlist[:]):
        if args.useID == "True" and (i in IDs) == False:
            continue
        #save_path = os.path.join(save_dir,"displacement_texture_" + str(i).zfill(4) + ".npy")
        base_id = os.path.basename(fittedmesh_path).split(".")[0].split("_")[-1].zfill(4)
        base_id_for_poseparam = str(int(base_id)).zfill(4)
        base_id_list.append(base_id)
        predictedNaked_posed_path = os.path.join(predictedNaked_posed_dir , "mesh_" +  base_id + ".ply")
        shellTemplate_posed_path  = os.path.join(shellTemplate_posed_dir  , "mesh_" +  base_id + ".ply")
        if dataset_type == "cape":
            smplparam_path = smplparam_path_headname +  base_id_for_poseparam + ".bin"
        elif dataset_type == "ours" or dataset_type == "pop":
            smplparam_path = smplparam_path_headname +  base_id_for_poseparam + ".binfitted_model.pkl"

        if uv_raw_mapping_headname != None:
            base_id5 = os.path.basename(fittedmesh_path).split(".")[0].split("_")[-1].zfill(5)
            uv_raw_mapping_path = uv_raw_mapping_headname + base_id5 + ".bin"
            print("loaded uv_raw_mapping_path : ",uv_raw_mapping_path)
        else:
            uv_raw_mapping_path = None

        disp_save_path = ""
        color_save_path = ""

        print("base_id                : ", base_id)
        print("loaded fittedmesh_path : ",fittedmesh_path)
        print("loaded smplparam_path       : ",smplparam_path)

        ###----------------Calculate translation(joint_T(=data depend t_pose) -> joints(=posed)-----###
        skeleton = np.zeros([24,4,4])
        if dataset_type == "cape":
            #pose_24_3    = skinning_utils.load_smpl(smplparam_path)     
            pose           = np.fromfile(smplparam_path , np.float32)
            beta          = None
        elif dataset_type == "ours" or dataset_type == "pop":
            pose          = np.array(pd.read_pickle(smplparam_path)["pose"])
            beta          = np.array(pd.read_pickle(smplparam_path)["betas"])
            trans         = np.array(pd.read_pickle(smplparam_path)["trans"])

        beta_save_path = os.path.join(Canonical_Mesh_folder,"beta.npy")
        if beta is not None:
            np.save(beta_save_path , beta)
        else:
            np.save(beta_save_path , np.zeros(1))
        
        pose_24_3     = pose.reshape(24,3)
        #joints   = skinning_utils.LoadSkeleton(pose_24_3 , T_joints , kintree , fix_hands = True , fix_foots = True , fix_wrists = False)   
        joints   = skinning_utils.LoadSkeleton(pose_24_3 , T_joints , kintree , fix_hands = False , fix_foots = False , fix_wrists = False)   
        for j in range(24):
            skeleton[j] = np.dot(joint_T[j] , skinning_utils.InverseTransfo(joints[j]))  

        if mode == 0 or mode == 1:
            #----------------LBS(smpl-Tpose → data-Tpose → pose)------------------------------#
            basis_list_i[:,0,:] = basis_list[:,0,:] + shellTemplate_shaped_vtx[:]  #Local→Global
            basis_list_i[:,1,:] = basis_list[:,1,:] + shellTemplate_shaped_vtx[:]  #Local→Global
            basis_list_i[:,2,:] = basis_list[:,2,:] + shellTemplate_shaped_vtx[:]  #Local→Global

            #shellTemplate_t_vtx     , t_basis                                = skinning_utils.SkinMeshandBasisLBS_gpu(shellTemplate_vtx   , basis_list_i , skinWeights_shellb , skeleton_t 
            #                                                                                                    , vtx_num , True)
            
            #shellTemplate_posed_vtx , posed_basis                            = skinning_utils.SkinMeshandBasisLBS_gpu(shellTemplate_t_vtx , t_basis    , skinWeights_shellb , skeleton   
            #                                                                                                    , vtx_num , True)
            
            shellTemplate_posed_vtx , posed_basis                            = skinning_utils.SkinMeshandBasisLBS_gpu(shellTemplate_shaped_vtx , basis_list_i    , skinWeights_shellb , skeleton   
                                                                                                                , vtx_num , True)
                
            posed_basis[:,0,:] = posed_basis[:,0,:] - shellTemplate_posed_vtx  #Global→Local
            posed_basis[:,1,:] = posed_basis[:,1,:] - shellTemplate_posed_vtx  #Global→Local
            posed_basis[:,2,:] = posed_basis[:,2,:] - shellTemplate_posed_vtx  #Global→Local
            
            if Debug:
                print("save shellTemplate_posed_path      : ", shellTemplate_posed_path)
                save_ply(shellTemplate_posed_path  , shellTemplate_posed_vtx  , None , None , face , vtx_num , face_num)
                #display_local_coordinate_system_as_color_on_ply(posed_basis , shellTemplate_posed_vtx , face , vtx_num , face_num , shellTemplate_posed_dir, "posed_Mesh_" + base_id + "with_posed")
                display_local_coordinate_system_as_arrow_on_ply(posed_basis , shellTemplate_posed_vtx , face , vtx_num , face_num , shellTemplate_posed_dir, base_id)

        elif mode == 2:
            #----------------initialize basis----------------------------------------------------#
            # ・・・
            #----------------LBS-----------------------------------------------------------------#
            # ・・・

            #print("save predictedNaked_posed_vtx_path : ",predictedNaked_posed_path)
            #save_ply(predictedNaked_posed_path , predictedNaked_posed_vtx , None , None , face , vtx_num , face_num)

            raise Exception("I haven't implemented it yet; I'll copy it from the x code.")
        
        #Orthonormal coordinate system by SVD
        """
        for j in range(posed_basis.shape[0]):
            U , s , V = np.linalg.svd(posed_basis[j,:,:])
            posed_basis[j,:,:] =  U@V
        """

        if Orthonormalize:
            def Cayley(A):
                I = np.identity(3)
                cay = np.linalg.inv((I+A))@(I-A)
                return cay

            #Orthonormal coordinate system by  low operation count
            for j in range(posed_basis.shape[0]):
                R_a = (posed_basis[j,:,:].T - posed_basis[j,:,:]) / (1 + np.trace(posed_basis[j,:,:]))
                posed_basis[j,:,:] = Cayley(R_a)

        #Orthonormal coordinate system by Projection
        """
        for j in range(posed_basis.shape[0]):      #Re-normalisation
            #f1
            posed_basis[j,0,:] = posed_basis[j,0,:] / norm(posed_basis[j,0,:])
            
            #f2
            p_nml = np.dot(np.dot(posed_basis[j,0,:] , posed_basis[j,1,:]) , posed_basis[j,0,:])
            p_pln = posed_basis[j,1,:] - p_nml
            posed_basis[j,1,:] = p_pln / norm(p_pln)

            #f3
            posed_basis[j,2,:] = np.cross(posed_basis[j,0,:] , posed_basis[j,1,:]) 
            posed_basis[j,2,:]  = posed_basis[j,2,:]/ norm(posed_basis[j,2,:] )
        """

        """
        for j in range(posed_basis.shape[0]):      #Re-normalisation
            #debug
            if np.abs(norm(posed_basis[j,0,:]) - 1.0) > 0.01:
                print(j,": f1 : " ,posed_basis[j,:,:] , "  norm : " ,norm(posed_basis[j,0,:]))
                sys.exit()
            elif np.abs(norm(posed_basis[j,1,:]) - 1.0) > 0.01:
                print(j,": f2 : " ,posed_basis[j,:,:] , "  norm : " ,norm(posed_basis[j,1,:]))
                sys.exit()
            elif np.abs(norm(posed_basis[j,2,:]) - 1.0) > 0.01:
                print(j,": f3 : " ,posed_basis[j,:,:] , "  norm : " ,norm(posed_basis[j,2,:]))
                sys.exit()
        """
        
        """
            posed_basis[j,0,:] = posed_basis[j,0,:] / norm(posed_basis[j,0,:])
            posed_basis[j,1,:] = posed_basis[j,1,:] / norm(posed_basis[j,1,:])
            posed_basis[j,2,:] = posed_basis[j,2,:] / norm(posed_basis[j,2,:])
        """
        #posed_basis = np.concatenate((Local_posed_basis_f1[:,np.newaxis,:],Local_posed_basis_f2[:,np.newaxis,:],Local_posed_basis_f3[:,np.newaxis,:]) , axis = 1)

        ###----------------Calculate displacement---------------------------------------------------------------###
        ft_vtx , _ , ft_rgb , _ , _ , _ = load_ply(fittedmesh_path)
        ft_vtx = np.array(ft_vtx , np.float32)
        if mode == 0:
            disp , color , disp_avg , uv_raw= make_displacementTexture_utils.Calc_displacement_on_LocalCoordinateSystem(shellTemplate_posed_vtx  , ft_vtx , ft_rgb , vtx_num , posed_basis, data_cnt=data_cnt , ColorTexture_from_raw_flg = False , uv_raw_mapping_path= uv_raw_mapping_path)
            
        if mode == 1 or mode == 2:
            #pose = skinning_utils.fix_smpl_parts(pose , fix_hands = True , fix_foots = True , fix_wrists = False)  
            if use_beta == True:
                smpl_shape_verts = smplpytorch_processor.predict_smpl_by_smplpytorch(pose ,beta)
            else:
                smpl_shape_verts = smplpytorch_processor.predict_smpl_by_smplpytorch(pose)
            if dataset_type == "ours" or dataset_type == "pop":
                smpl_shape_verts = smpl_shape_verts + torch.from_numpy(trans.astype(np.float32)).cuda()
            Naked_posed_vtx  = smplpytorch_processor.Dilatation_smpl_to_Outershell_gpu_caller(smpl_shape_verts , SMPL_and_Outershell_Bary_cuda , SMPL_and_Outershell_Corr_Face_cuda)
            disp , color , disp_avg , uv_raw= make_displacementTexture_utils.Calc_displacement_on_LocalCoordinateSystem(Naked_posed_vtx , ft_vtx , ft_rgb , vtx_num , posed_basis, data_cnt=data_cnt , ColorTexture_from_raw_flg = False , uv_raw_mapping_path= uv_raw_mapping_path)
            
            if Debug:
                Naked_posed_vtx_np = Naked_posed_vtx.copy_to_host() 
                save_ply(predictedNaked_posed_path , Naked_posed_vtx_np, None , None , face , vtx_num , face_num)
                debug_display_displacement_edge_save_path = os.path.join(predictedNaked_posed_dir ,"debug_disp_edge_" + base_id + ".ply")
                debug_color = ((disp - (- 0.1)) / (0.1 - (- 0.1))) * 255 
                debug_color = np.clip(debug_color , 0 , 255).astype(np.uint8)
                display_displacement_edge(ft_vtx , Naked_posed_vtx_np ,debug_color , debug_display_displacement_edge_save_path)
        disp_list.append(disp)
        dis_avg_list.append(disp_avg)
        color_list.append(color)
        uv_raw_list.append(uv_raw)
        
    base_id_list = natsorted(base_id_list)

    for i, (disp , color , base_id , uv_raw) in enumerate(zip(disp_list,color_list,base_id_list,uv_raw_list)):
        #base_id = str(i+1).zfill(4)
        if data_cnt == 2 or data_cnt == 3 :
            if ColorTexture_from_raw_flg == False:
                colorfaces = color[ft_face]  
            else:
                uv_raw_faces = uv_raw[ft_face]

        print("start drawing")
        if data_cnt == 1 or data_cnt == 3 :
            dispfaces = disp[face]
            dispfaces = np.array(dispfaces)

            threads_per_block = 16 , 16   #i.e) block dim
            blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim  
            #make texture
            texture_disp = np.zeros((height,width,3), dtype=np.float32)
            make_displacementTexture_utils.drawing_gpu2[blocks_per_grid,threads_per_block](texture_disp,dispfaces,FaceIDTexture , BaryTexture , height , width)
            #debug(drawing as pixel)
            """
            for f in range(face_num):
                uvIDs = uv_faceID2uvIDset[f]
                for g, u in enumerate(uvIDs):
                    uvs = uv_txr[u].astype(np.int32)
                    texture_disp[uvs[1] , uvs[0], 0] = dispfaces[f][g][0]
                    texture_disp[uvs[1] , uvs[0], 1] = dispfaces[f][g][1]
                    texture_disp[uvs[1] , uvs[0], 2] = dispfaces[f][g][2]
            """
            
            texture_disp_list.append(texture_disp)

            print("saving disp")
            disp_save_path = os.path.join(save_disp_dir,"displacement_texture_" + base_id + ".npy")
            print("disp : ",disp_save_path)
            np.save(disp_save_path,texture_disp)
            disp_save_path = os.path.join(save_disp_dir,"debug_texture_rgb_" + base_id + ".png")
            debug_color = ((texture_disp - (-0.1))/ (0.1 - (-0.1)))*255
            debug_color = np.clip(debug_color , 0 , 255).astype(np.uint8)
            cv2.imwrite(disp_save_path,debug_color)
            
            
        if data_cnt == 2 or data_cnt == 3 :
            if ColorTexture_from_raw_flg == False:
                threads_per_block = 16 , 16   #i.e) block dim
                blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim  
                #make texture
                texture_color = np.zeros((height,width,3), dtype=np.float32)
                colorfaces = np.array(colorfaces)
                make_displacementTexture_utils.drawing_gpu2[blocks_per_grid,threads_per_block](texture_color,colorfaces,FaceIDTexture,BaryTexture)
                texture_color = texture_avg_gpu.texture_avg(texture_color,data = "color")
                color_save_path = os.path.join(save_color_dir,"color_texture_" + base_id + ".npy")
                np.save(color_save_path,texture_color)
            else:
                threads_per_block = 16 , 16   #i.e) block dim
                blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim  
                #make texture
                texture_uv= np.zeros((height,width,2), dtype=np.float32)
                uv_raw_faces = np.array(uv_raw_faces)
                #print(uv_raw_faces.shape)
                #sys.exit()
                make_displacementTexture_utils.drawing_gpu3[blocks_per_grid,threads_per_block](texture_uv,uv_raw_faces,FaceIDTexture,BaryTexture)
                #load correspondence raw color texture by using uv mapping 
                base_id6 = str(int(base_id)).zfill(5)
                print(type(base_id6))
                color_raw_texture_path = color_raw_texture_headname + base_id6 + ".png"
                print("color_raw_texture_path : " , color_raw_texture_path)
                color_raw_texture = cv2.imread(color_raw_texture_path)
                print(color_raw_texture.shape)
                color_raw_texture = cv2.cvtColor(color_raw_texture, cv2.COLOR_BGR2RGB)
                texture_color = np.zeros((height,width,3), dtype=np.float32)
                raw_h = color_raw_texture.shape[0]
                raw_w = color_raw_texture.shape[1]
                print(raw_h , " , " , raw_w)

                for h in range(height):
                    for w in range(width):
                        color_raw_texture_tmp = color_raw_texture[raw_h - round(texture_uv[h,w,0] * raw_h) - 1,round(texture_uv[h,w,1] * raw_w)]
                        texture_color[h,w,0] = color_raw_texture_tmp[0]
                        texture_color[h,w,1] = color_raw_texture_tmp[1]
                        texture_color[h,w,2] = color_raw_texture_tmp[2]

                        if texture_color[h,w,0] == 128 and texture_color[h,w,1] == 128 and texture_color[h,w,2] == 128:
                            texture_color[h,w,0] = 0.0
                            texture_color[h,w,1] = 0.0
                            texture_color[h,w,2] = 0.0

                        texture_color[h,w,0] = round(texture_color[h,w,0]/255,6)
                        texture_color[h,w,1] = round(texture_color[h,w,1]/255,6)
                        texture_color[h,w,2] = round(texture_color[h,w,2]/255,6)

                texture_color = texture_avg_gpu.texture_avg(texture_color,data = "color")
                color_save_path = os.path.join(save_color_dir,"color_texture_" + base_id + ".npy")
                print(color_save_path)
                np.save(color_save_path,texture_color)

        
    ###### calculate average texture #################################################
    if args.save_average_texture == "True":
        texture_disps = np.array(texture_disp_list)
        datanum = texture_disps.shape[0]
        texture_disp_sum = np.sum(texture_disps , axis=0)
        texture_disp_cnt = np.count_nonzero(texture_disps , axis=0)
        thresh = 10
        nonzero_id       = np.nonzero(np.where(((texture_disp_cnt >= thresh) | (texture_disp_cnt == datanum)) , 1 , 0))                         # if 10 frame valid → average pixel valid
        #nonzero_id       = np.nonzero(np.where(texture_disp_cnt == texture_disps.shape[0] , 1 , 0))    #if all frame valid → valid
        
        texture_disp_avg = np.zeros_like(texture_disp_sum)
        texture_disp_avg[nonzero_id] = texture_disp_sum[nonzero_id] / texture_disp_cnt[nonzero_id]

        Debug_before_gauss_save_path = os.path.join(Canonical_Mesh_folder,"debug_before_gauss.png")
        #cv2.imwrite(Debug_before_gauss_save_path,(texture_disp_avg - np.min(texture_disp_avg)) / (np.max(texture_disp_avg) - np.min(texture_disp_avg))*255)
        debug_color = ((texture_disp_avg - (-0.1))/ (0.1 - (-0.1)))*255
        debug_color = np.clip(debug_color , 0 , 255).astype(np.uint8)
        cv2.imwrite(Debug_before_gauss_save_path,debug_color)

        texture_disp_avg = texture_avg_gpu.texture_avg(texture_disp_avg,data = "disp",iter = 30 , kernel_size= 3)  #Fill hole

        nonzero_id_mu = np.any(np.where(texture_disp_avg != 0.0 , 1 , 0),axis = 2)                  
                                                       
        """ #debug0310
        #nonzero_id_mu_save_path = os.path.join(Canonical_Mesh_folder,"base_displacement_texture_mask.npy")  
        #np.save(nonzero_id_mu_save_path,nonzero_id_mu)
        #nonzero_id_debug = np.concatenate([nonzero_id_mu[:,:,np.newaxis],nonzero_id_mu[:,:,np.newaxis],nonzero_id_mu[:,:,np.newaxis]],axis = 2).astype(np.int32)
        #nonzero_id_debug_save_path = os.path.join(Canonical_Mesh_folder,"base_displacement_texture_mask_rgb.png")
        #cv2.imwrite(nonzero_id_debug_save_path,nonzero_id_debug*255)
        #mask_img = np.where(nonzero_id_debug == 1 , 0 , 255)
        """

        #gaussian filter
        """
        sum_for_gaussian                    = cuda.to_device(np.zeros((768 , 768  ),dtype = "float32"))
        texture_clothes_for_gaussian        = cuda.to_device(np.zeros((768 , 768 , 3),dtype = "float32"))
        GaussianKernel = Make_kernel(kernel_size = 25 , sigma = 6 , device = "cuda:0")   
        texture_disp_avg = cuda.to_device(texture_disp_avg)
        texture_disp_avg = Mygaussian_gpu(texture_clothes_for_gaussian , texture_disp_avg , sum_for_gaussian , nonzero_id_mu.astype(np.float32) , GaussianKernel)
        texture_disp_avg = texture_disp_avg.copy_to_host()
        """

        #median filter
        kernel_size = 25
        valid_texture_for_median            = cuda.to_device(np.zeros((768 , 768 , kernel_size * kernel_size ),dtype = "float32"))
        texture_clothes_for_median          = cuda.to_device(np.zeros((768 , 768 , 3 , kernel_size * kernel_size),dtype = "float32"))

        Mymedian_gpu(texture_clothes_for_median , valid_texture_for_median , texture_disp_avg , nonzero_id_mu.astype(np.float32) , kernel_size )
        texture_clothes_for_median = texture_clothes_for_median.copy_to_host()
        valid_texture_for_median   = valid_texture_for_median.copy_to_host().astype(bool)

        valid_texture_for_median = np.concatenate([valid_texture_for_median[:,:,np.newaxis,:] , valid_texture_for_median[:,:,np.newaxis,:] ,valid_texture_for_median[:,:,np.newaxis,:] ] , axis = 2)
        texture_clothes_for_median = np.where(valid_texture_for_median , texture_clothes_for_median , np.nan)
        texture_disp_avg = np.nanmedian(texture_clothes_for_median , axis = 3)
        texture_disp_avg = np.nan_to_num(texture_disp_avg , nan = 0.0)


        mu_save_path = os.path.join(Canonical_Mesh_folder,"base_displacement_texture.npy")
        np.save(mu_save_path,texture_disp_avg)
        disp_debug_save_path = os.path.join(Canonical_Mesh_folder,"base_displacement_texture.png")
        debug_color = ((texture_disp_avg - (-0.1))/ (0.1 - (-0.1)))*255
        debug_color = np.clip(debug_color , 0 , 255).astype(np.uint8)
        cv2.imwrite(disp_debug_save_path,debug_color)


        #calculate and save gamma
        mu  = np.sum(texture_disps) / np.count_nonzero(texture_disps)
        nonzero_count_for_gamma = np.count_nonzero(texture_disps)
        nonzero_for_gamma = np.nonzero(texture_disps)
        gamma = np.sqrt(np.sum((texture_disps[nonzero_for_gamma] - mu) ** 2) / nonzero_count_for_gamma)
        gamma_save_path = os.path.join(Canonical_Mesh_folder,"gamma.npy")
        print(gamma)
        np.save(gamma_save_path , gamma)

    ###### normalization from here #################################################
    texture_root_path = save_dir_root
    texture_src_dir_path = os.path.join(texture_root_path,"texture_src")
    texture_src_path_list = natsorted(glob.glob(os.path.join(texture_src_dir_path , "displacement_texture_*.npy")))

    save_dir = os.path.join(texture_root_path , "texture_disp")
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    
    mu_path = os.path.join(Canonical_Mesh_folder,"base_displacement_texture.npy")

    #gamma = args.fixed_gamma
    mu_disp = np.load(mu_path)

    #nonzero_id = np.nonzero(mu_disp)

    alpha = args.norm_a

    final_disp_texture_list=[]
    #print("calculate disp by standadization")

    texture_src_list = []
    for i, texture_src_path in tqdm.tqdm(enumerate(texture_src_path_list)):
        texture_src = np.load(texture_src_path) 
        texture_src_list.append(texture_src)

    #caluculate gamma
    """
    texture_srcs = np.array(texture_src_list)
    count_nonzero = np.count_nonzero(np.any(texture_srcs != 0 , axis = 3))
    count_nonzero = count_nonzero * 3
    avg = np.sum(texture_srcs) / count_nonzero
        
    var_tmp = (texture_srcs - avg) ** 2
    #print(texture_srcs.shape)
    gamma = np.sqrt((np.sum(var_tmp) / count_nonzero))
    np.save(os.path.join(save_dir,"dd_gamma.npy"),gamma)
    """

    #debug0310
    #nonzero_id_mu_load_path = os.path.join(Canonical_Mesh_folder,"base_displacement_texture_mask.npy")
    #nonzero_id_mu = np.load(nonzero_id_mu_load_path)

    for i, texture_src_path in tqdm.tqdm(enumerate(texture_src_path_list)):
        #base_id = str(i+1).zfill(4)
        base_id = os.path.basename(texture_src_path).split(".")[0].split("_")[-1].zfill(4)
        texture_src = np.load(texture_src_path) 

        nonzero_src = np.any(np.where(texture_src != 0.0 , 1 , 0),axis = 2)
        nonzero = nonzero_src #* nonzero_id_mu  #debug0310
        nonzero_xyz = np.concatenate([nonzero[:,:,np.newaxis],nonzero[:,:,np.newaxis],nonzero[:,:,np.newaxis]],axis = 2)
        nonzero_id = np.nonzero(nonzero_xyz)
        texture_disp = calc_normalization(texture_src, mu_disp , gamma , nonzero_id , alpha= alpha ,shift = True)
        #texture_disp = texture_avg_gpu.texture_avg(texture_disp,data = "disp",iter = 30 , kernel_size= 3) 
        final_disp_texture_list.append(texture_disp)
        
        #texture_disp = cut_over0and1(texture_disp)

        print("saving disp")
        print("disp : ",disp_save_path)
        disp_save_path = os.path.join(save_dir,"displacement_texture_" + base_id + ".npy")
        np.save(disp_save_path,texture_disp)
        disp_rgb_save_path = os.path.join(save_dir,"debug_texture_rgb_" + base_id + ".png")
        debug_color = np.clip(texture_disp*255 , 0 , 255).astype(np.uint8)
        cv2.imwrite(disp_rgb_save_path,debug_color)
        disp_mask_save_path = os.path.join(save_dir,"displacement_texture_mask_" + base_id + ".npy")
        np.save(disp_mask_save_path,nonzero)
        disp_mask_rgb_save_path = os.path.join(save_dir,"displacement_texture_mask_rgb_" + base_id + ".png")
        nonzero_debug = nonzero_xyz.astype(np.int32)
        cv2.imwrite(disp_mask_rgb_save_path,nonzero_debug*255)
    
    debug_normed_disp_list = np.array(final_disp_texture_list)
    plt.hist(debug_normed_disp_list.flatten() , bins= 100) 
    hist_save_path = os.path.join(Canonical_Mesh_folder , "disp_hist.png" )
    plt.savefig(hist_save_path, format="png", dpi=300)
    #plt.show()

    ###### smpl copy ########################################################################
    texture_root_path = save_dir_root
    duplicate_smpl = args.duplicate_smpl

    if args.duplicate_smpl == "True":
        duplicate_smpl = True
    elif args.duplicate_smpl == "False":
        duplicate_smpl = False
    else:
        print("args.duplicate_smpl error")
        sys.exit()

    texture_disp_folder_path = os.path.join(texture_root_path,"texture_disp")
    texture_disp_paths = natsorted(glob.glob(os.path.join(texture_disp_folder_path,"displacement_texture_*.npy")))

    save_root_path = os.path.join(texture_root_path,"smplparams")
    if os.path.exists(save_root_path) == False:
        os.mkdir(save_root_path)

    base_id_list = []
    for texture_disp_path in texture_disp_paths:
        base_id = str(int(os.path.basename(texture_disp_path).split(".")[0].split("_")[-1])).zfill(4)
        save_folder_path = os.path.join(save_root_path,"poses_" + base_id)
        if os.path.exists(save_folder_path) == False:
            os.mkdir(save_folder_path)
        
        
        if duplicate_smpl:
            id = int(base_id)
            id = str(id).zfill(4)

            if dataset_type == "ours" or dataset_type == "pop":
                smpl_path = os.path.join(pose_folder_path,"pose_" + id + ".binfitted_model.pkl")
            elif dataset_type == "cape":
                smpl_path = os.path.join(pose_folder_path,"pose_" + id + ".bin")

            for i in range(3):
                if dataset_type == "ours" or dataset_type == "pop":
                    save_path = os.path.join(save_folder_path,"pose_" + id + "_" + str(i) + ".pkl")
                elif dataset_type == "cape":
                    save_path = os.path.join(save_folder_path,"pose_" + id + "_" + str(i) + ".bin")
                shutil.copy(smpl_path,save_path)
        else:
            for i in range(3):
                id = int(base_id) - 2 + i
                id = str(id).zfill(4)

                if dataset_type == "ours" or dataset_type == "pop":
                    smpl_path = os.path.join(pose_folder_path,"pose_" + id + ".binfitted_model.pkl")
                    save_path = os.path.join(save_folder_path,"pose_" + id + ".pkl")
                elif dataset_type == "cape":
                    smpl_path = os.path.join(pose_folder_path,"pose_" + id + ".bin")
                    save_path = os.path.join(save_folder_path,"pose_" + id + ".bin")
                
                shutil.copy(smpl_path,save_path)
                print(save_path)
                
if __name__ == "__main__":
    main()