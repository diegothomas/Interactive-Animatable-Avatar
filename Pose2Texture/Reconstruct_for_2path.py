import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy.lib.function_base import average
import tqdm
import json
from natsort import natsorted
import time
from numba import jit,cuda , njit
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import trimesh
import pandas as pd
import glob

import hydra
from omegaconf import DictConfig, OmegaConf

if os.name == 'posix':  #ubuntu
    from psbody.mesh import Mesh, MeshViewer

import copy
from lib import skinning_utils 
from lib.meshes_utils import load_ply , load_obj , save_ply , get_vtxID2uvCoordinate
                           
from scipy import interpolate
from scipy.linalg import norm

from pathlib import Path
def parentpath(path='.', f=0):
    return Path(path).resolve().parents[f]

from lib import reconstruct_utils
from lib.smplpytorch_for_nakedBody import Dilatation_smpl_to_Outershell_gpu , smplpytorch_processor

if __name__ == "__main__":
    from models import dataloader

#def parentpath(path=__file__, f=0):
#    return str('/'.join(os.path.abspath(path).split('/')[0:-1-f]))

def delete_invalid_faces_cpu(vtx , face , face_num):
    new_faces = np.array(face)
    for f in range(face_num):
        s0 = face[f][0]
        s1 = face[f][1]
        s2 = face[f][2]

        v0 = vtx[s0]
        v1 = vtx[s1]
        v2 = vtx[s2]
        
        #if v0 == [0.0 , 0.0 , 0.0] or v1 ==  [0.0 , 0.0 , 0.0] or v2 ==  [0.0 , 0.0 , 0.0]:
        if np.all(v0 == (np.zeros(3))) or np.all(v1 == (np.zeros(3))) or np.all(v2 == (np.zeros(3))):
            new_faces[f][0] = 0
            new_faces[f][1] = 0
            new_faces[f][2] = 0
    return new_faces

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


@cuda.jit
def make_homologous4(out_array , in_array):
    out_array[0] = in_array[0]
    out_array[1] = in_array[1]
    out_array[2] = in_array[2]
    out_array[3] = 1.0

@cuda.jit
def make_homologous3_4(out_array , in_array):
    for i in range(3):
        out_array[i][0] = in_array[i][0]
        out_array[i][1] = in_array[i][1]
        out_array[i][2] = in_array[i][2]
        out_array[i][3] = 1.0

def disp_decode(n,norm_t):
    #return round(float(n * (2*norm_t) - norm_t),6)
    return float(n * (2*norm_t) - norm_t)

def mix_texture_with_mu_cpu(src_texture , mask_img, ringID_lists , uv_cod_list, loop = 12 , face_flg =True , handsfeets_flg = True):        #TODO:cudaで並列化する。
    out_texture = copy.deepcopy(src_texture)
    
    #alpha blending with mu texture
    base = np.array([0.500001,0.500001,0.500001])      #mu (if 0.5 , removed when after process , so use 0.500001) 
    for r in range(loop):
        ratio = (r / loop) ** 2
        ringID_list = ringID_lists[str(r)]    
        for ringID in ringID_list:
            for i in range(len(uv_cod_list[ringID])):
                uv_cod = uv_cod_list[ringID][i]
                #if that coordinate is on face or hands or feets, blend it with mu
                if handsfeets_flg == True and (mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 60  or \
                                               mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 90  or \
                                               mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 120 or \
                                               mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 150 )  \
                or face_flg       == True and (mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 180 ) :
                    out_texture[int(uv_cod[1]) , int(uv_cod[0])] = src_texture[int(uv_cod[1]) , int(uv_cod[0])] * (1 - ratio) + base * ratio             
               
    #replace left face and feets with mu
    left_loop = len(ringID_lists)
    for r in range(loop ,left_loop):
        ringID_list = ringID_lists[str(r)]
        for ringID in ringID_list:
            for i in range(len(uv_cod_list[ringID])):
                uv_cod = uv_cod_list[ringID][i]
                if handsfeets_flg == True and (mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 60  or \
                                               mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 90  or \
                                               mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 120 or \
                                               mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 150 )  \
                or face_flg       == True and (mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 180 ) :
                    out_texture[int(uv_cod[1]) , int(uv_cod[0])] = base
               
    return out_texture

@cuda.jit(device=True)
def replace_hands_and_feets_cuda_sub(replaced_out_vtx , handfeet_dist, parts_number , org_vtx , uv_cod_list , mask_img , predictedNaked_posed_vtx , posed_basis_inv , mu = None ):
    """ 
    "mu" is not None : → Use mu hands and foots to replace
    "mu" is     None : → Use SMPL hands to replace
    """
    handsfeets_flg = False
    Righthand_flg  = False
    Lefthand_flg   = False
    Rightfeet_flg  = False
    Leftfeet_flg   = False
    Face_flg       = False
    connector_flg  = False
    
    for j in range(len(uv_cod_list)):
        if mask_img[uv_cod_list[j][1] , uv_cod_list[j][0]][0] == 60:      #Righthand
            handsfeets_flg = True
            Righthand_flg = True
            parts_number[0] = 1
        elif mask_img[uv_cod_list[j][1] , uv_cod_list[j][0]][0] == 90:    #Lefthand
            handsfeets_flg = True
            Lefthand_flg = True
            parts_number[0] = 2
        elif  mask_img[uv_cod_list[j][1] , uv_cod_list[j][0]][0] == 120:  #Rightfeet
            handsfeets_flg = True
            Rightfeet_flg = True
            parts_number[0] = 3
        elif mask_img[uv_cod_list[j][1] , uv_cod_list[j][0]][0] == 150:   #Leftfeet
            handsfeets_flg = True
            Leftfeet_flg = True
            parts_number[0] = 4
        elif mask_img[uv_cod_list[j][1] , uv_cod_list[j][0]][0] == 30:     #Other parts → this mean that this point is shared by other parts        
            connector_flg = True
        """
        elif mask_img[uv_cod_list[j][1] , uv_cod_list[j][0]][0] == 180:   #Face
            handsfeets_flg = True
            Face_flg = True
            parts_number[0] = 5
        """ 
    if handsfeets_flg and org_vtx[0] != 0 and org_vtx[1] != 0 and org_vtx[2] != 0:
        out_vtx_tmp  = cuda.local.array((3), np.float32)
        replace_vtx = cuda.local.array((3), np.float32)
        #posed_basis_inv = cuda.local.array((3,3), np.float32)

        #getInverse3_3(posed_basis_inv , posed_basis)
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        if mu is not None:
            skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , mu)    #Local→Global
            skinning_utils.my_add3(replace_vtx , out_vtx_tmp , predictedNaked_posed_vtx)
        else:
            replace_vtx = predictedNaked_posed_vtx

        #calculate disance between before replacing and after replacing → used by translating replaces parts
        if   Righthand_flg and connector_flg:
            for j in range(3):
                handfeet_dist[0][j] = replace_vtx[j] - org_vtx[j]
        elif Lefthand_flg and connector_flg:
            for j in range(3):
                handfeet_dist[1][j] = replace_vtx[j] - org_vtx[j]
        elif Rightfeet_flg and connector_flg:
            for j in range(3):
                handfeet_dist[2][j] = replace_vtx[j] - org_vtx[j]
        elif Leftfeet_flg and connector_flg:
            for j in range(3):
                handfeet_dist[3][j] = replace_vtx[j] - org_vtx[j]
        """
        elif Face_flg and connector_flg:
            for j in range(3):
                handfeet_dist[4][j] = replace_vtx[j] - org_vtx[j]
        """

        #replace with mu (but no translation)
        for j in range(3):
            replaced_out_vtx[j] = replace_vtx[j]
    else:
        for j in range(3):
            replaced_out_vtx[j] = org_vtx[j]

def displacement_show(texture,save_path):
    if type(texture) == torch.Tensor:
        texture = texture.to('cpu').detach().numpy().copy()
    debug_color = texture*255
    debug_color = np.clip(debug_color , 0 , 255).astype(np.uint8)
    cv2.imwrite(save_path,debug_color)

def textureTovtx(texture ,uv_cod_list, grid_x_list , grid_y_list , vtx_num, height ,width , valid_uv_list = []):
    if valid_uv_list == []:
        init_valid = True
    else:
        init_valid = False
    vtx_values = []
    test = time.time()
    valid_uv_num_list = []
    for i in range(vtx_num):
        grid_x = grid_x_list[i]
        grid_y = grid_y_list[i]
        uv_cod = uv_cod_list[i]
        #grid_uv = np.array([texture[grid_y[0],grid_x[0]],texture[grid_y[1],grid_x[1]],texture[grid_y[2],grid_x[2]],texture[grid_y[3],grid_x[3]]])
        grid_uv = np.array([texture[grid_y[0],grid_x[0]],texture[grid_y[1],grid_x[0]],texture[grid_y[0],grid_x[1]],texture[grid_y[1],grid_x[1]]])
        
        if init_valid:
            valid_uv = np.any(grid_uv,1)
            #valid_uv = np.nonzero(np.any(grid_uv,1))
            valid_uv_list.append(valid_uv.tolist())
            #valid_uv_num = len(valid_uv) 
            
        else:
            valid_uv = valid_uv_list[i]
        valid_uv_num = np.count_nonzero(valid_uv) 
        valid_uv_num_list.append(valid_uv_num)

        vtx_value = np.zeros(3)

        if valid_uv_num == 4 :
            itp_func_r = interpolate.interp2d(grid_x, grid_y, grid_uv[:,0] ,kind='linear')
            itp_func_g = interpolate.interp2d(grid_x, grid_y, grid_uv[:,1] ,kind='linear')
            itp_func_b = interpolate.interp2d(grid_x, grid_y, grid_uv[:,2] ,kind='linear')
            vtx_value[0] = itp_func_r(uv_cod[0]*width,uv_cod[1]*height)
            vtx_value[1] = itp_func_g(uv_cod[0]*width,uv_cod[1]*height)
            vtx_value[2] = itp_func_b(uv_cod[0]*width,uv_cod[1]*height)
        elif valid_uv_num > 0 :
            grid_uv = grid_uv[valid_uv]
            for k in range(3):
                vtx_value[k] = average(grid_uv[:,k])
        """
        else:
            print("strange")
            print("grid_x : ",grid_x)
            print("grid_y : ",grid_y)
            print(grid_uv)
            sys.exit()
        """
        vtx_values.append(vtx_value)
    print("test time : " ,time.time() -  test)
    if init_valid:
        return vtx_values , valid_uv_list , valid_uv_num_list
    else:
        return vtx_values

@cuda.jit #('void(float32[:,:] , float32[:,:,:] , int16[:,:,:] , int32)')
def textureTovtx_from_alignment_gpu(vtx_values , texture, uv_cod_list , vtx_num):
    i = cuda.grid(1)
    if i < vtx_num:
        vtx_value =  cuda.local.array((3), np.float32)
        cnt = 0
        for j in range(len(uv_cod_list[i])):
            uv_cod = uv_cod_list[i][j]
            if (uv_cod[0] != 0 or uv_cod[1] != 0) and ((texture[uv_cod[1] , uv_cod[0]][0] != 0.0) or (texture[uv_cod[1] , uv_cod[0]][1] != 0.0) or (texture[uv_cod[1] , uv_cod[0]][2] != 0.0)):                           
                vtx_value[0] += texture[uv_cod[1] , uv_cod[0]][0]
                vtx_value[1] += texture[uv_cod[1] , uv_cod[0]][1]
                vtx_value[2] += texture[uv_cod[1] , uv_cod[0]][2]
                cnt += 1
        if cnt == 1:
            vtx_values[i][0] = vtx_value[0] 
            vtx_values[i][1] = vtx_value[1] 
            vtx_values[i][2] = vtx_value[2] 
        elif cnt > 1:
            vtx_values[i][0] = vtx_value[0] / cnt
            vtx_values[i][1] = vtx_value[1] / cnt
            vtx_values[i][2] = vtx_value[2] / cnt
        else:
            vtx_values[i][0] = 0.0 
            vtx_values[i][1] = 0.0 
            vtx_values[i][2] = 0.0         

def textureTovtx_from_alignment_gpu_caller(texture ,uv_cod_list ,vtx_num ,device):
    threads_per_block                   = 64  #i.e) block dim
    blocks_per_grid                     = int(divUp(vtx_num,threads_per_block))     #i.e) grid dim  
    
    vtx_values                     = torch.zeros((vtx_num , 3)).to(device)
    vtx_values_cuda                = cuda.as_cuda_array(vtx_values)       
    texture_cuda                   = cuda.to_device(texture)
    textureTovtx_from_alignment_gpu[blocks_per_grid,threads_per_block](vtx_values_cuda , texture_cuda , uv_cod_list , vtx_num)
    return vtx_values_cuda

@cuda.jit(device=True)
def textureTovtx_cuda_sub(out_vtx , texture ,uv_cod_list, grid_x_list , grid_y_list , height ,width , valid_uv_num):
    grid_x = grid_x_list
    grid_y = grid_y_list
    uv_cod = uv_cod_list
    grid_uv_0 = cuda.local.array((3), np.float32)
    grid_uv_0 = texture[grid_y[0]][grid_x[0]]
    grid_uv_1 = cuda.local.array((3), np.float32)
    grid_uv_1 = texture[grid_y[1]][grid_x[0]]
    grid_uv_2 = cuda.local.array((3), np.float32)
    grid_uv_2 = texture[grid_y[0]][grid_x[1]]
    grid_uv_3 = cuda.local.array((3), np.float32)
    grid_uv_3 = texture[grid_y[1]][grid_x[1]]

    grid_uv_x = cuda.local.array((4), np.float32)
    grid_uv_x[0] = grid_uv_0[0]
    grid_uv_x[1] = grid_uv_1[0]
    grid_uv_x[2] = grid_uv_2[0]
    grid_uv_x[3] = grid_uv_3[0]

    grid_uv_y = cuda.local.array((4), np.float32)
    grid_uv_y[0] = grid_uv_0[1]
    grid_uv_y[1] = grid_uv_1[1]
    grid_uv_y[2] = grid_uv_2[1]
    grid_uv_y[3] = grid_uv_3[1]

    grid_uv_z = cuda.local.array((4), np.float32)
    grid_uv_z[0] = grid_uv_0[2]
    grid_uv_z[1] = grid_uv_1[2]
    grid_uv_z[2] = grid_uv_2[2]
    grid_uv_z[3] = grid_uv_3[2]
    

    if valid_uv_num == 4 :
        vtx_0 = my_bilinear_interplation(grid_x , grid_y , grid_uv_x, uv_cod[0]*width, uv_cod[1]*height)
        vtx_1 = my_bilinear_interplation(grid_x , grid_y , grid_uv_y, uv_cod[0]*width, uv_cod[1]*height)
        vtx_2 = my_bilinear_interplation(grid_x , grid_y , grid_uv_z, uv_cod[0]*width, uv_cod[1]*height)
        out_vtx[0] = vtx_0
        out_vtx[1] = vtx_1
        out_vtx[2] = vtx_2
    else:
        #print("valid_uv_num : ")
        #print(valid_uv_num)
        return

@cuda.jit(device=True)
def textureTovtx_alignment_cuda_sub(out_vtx , texture ,uv_cod_list):
    tmp_vtx =  cuda.local.array((3), np.float32)
    cnt = 0
    for i in range(len(uv_cod_list)):
        uv_cod = uv_cod_list[i]

        if (uv_cod[0] != 0 or uv_cod[1] != 0) and ((texture[uv_cod[1] , uv_cod[0]][0] != 0.5) or (texture[uv_cod[1] , uv_cod[0]][1] != 0.5) or (texture[uv_cod[1] , uv_cod[0]][2] != 0.5)):
            tmp_vtx[0] += texture[uv_cod[1] , uv_cod[0]][0]
            tmp_vtx[1] += texture[uv_cod[1] , uv_cod[0]][1]
            tmp_vtx[2] += texture[uv_cod[1] , uv_cod[0]][2]
            cnt += 1
            
    if cnt == 1:
        out_vtx[0] = tmp_vtx[0] 
        out_vtx[1] = tmp_vtx[1] 
        out_vtx[2] = tmp_vtx[2] 
    elif cnt >= 2:                  #boudary
        out_vtx[0] = tmp_vtx[0] / cnt
        out_vtx[1] = tmp_vtx[1] / cnt
        out_vtx[2] = tmp_vtx[2] / cnt
    else:
        out_vtx[0] = 0.5
        out_vtx[1] = 0.5
        out_vtx[2] = 0.5

@cuda.jit(device=True)
def textureTovtx_alignment_withmask_cuda_sub(out_vtx , texture ,uv_cod_list ,mask ):
    tmp_vtx =  cuda.local.array((3), np.float32)
    cnt = 0
    for i in range(len(uv_cod_list)):
        uv_cod = uv_cod_list[i]

        #if (uv_cod[0] != 0 or uv_cod[1] != 0) and ((texture[uv_cod[1] , uv_cod[0]][0] != 0.5) or (texture[uv_cod[1] , uv_cod[0]][1] != 0.5) or (texture[uv_cod[1] , uv_cod[0]][2] != 0.5) and (mask[uv_cod[1] , uv_cod[0]] == 1.0)):
        if mask[uv_cod[1] , uv_cod[0]] == 1.0:
            tmp_vtx[0] += texture[uv_cod[1] , uv_cod[0]][0]
            tmp_vtx[1] += texture[uv_cod[1] , uv_cod[0]][1]
            tmp_vtx[2] += texture[uv_cod[1] , uv_cod[0]][2]
            cnt += 1
            
    if cnt == 1:
        out_vtx[0] = tmp_vtx[0] 
        out_vtx[1] = tmp_vtx[1] 
        out_vtx[2] = tmp_vtx[2] 
    elif cnt >= 2:                  #boudary
        out_vtx[0] = tmp_vtx[0] / cnt
        out_vtx[1] = tmp_vtx[1] / cnt
        out_vtx[2] = tmp_vtx[2] / cnt
    else:
        out_vtx[0] = 0.5
        out_vtx[1] = 0.5
        out_vtx[2] = 0.5

@cuda.jit(device=True)
def textureTovtx_alignment_bool_cuda_sub(texture ,uv_cod_list):
    for i in range(len(uv_cod_list)):
        uv_cod = uv_cod_list[i]

        if texture[uv_cod[1] , uv_cod[0]] == 1.0:
            return 1
    
    return 0

def get_disp_cpu_new(ps_vtx , uv_cod_list , grid_x_list , grid_y_list , vtx_num  , texture_disp , basis ,mu_list , gamma_list ,  valid_uv_list , height , width , norm_a):
    disp_lists = textureTovtx(texture_disp , uv_cod_list , grid_x_list , grid_y_list ,vtx_num ,  height , width , valid_uv_list)
    out_vtx = copy.deepcopy(ps_vtx)
    for i, v in enumerate(ps_vtx):
        disp = disp_lists[i]
        mu = mu_list[i]
        gamma = gamma_list[i]
        disp_x = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        disp_y = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        disp_z = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2] 
        displace = np.array([disp_x,disp_y,disp_z])
        fs = basis[i]
        out_vtx[i] = np.dot(fs.T , displace) + v
        #out_vtx[i] = np.dot(np.linalg.inv(fs) , displace) + v
        #out_vtx[i] = displace + v

    return out_vtx

def get_disp_gpu(ps_vtx , uv_txr , vtx_num , vtx2UV , texture_disp , basis ,mu_list , gamma_list ,  valid_uv_list , height , width , norm_a):
    threads_per_block = 64  #i.e) block dim
    blocks_per_grid = int(divUp(vtx_num,threads_per_block))     #i.e) grid dim  
    out_vtx = np.zeros((vtx_num , 3) , dtype= "float32")
    
    get_disp_cuda[blocks_per_grid,threads_per_block](out_vtx , ps_vtx , uv_txr , vtx_num , vtx2UV , texture_disp , basis ,mu_list , gamma_list ,  valid_uv_list , height , width , norm_a )
    return out_vtx

def get_color_cpu(vtx,txr,mapVtx2UV,texture_float):
    height = texture_float.shape[0]
    width = texture_float.shape[1]
    color = []
    for i, v in enumerate(vtx):
        uv_cod = txr[mapVtx2UV[i]]
        clr = texture_float[round(uv_cod[1]*height),round(uv_cod[0]*width)]
        clr_x = round(clr[0]*255)
        clr_y = round(clr[1]*255)
        clr_z = round(clr[2]*255)
        color.append([clr_x,clr_y,clr_z])
    return color

def get_sw_cpu(vtx,txr,mapVtx2UV,texture_float):
    height = texture_float.shape[0]
    width = texture_float.shape[1]
    skinweight = []
    for i, v in enumerate(vtx):
        uv_cod = txr[mapVtx2UV[i]]
        sw = texture_float[round(uv_cod[1]*height),round(uv_cod[0]*width)]
        tmp_sw_list = []
        for j in range(24):
            skinweight.append(sw[j])
        #skinweight.append(tmp_sw_list)
        #skinweight = np.array(skinweight)
    return skinweight

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

@cuda.jit(device=True)
def my_bilinear_interplation( xs, ys, c, x, y):
    x_len = xs[1] - xs[0]
    y_len = ys[1] - ys[0]
    dx = (x - xs[0]) / x_len
    dy = (y - ys[0]) / y_len
    """
    out =  (1-dx) * (1-dy) * c[0] \
        +      dx * (1-dy) * c[1] \
        +  (1-dx) *     dy * c[2] \
        +      dx *     dy * c[3]
    """
    
    out =  (1-dx) * (1-dy) * c[0] \
        +  (1-dx) *     dy * c[1] \
        +      dx * (1-dy) * c[2] \
        +      dx *     dy * c[3]

    return out

@cuda.jit(device=True)
def getInverse3_3(res, m):
    det_a = m[0][0]*m[1][1]*m[2][2] \
          + m[0][1]*m[1][2]*m[2][0] \
          + m[0][2]*m[1][0]*m[2][1] \
          - m[0][2]*m[1][1]*m[2][0] \
          - m[0][1]*m[1][0]*m[2][2] \
          - m[0][0]*m[1][2]*m[2][1]

    res[0][0] =  (m[1][1]*m[2][2] - m[1][2]*m[2][1]) / det_a
    res[0][1] = -(m[0][1]*m[2][2] - m[0][2]*m[2][1]) / det_a
    res[0][2] =  (m[0][1]*m[1][2] - m[0][2]*m[1][1]) / det_a
    res[1][0] = -(m[1][0]*m[2][2] - m[1][2]*m[2][0]) / det_a
    res[1][1] =  (m[0][0]*m[2][2] - m[0][2]*m[2][0]) / det_a
    res[1][2] = -(m[0][0]*m[1][2] - m[0][2]*m[1][0]) / det_a
    res[2][0] =  (m[1][0]*m[2][1] - m[1][1]*m[2][0]) / det_a
    res[2][1] = -(m[0][0]*m[2][1] - m[0][1]*m[2][0]) / det_a
    res[2][2] =  (m[0][0]*m[1][1] - m[0][1]*m[1][0]) / det_a 

@cuda.jit(device=True)
def Cayley(cay , A , I):
        IplsA  =  cuda.local.array((3,3), np.float32)
        IplsA_inv = cuda.local.array((3,3), np.float32)
        ImnsA  =  cuda.local.array((3,3), np.float32)
        #IA_dot =   cuda.local.array((3,3), np.float32)
        skinning_utils.my_add3_3(IplsA , I , A)
        getInverse3_3(IplsA_inv , IplsA)
        skinning_utils.my_mns3_3_b(ImnsA , I , A)
        skinning_utils.my_dot3_3and3_3(cay , IplsA_inv , ImnsA)

@cuda.jit
def Reconstruct_mesh_gpu(out_vtx , replaced_out_vtx , out_shellTemplate_posed_vtx, handfeet_dist, parts_number, uv_cod_list , in_vtx , vtx_num , in_basis, skinWeights, skeleton , disp_texture , mu_list , gamma , norm_a , mask_img):
    i = cuda.grid(1)
    if i < vtx_num:
        shellTemplate_posed_vtx =  cuda.local.array((3), np.float32)
        posed_basis      =  cuda.local.array((3,3), np.float32)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis , in_vtx[i]  , in_basis[i] , skinWeights[i] , skeleton   , True)
        
        posed_basis[0,0] = posed_basis[0,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - shellTemplate_posed_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        out_shellTemplate_posed_vtx[i][0] = shellTemplate_posed_vtx[0]
        out_shellTemplate_posed_vtx[i][1] = shellTemplate_posed_vtx[1]
        out_shellTemplate_posed_vtx[i][2] = shellTemplate_posed_vtx[2]

        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        """
        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """
        
        disp =  cuda.local.array((3), np.float32)

        #textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , disp_texture , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """

        displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
        displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
        displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis)
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , shellTemplate_posed_vtx)

        
        replaced_out_vtx[i][0] = out_vtx[i][0]  #copy
        replaced_out_vtx[i][1] = out_vtx[i][1]  #copy
        replaced_out_vtx[i][2] = out_vtx[i][2]  #copy

        #replace hands and feets from here
        """
        handsfeets_flg = False
        Righthand_flg = False
        Lefthand_flg = False
        Rightfeet_flg = False
        Leftfeet_flg = False
        Face_flg = False
        connector_flg = False
        
        for j in range(len(uv_cod_list[i])):
            if mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 60:      #Righthand
                handsfeets_flg = True
                Righthand_flg = True
                parts_number[i] = 0
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 90:    #Lefthand
                handsfeets_flg = True
                Lefthand_flg = True
                parts_number[i] = 1
            elif  mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 120:  #Rightfeet
                handsfeets_flg = True
                Rightfeet_flg = True
                parts_number[i] = 2
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 150:   #Leftfeet
                handsfeets_flg = True
                Leftfeet_flg = True
                parts_number[i] = 3
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 180:   #Face
                handsfeets_flg = True
                Face_flg = True
                parts_number[i] = 4
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 30:     #Other parts → this mean that this point is shared by other parts        
                connector_flg = True
        
        if handsfeets_flg and out_vtx[i][0] != 0 and out_vtx[i][1] != 0 and out_vtx[i][2] != 0:
            out_vtx_tmp  = cuda.local.array((3), np.float32)
            replace_vtx_mu = cuda.local.array((3), np.float32)
            skinning_utils.my_dot3_3and3(out_vtx_tmp , fs.T , mu)
            skinning_utils.my_add3(replace_vtx_mu, out_vtx_tmp , posedsurface_vtx)

            #calculate disance between before replacing and after replacing → used by translating replaces parts
            if   Righthand_flg and connector_flg:
                handfeet_dist[0][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[0][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[0][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Lefthand_flg and connector_flg:
                handfeet_dist[1][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[1][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[1][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Rightfeet_flg and connector_flg:
                handfeet_dist[2][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[2][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[2][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Leftfeet_flg and connector_flg:
                handfeet_dist[3][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[3][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[3][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Face_flg and connector_flg:
                handfeet_dist[4][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[4][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[4][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
                        
            #replace with mu (but no translation)
            replaced_out_vtx[i][0] = replace_vtx_mu[0]
            replaced_out_vtx[i][1] = replace_vtx_mu[1]
            replaced_out_vtx[i][2] = replace_vtx_mu[2]
        """

@cuda.jit
def Reconstruct_CanonicalNakedmesh_from_CanonicalNakedTexture_gpu(out_vtx , uv_cod_list , uv_vtx , vtx_num , disp_texture ,min ,max):
    i = cuda.grid(1)
    if i < vtx_num:
        disp =  cuda.local.array((3), np.float32)

        #textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , disp_texture , uv_cod_list[i])
        
        out_vtx[i][0] = disp[0] * (max - min) + min + uv_vtx[i][0]
        out_vtx[i][1] = disp[1] * (max - min) + min + uv_vtx[i][1] 
        out_vtx[i][2] = disp[2] * (max - min) + min + uv_vtx[i][2] 

@cuda.jit('void(float32[:,:],float32[:,:], float32[:,:], int16[:,:,:], float32[:,:], int32, float32[:,:,:], float32[:,:], float32[:,:,:], float32[:,:,:], float32[:,:,:],float32[:,:], float32[:], int32[:], float32[:,:], float32, int8)')
def Reconstruct_mesh_2path_from_posedNaked_gpu(out_vtx , predictedNaked_posed_vtx , posed_shellTemplate_vtx , uv_cod_list , shellTemplate_vtx , vtx_num , in_basis, skinWeights, skeleton_t , skeleton , disp_texture , smpl_vtxs , bary_cod, correspond_face, mu_list , gamma , norm_a ):
    i = cuda.grid(1)
    if i < vtx_num:
        #shellTemplate_t_vtx      =  cuda.local.array((3), np.float32)
        in_basis_i               =  cuda.local.array((3,3), np.float32)
        #t_basis                  =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        posed_basis              =  cuda.local.array((3,3), np.float32)
        
        ### Interpolate_smpl_to_outershell ###
        for j in range(3):
            predictedNaked_posed_vtx[i][j]   = bary_cod[i * 3    ] * smpl_vtxs[correspond_face[i * 3    ]][j] + \
                                             + bary_cod[i * 3 + 1] * smpl_vtxs[correspond_face[i * 3 + 1]][j] + \
                                             + bary_cod[i * 3 + 2] * smpl_vtxs[correspond_face[i * 3 + 2]][j]

        in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

        in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
        
        in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global
 
        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp     = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        #skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_t_vtx     , t_basis     , shellTemplate_vtx_homo  ,  in_basis_i_homo , skinWeights[i] , skeleton_t , True)
        
        #shellTemplate_t_vtx_homo    = cuda.local.array((4), np.float32) 
        #t_basis_homo                = cuda.local.array((3,4), np.float32) 
        #make_homologous4(shellTemplate_t_vtx_homo   , shellTemplate_t_vtx)
        #make_homologous3_4(t_basis_homo             , t_basis)

        #skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis, shellTemplate_t_vtx_homo , t_basis_homo , skinWeights[i] , skeleton   , True)
        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis, shellTemplate_vtx_homo , in_basis_i_homo  , skinWeights[i] , skeleton   , True)
    
        posed_basis[0,0] = posed_basis[0,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - shellTemplate_posed_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - shellTemplate_posed_vtx[2]  #Global→Local

        """
        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """
        
        disp =  cuda.local.array((3), np.float32)

        #textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , disp_texture , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0]
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """
        
        #if (disp[0] != 0.5 or disp[1] != 0.5 or disp[2] != 0.5) and (mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0):
        if mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0:
            displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
            displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
            displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]
        else :
            displace[0] = 0.0
            displace[1] = 0.0
            displace[2] = 0.0
             
        out_vtx_tmp     = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)
        
        getInverse3_3(posed_basis_inv , posed_basis)
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3_nonzero(out_vtx[i] , out_vtx_tmp , predictedNaked_posed_vtx[i])

        for j in range(3):
            posed_shellTemplate_vtx[i][j] = shellTemplate_posed_vtx[j]

@cuda.jit('void(float32[:,:],float32[:,:], float32[:,:], int32, float32[:,:,:], float32[:,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:], int32[:], float32[:,:])')
def Reconstruct_debug_mu_gpu(out_vtx , predictedNaked_posed_vtx , shellTemplate_vtx , vtx_num , in_basis, skinWeights, skeleton_t , skeleton , smpl_vtxs , bary_cod, correspond_face, mu_list):
    i = cuda.grid(1)
    if i < vtx_num:
        shellTemplate_t_vtx      =  cuda.local.array((3), np.float32)
        in_basis_i               =  cuda.local.array((3,3), np.float32)
        t_basis                  =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        posed_basis              =  cuda.local.array((3,3), np.float32)
        
        ### Interpolate_smpl_to_outershell ###
        for j in range(3):
            predictedNaked_posed_vtx[i][j]   = bary_cod[i * 3    ] * smpl_vtxs[correspond_face[i * 3    ]][j] + \
                                             + bary_cod[i * 3 + 1] * smpl_vtxs[correspond_face[i * 3 + 1]][j] + \
                                             + bary_cod[i * 3 + 2] * smpl_vtxs[correspond_face[i * 3 + 2]][j]

        in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

        in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
        
        in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global
 
        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp     = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_t_vtx     , t_basis     , shellTemplate_vtx_homo  ,  in_basis_i_homo , skinWeights[i] , skeleton_t , True)
        shellTemplate_t_vtx_homo    = cuda.local.array((4), np.float32) 
        t_basis_homo                = cuda.local.array((3,4), np.float32) 
        make_homologous4(shellTemplate_t_vtx_homo   , shellTemplate_t_vtx)
        make_homologous3_4(t_basis_homo             , t_basis)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis , shellTemplate_t_vtx_homo , t_basis_homo    , skinWeights[i] , skeleton   , True)
    
        posed_basis[0,0] = posed_basis[0,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - shellTemplate_posed_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - shellTemplate_posed_vtx[2]  #Global→Local

        mu = mu_list[i]
        displace = cuda.local.array((3), np.float32)

        if (mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0):
            displace[0] = mu[0]
            displace[1] = mu[1]
            displace[2] = mu[2]
        else :
            displace[0] = 0.0
            displace[1] = 0.0
            displace[2] = 0.0

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis)
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3_nonzero(out_vtx[i] , out_vtx_tmp , predictedNaked_posed_vtx[i])

@cuda.jit('void(float32[:,:] ,float32[:,:], int16[:,:,:],float32[:,:], int32 ,float32[:,:,:] ,float32[:,:] ,float32[:,:,:] ,float32[:,:,:] ,float32[:,:,:] ,float32[:,:,:] ,float32[:,:] , float32 ,int8 , int8, float32 , float32)')
def Reconstruct_mesh_2path_from_CanonicalNakedTexture_gpu(out_vtx , out_predictedNaked_posed_vtx, uv_cod_list , shellTemplate_vtx, vtx_num , in_basis, skinWeights, skeleton_t , skeleton , texture_clothes ,texture_naked , mu_list , gamma , norm_a , mode , min_for_naked , max_for_naked):
    i = cuda.grid(1)
    if i < vtx_num:
        shellTemplate_t_vtx      =  cuda.local.array((3), np.float32)
        predictedNaked_t_vtx     =  cuda.local.array((3), np.float32)
        in_basis_i               =  cuda.local.array((3,3), np.float32)
        t_basis                  =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        predictedNaked_posed_vtx =  cuda.local.array((3), np.float32)
        posed_basis              =  cuda.local.array((3,3), np.float32)
        predictedNaked_vtx       =  cuda.local.array((3), np.float32)
        
        disp =  cuda.local.array((3), np.float32)

        ### reconst naked (t-pose)
        textureTovtx_alignment_cuda_sub(disp , texture_naked , uv_cod_list[i])
        
        predictedNaked_vtx[0] = disp[0] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][0]
        predictedNaked_vtx[1] = disp[1] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][1] 
        predictedNaked_vtx[2] = disp[2] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][2] 

        if mode == 1:
            in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
            in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
            in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

            in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
            in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
            in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
            
            in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
            in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
            in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global
        elif mode == 2:
            in_basis_i[0,0] = in_basis[i][0,0] + predictedNaked_vtx[0]  #Local→Global
            in_basis_i[0,1] = in_basis[i][0,1] + predictedNaked_vtx[1]  #Local→Global
            in_basis_i[0,2] = in_basis[i][0,2] + predictedNaked_vtx[2]  #Local→Global

            in_basis_i[1,0] = in_basis[i][1,0] + predictedNaked_vtx[0]  #Local→Global
            in_basis_i[1,1] = in_basis[i][1,1] + predictedNaked_vtx[1]  #Local→Global
            in_basis_i[1,2] = in_basis[i][1,2] + predictedNaked_vtx[2]  #Local→Global
            
            in_basis_i[2,0] = in_basis[i][2,0] + predictedNaked_vtx[0]  #Local→Global
            in_basis_i[2,1] = in_basis[i][2,1] + predictedNaked_vtx[1]  #Local→Global
            in_basis_i[2,2] = in_basis[i][2,2] + predictedNaked_vtx[2]  #Local→Global

        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        predictedNaked_vtx_homo   = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous4(predictedNaked_vtx_homo  , predictedNaked_vtx)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        skinning_utils.SkinMeshandBasisLBS_additionalVtx_cuda_sub(shellTemplate_t_vtx     , predictedNaked_t_vtx    , t_basis     , shellTemplate_vtx_homo  , predictedNaked_vtx_homo   , in_basis_i_homo , skinWeights[i] , skeleton_t , True)

        shellTemplate_t_vtx_homo    = cuda.local.array((4), np.float32) 
        predictedNaked_t_vtx_homo   = cuda.local.array((4), np.float32) 
        t_basis_homo                = cuda.local.array((3,4), np.float32) 
        make_homologous4(shellTemplate_t_vtx_homo   , shellTemplate_t_vtx)
        make_homologous4(predictedNaked_t_vtx_homo  , predictedNaked_t_vtx)
        make_homologous3_4(t_basis_homo             , t_basis)

        skinning_utils.SkinMeshandBasisLBS_additionalVtx_cuda_sub(shellTemplate_posed_vtx , predictedNaked_posed_vtx, posed_basis , shellTemplate_t_vtx_homo    , predictedNaked_t_vtx_homo    , t_basis_homo    , skinWeights[i] , skeleton   , True)
        
        out_predictedNaked_posed_vtx[i][0] = predictedNaked_posed_vtx[0]
        out_predictedNaked_posed_vtx[i][1] = predictedNaked_posed_vtx[1]
        out_predictedNaked_posed_vtx[i][2] = predictedNaked_posed_vtx[2]

        if mode == 1:
            posed_basis[0,0] = posed_basis[0,0] - shellTemplate_posed_vtx[0]  #Global→Local
            posed_basis[0,1] = posed_basis[0,1] - shellTemplate_posed_vtx[1]  #Global→Local
            posed_basis[0,2] = posed_basis[0,2] - shellTemplate_posed_vtx[2]  #Global→Local

            posed_basis[1,0] = posed_basis[1,0] - shellTemplate_posed_vtx[0]  #Global→Local
            posed_basis[1,1] = posed_basis[1,1] - shellTemplate_posed_vtx[1]  #Global→Local
            posed_basis[1,2] = posed_basis[1,2] - shellTemplate_posed_vtx[2]  #Global→Local
            
            posed_basis[2,0] = posed_basis[2,0] - shellTemplate_posed_vtx[0]  #Global→Local
            posed_basis[2,1] = posed_basis[2,1] - shellTemplate_posed_vtx[1]  #Global→Local
            posed_basis[2,2] = posed_basis[2,2] - shellTemplate_posed_vtx[2]  #Global→Local
        elif mode == 2:
            posed_basis[0,0] = posed_basis[0,0] - predictedNaked_posed_vtx[0]  #Global→Local
            posed_basis[0,1] = posed_basis[0,1] - predictedNaked_posed_vtx[1]  #Global→Local
            posed_basis[0,2] = posed_basis[0,2] - predictedNaked_posed_vtx[2]  #Global→Local

            posed_basis[1,0] = posed_basis[1,0] - predictedNaked_posed_vtx[0]  #Global→Local
            posed_basis[1,1] = posed_basis[1,1] - predictedNaked_posed_vtx[1]  #Global→Local
            posed_basis[1,2] = posed_basis[1,2] - predictedNaked_posed_vtx[2]  #Global→Local
            
            posed_basis[2,0] = posed_basis[2,0] - predictedNaked_posed_vtx[0]  #Global→Local
            posed_basis[2,1] = posed_basis[2,1] - predictedNaked_posed_vtx[1]  #Global→Local
            posed_basis[2,2] = posed_basis[2,2] - predictedNaked_posed_vtx[2]  #Global→Local

        """
        out_predictedNaked_posed_vtx[i][0] = predictedNaked_posed_vtx[0]
        out_predictedNaked_posed_vtx[i][1] = predictedNaked_posed_vtx[1]
        out_predictedNaked_posed_vtx[i][2] = predictedNaked_posed_vtx[2]
        """

        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        """
        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """

        #textureTovtx_cuda_sub(disp , texture_clothes , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , texture_clothes , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """

        displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
        displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
        displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis)
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , predictedNaked_posed_vtx)

@cuda.jit('void(float32[:,:] , int16[:,:,:], float32[:,:], int32 , float32[:,:] ,float32[:,:,:] ,float32[:,:,:] ,float32 , float32)')
def Reconstruct_nakedmesh_from_CanonicalNakedTexture_gpu(out_vtx ,  uv_cod_list , shellTemplate_vtx, vtx_num , skinWeights, skeleton , texture_naked , min_for_naked , max_for_naked):
    i = cuda.grid(1)
    if i < vtx_num:
        predictedNaked_posed_vtx =  cuda.local.array((3), np.float32)
        predictedNaked_vtx       =  cuda.local.array((3), np.float32)
        
        disp =  cuda.local.array((3), np.float32)

        ### reconst naked (t-pose)
        textureTovtx_alignment_cuda_sub(disp , texture_naked , uv_cod_list[i])
        
        predictedNaked_vtx[0] = disp[0] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][0]
        predictedNaked_vtx[1] = disp[1] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][1] 
        predictedNaked_vtx[2] = disp[2] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][2] 


        predictedNaked_vtx_homo   = cuda.local.array((4), np.float32) 
        make_homologous4(predictedNaked_vtx_homo  , predictedNaked_vtx)
        
        skinning_utils.SkinMeshLBS_cuda_sub(predictedNaked_posed_vtx , predictedNaked_vtx_homo , skinWeights[i] , skeleton , True)

        out_vtx[i][0] = predictedNaked_posed_vtx[0]
        out_vtx[i][1] = predictedNaked_posed_vtx[1]
        out_vtx[i][2] = predictedNaked_posed_vtx[2]

@cuda.jit('void(float32[:,:] ,float32[:,:], int16[:,:,:],float32[:,:], int32 ,float32[:,:,:] ,float32[:,:] ,float32[:,:,:] ,float32[:,:,:] ,float32[:,:,:] ,float32[:,:,:] ,float32[:,:] , float32 ,int8 , float32 , float32)')
def Reconstruct_mesh_2path_from_PosedNakedTexture_gpu(out_vtx , out_predictedNaked_posed_vtx, uv_cod_list , shellTemplate_vtx, vtx_num , in_basis, skinWeights, skeleton_t , skeleton , texture_clothes ,texture_naked , mu_list , gamma , norm_a , min_for_naked , max_for_naked):
    i = cuda.grid(1)
    if i < vtx_num:
        shellTemplate_t_vtx      =  cuda.local.array((3), np.float32)
        in_basis_i               =  cuda.local.array((3,3), np.float32)
        t_basis                  =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        posed_basis              =  cuda.local.array((3,3), np.float32)
        
        disp_clothes =  cuda.local.array((3), np.float32)
        disp_naked   =  cuda.local.array((3), np.float32)

        in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

        in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
        
        in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global
    

        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_t_vtx     , t_basis     , shellTemplate_vtx_homo  ,  in_basis_i_homo , skinWeights[i] , skeleton_t , True)

        shellTemplate_t_vtx_homo    = cuda.local.array((4), np.float32) 
        t_basis_homo                = cuda.local.array((3,4), np.float32) 
        make_homologous4(shellTemplate_t_vtx_homo   , shellTemplate_t_vtx)
        make_homologous3_4(t_basis_homo             , t_basis)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis , shellTemplate_t_vtx_homo , t_basis_homo    , skinWeights[i] , skeleton   , True)
 
        posed_basis[0,0] = posed_basis[0,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - shellTemplate_posed_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - shellTemplate_posed_vtx[2]  #Global→Local


        """
        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """

        #textureTovtx_cuda_sub(disp , texture_clothes , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp_clothes , texture_clothes , uv_cod_list[i])
        textureTovtx_alignment_cuda_sub(disp_naked   , texture_naked   , uv_cod_list[i])

        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        ### naked texture
        displace[0] = disp_naked[0] * (max_for_naked - min_for_naked) + min_for_naked 
        displace[1] = disp_naked[1] * (max_for_naked - min_for_naked) + min_for_naked
        displace[2] = disp_naked[2] * (max_for_naked - min_for_naked) + min_for_naked

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis)
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3(out_predictedNaked_posed_vtx[i] , out_vtx_tmp , shellTemplate_posed_vtx)

        ### clothes texture
        displace[0] = (disp_clothes[0] - 0.5) *  (gamma * norm_a) + mu[0]
        displace[1] = (disp_clothes[1] - 0.5) *  (gamma * norm_a) + mu[1]
        displace[2] = (disp_clothes[2] - 0.5) *  (gamma * norm_a) + mu[2]

        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , out_predictedNaked_posed_vtx[i])

@cuda.jit('void(float32[:,:] , int16[:,:,:],float32[:,:] , int32 , float32[:,:,:] ,float32[:,:] ,float32[:,:,:] ,float32[:,:,:] , float32 , float32 )')
def Reconstruct_nakedmesh_from_PosedNakedTexture_gpu(out_vtx , uv_cod_list , shellTemplate_vtx, vtx_num , in_basis, skinWeights, skeleton , texture_naked , min_for_naked , max_for_naked):
    i = cuda.grid(1)
    if i < vtx_num:
        in_basis_i               =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        posed_basis              =  cuda.local.array((3,3), np.float32)
        
        disp_naked   =  cuda.local.array((3), np.float32)

        in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

        in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
        
        in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global

        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis , shellTemplate_vtx_homo , in_basis_i_homo    , skinWeights[i] , skeleton   , True)
 
        posed_basis[0,0] = posed_basis[0,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - shellTemplate_posed_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - shellTemplate_posed_vtx[2]  #Global→Local

        """
        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """

        #textureTovtx_cuda_sub(disp , texture_clothes , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp_naked   , texture_naked   , uv_cod_list[i])

        displace = cuda.local.array((3), np.float32)

        displace[0] = disp_naked[0] * (max_for_naked - min_for_naked) + min_for_naked 
        displace[1] = disp_naked[1] * (max_for_naked - min_for_naked) + min_for_naked
        displace[2] = disp_naked[2] * (max_for_naked - min_for_naked) + min_for_naked

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis)
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , shellTemplate_posed_vtx)

@cuda.jit
def Reconstruct_mesh_2path_from_posedNaked_replace_on_mesh_with_mask_gpu(out_vtx , replaced_out_vtx , predictedNaked_posed_vtx, posed_shellTemplate_vtx , handfeet_dist, parts_number, uv_cod_list , shellTemplate_vtx , vtx_num , posed_basis, in_basis, skinWeights, skeleton_t , skeleton , disp_texture , smpl_vtxs , bary_cod, correspond_face , mu_list , gamma , norm_a , mask_img , predict_mask):
    i = cuda.grid(1)
    if i < vtx_num:
        #shellTemplate_t_vtx      =  cuda.local.array((3), np.float32)
        in_basis_i               =  cuda.local.array((3,3), np.float32)
        #t_basis                  =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        #posed_basis              =  cuda.local.array((3,3), np.float32)
        
        ### Interpolate_smpl_to_outershell ###
        for j in range(3):
            predictedNaked_posed_vtx[i][j]   = bary_cod[i * 3    ] * smpl_vtxs[correspond_face[i * 3    ]][j] + \
                                             + bary_cod[i * 3 + 1] * smpl_vtxs[correspond_face[i * 3 + 1]][j] + \
                                             + bary_cod[i * 3 + 2] * smpl_vtxs[correspond_face[i * 3 + 2]][j]

        in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

        in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
        
        in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global
 
        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp     = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        #skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_t_vtx     , t_basis     , shellTemplate_vtx_homo  ,  in_basis_i_homo , skinWeights[i] , skeleton_t , True)
        #shellTemplate_t_vtx_homo    = cuda.local.array((4), np.float32) 
        #t_basis_homo                = cuda.local.array((3,4), np.float32) 
        #make_homologous4(shellTemplate_t_vtx_homo   , shellTemplate_t_vtx)
        #make_homologous3_4(t_basis_homo             , t_basis)

        #skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis , shellTemplate_t_vtx_homo , t_basis_homo    , skinWeights[i] , skeleton   , True)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis[i] , shellTemplate_vtx_homo , in_basis_i_homo    , skinWeights[i] , skeleton   , True)
    
        posed_basis[i][0,0] = posed_basis[i][0,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[i][0,1] = posed_basis[i][0,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[i][0,2] = posed_basis[i][0,2] - shellTemplate_posed_vtx[2]  #Global→Local

        posed_basis[i][1,0] = posed_basis[i][1,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[i][1,1] = posed_basis[i][1,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[i][1,2] = posed_basis[i][1,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        posed_basis[i][2,0] = posed_basis[i][2,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[i][2,1] = posed_basis[i][2,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[i][2,2] = posed_basis[i][2,2] - shellTemplate_posed_vtx[2]  #Global→Local

        """
        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """
        
        disp =  cuda.local.array((3), np.float32)

        #textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_withmask_cuda_sub(disp , disp_texture , uv_cod_list[i] , predict_mask)
        #disp_mask = textureTovtx_alignment_bool_cuda_sub(predict_mask , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """

        #if (disp[0] != 0.5 or disp[1] != 0.5 or disp[2] != 0.5) and (mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0):
        #if ((mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0) and disp_mask != 0):
        #if (disp_mask == 1):
        if (disp[0] != 0.5 or disp[1] != 0.5 or disp[2] != 0.5):
            displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
            displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
            displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]
        else : 
            displace[0] = 0.0
            displace[1] = 0.0
            displace[2] = 0.0

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis[i])
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3_nonzero(out_vtx[i] , out_vtx_tmp , predictedNaked_posed_vtx[i])
        
        #replace hands and feets from here
        replaced_out_vtx_tmp = cuda.local.array((3), np.float32)
        handfeet_dist_tmp    = cuda.local.array((5,3), np.float32)
        parts_number_tmp = cuda.local.array((1), np.float32)
        #replace_hands_and_feets_cuda_sub(replaced_out_vtx_tmp , handfeet_dist_tmp, parts_number_tmp, out_vtx[i] , uv_cod_list[i] , mask_img , predictedNaked_posed_vtx[i] , posed_basis_inv , mu)
        replace_hands_and_feets_cuda_sub(replaced_out_vtx_tmp , handfeet_dist_tmp, parts_number_tmp, out_vtx[i] , uv_cod_list[i] , mask_img , predictedNaked_posed_vtx[i] , posed_basis_inv , None)

        for j in range(3):
            replaced_out_vtx[i][j] = replaced_out_vtx_tmp[j]  #copy

        for j in range(5):
            for k in range(3):
                handfeet_dist[i][j][k] = handfeet_dist_tmp[j][k]  #copy

        parts_number[i] = parts_number_tmp[0]

        for j in range(3):
            posed_shellTemplate_vtx[i][j] = shellTemplate_posed_vtx[j]

@cuda.jit
def Reconstruct_mesh_2path_from_posedNaked_replace_on_mesh_gpu(out_vtx , replaced_out_vtx , predictedNaked_posed_vtx, posed_shellTemplate_vtx , handfeet_dist, parts_number, uv_cod_list , shellTemplate_vtx , vtx_num , posed_basis, in_basis, skinWeights, skeleton_t , skeleton , disp_texture , smpl_vtxs , bary_cod, correspond_face , mu_list , gamma , norm_a , mask_img):
    i = cuda.grid(1)
    if i < vtx_num:
        #shellTemplate_t_vtx      =  cuda.local.array((3), np.float32)
        in_basis_i               =  cuda.local.array((3,3), np.float32)
        #t_basis                  =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        #posed_basis              =  cuda.local.array((3,3), np.float32)
        
        ### Interpolate_smpl_to_outershell ###
        for j in range(3):
            predictedNaked_posed_vtx[i][j]   = bary_cod[i * 3    ] * smpl_vtxs[correspond_face[i * 3    ]][j] + \
                                             + bary_cod[i * 3 + 1] * smpl_vtxs[correspond_face[i * 3 + 1]][j] + \
                                             + bary_cod[i * 3 + 2] * smpl_vtxs[correspond_face[i * 3 + 2]][j]

        in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

        in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
        
        in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
        in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
        in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global
 
        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp     = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        #skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_t_vtx     , t_basis     , shellTemplate_vtx_homo  ,  in_basis_i_homo , skinWeights[i] , skeleton_t , True)
        #shellTemplate_t_vtx_homo    = cuda.local.array((4), np.float32) 
        #t_basis_homo                = cuda.local.array((3,4), np.float32) 
        #make_homologous4(shellTemplate_t_vtx_homo   , shellTemplate_t_vtx)
        #make_homologous3_4(t_basis_homo             , t_basis)

        #skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis , shellTemplate_t_vtx_homo , t_basis_homo    , skinWeights[i] , skeleton   , True)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(shellTemplate_posed_vtx , posed_basis[i] , shellTemplate_vtx_homo , in_basis_i_homo    , skinWeights[i] , skeleton   , True)
    
        posed_basis[i][0,0] = posed_basis[i][0,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[i][0,1] = posed_basis[i][0,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[i][0,2] = posed_basis[i][0,2] - shellTemplate_posed_vtx[2]  #Global→Local

        posed_basis[i][1,0] = posed_basis[i][1,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[i][1,1] = posed_basis[i][1,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[i][1,2] = posed_basis[i][1,2] - shellTemplate_posed_vtx[2]  #Global→Local
        
        posed_basis[i][2,0] = posed_basis[i][2,0] - shellTemplate_posed_vtx[0]  #Global→Local
        posed_basis[i][2,1] = posed_basis[i][2,1] - shellTemplate_posed_vtx[1]  #Global→Local
        posed_basis[i][2,2] = posed_basis[i][2,2] - shellTemplate_posed_vtx[2]  #Global→Local

        """
        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """
        
        disp =  cuda.local.array((3), np.float32)

        #textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , disp_texture , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """

        #if (disp[0] != 0.5 or disp[1] != 0.5 or disp[2] != 0.5) and (mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0):
        if mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0:
            displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
            displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
            displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]
        else : 
            displace[0] = 0.0 
            displace[1] = 0.0
            displace[2] = 0.0

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis[i])
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3_nonzero(out_vtx[i] , out_vtx_tmp , predictedNaked_posed_vtx[i])
        
        #replace hands and feets from here
        replaced_out_vtx_tmp = cuda.local.array((3), np.float32)
        handfeet_dist_tmp    = cuda.local.array((5,3), np.float32)
        parts_number_tmp = cuda.local.array((1), np.float32)
        #replace_hands_and_feets_cuda_sub(replaced_out_vtx_tmp , handfeet_dist_tmp, parts_number_tmp, out_vtx[i] , uv_cod_list[i] , mask_img , predictedNaked_posed_vtx[i] , posed_basis_inv , mu)
        replace_hands_and_feets_cuda_sub(replaced_out_vtx_tmp , handfeet_dist_tmp, parts_number_tmp, out_vtx[i] , uv_cod_list[i] , mask_img , predictedNaked_posed_vtx[i] , posed_basis_inv , None)

        for j in range(3):
            replaced_out_vtx[i][j] = replaced_out_vtx_tmp[j]  #copy

        for j in range(5):
            for k in range(3):
                handfeet_dist[i][j][k] = handfeet_dist_tmp[j][k]  #copy

        parts_number[i] = parts_number_tmp[0]

        for j in range(3):
            posed_shellTemplate_vtx[i][j] = shellTemplate_posed_vtx[j]

@cuda.jit
def Reconstruct_mesh_2path_from_CanonicalNakedTexture_replace_on_mesh_gpu(out_vtx , replaced_out_vtx , out_predictedNaked_posed_vtx, handfeet_dist, parts_number, uv_cod_list , shellTemplate_vtx , vtx_num , in_basis, skinWeights, skeleton_t , skeleton , texture_clothes , texture_naked , mu_list , gamma , norm_a , mask_img ,mode , min_for_naked , max_for_naked):
    i = cuda.grid(1)
    if i < vtx_num:
        shellTemplate_t_vtx      =  cuda.local.array((3), np.float32)
        predictedNaked_t_vtx     =  cuda.local.array((3), np.float32)
        in_basis_i               =  cuda.local.array((3,3), np.float32)
        t_basis                  =  cuda.local.array((3,3), np.float32)

        shellTemplate_posed_vtx  =  cuda.local.array((3), np.float32)
        predictedNaked_posed_vtx =  cuda.local.array((3), np.float32)
        posed_basis              =  cuda.local.array((3,3), np.float32)
        predictedNaked_vtx       =  cuda.local.array((3), np.float32)
        disp =  cuda.local.array((3), np.float32)
        
        ### reconst naked (t-pose)
        textureTovtx_alignment_cuda_sub(disp , texture_naked , uv_cod_list[i])
        
        predictedNaked_vtx[0] = disp[0] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][0] 
        predictedNaked_vtx[1] = disp[1] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][1] 
        predictedNaked_vtx[2] = disp[2] * (max_for_naked - min_for_naked) + min_for_naked + shellTemplate_vtx[i][2] 

        if mode == 1:
            in_basis_i[0,0] = in_basis[i][0,0] + shellTemplate_vtx[i][0]  #Local→Global
            in_basis_i[0,1] = in_basis[i][0,1] + shellTemplate_vtx[i][1]  #Local→Global
            in_basis_i[0,2] = in_basis[i][0,2] + shellTemplate_vtx[i][2]  #Local→Global

            in_basis_i[1,0] = in_basis[i][1,0] + shellTemplate_vtx[i][0]  #Local→Global
            in_basis_i[1,1] = in_basis[i][1,1] + shellTemplate_vtx[i][1]  #Local→Global
            in_basis_i[1,2] = in_basis[i][1,2] + shellTemplate_vtx[i][2]  #Local→Global
            
            in_basis_i[2,0] = in_basis[i][2,0] + shellTemplate_vtx[i][0]  #Local→Global
            in_basis_i[2,1] = in_basis[i][2,1] + shellTemplate_vtx[i][1]  #Local→Global
            in_basis_i[2,2] = in_basis[i][2,2] + shellTemplate_vtx[i][2]  #Local→Global
            
        elif mode == 2:
            in_basis_i[0,0] = in_basis[i][0,0] + predictedNaked_vtx[0]  #Local→Global
            in_basis_i[0,1] = in_basis[i][0,1] + predictedNaked_vtx[1]  #Local→Global
            in_basis_i[0,2] = in_basis[i][0,2] + predictedNaked_vtx[2]  #Local→Global

            in_basis_i[1,0] = in_basis[i][1,0] + predictedNaked_vtx[0]  #Local→Global
            in_basis_i[1,1] = in_basis[i][1,1] + predictedNaked_vtx[1]  #Local→Global
            in_basis_i[1,2] = in_basis[i][1,2] + predictedNaked_vtx[2]  #Local→Global
            
            in_basis_i[2,0] = in_basis[i][2,0] + predictedNaked_vtx[0]  #Local→Global
            in_basis_i[2,1] = in_basis[i][2,1] + predictedNaked_vtx[1]  #Local→Global
            in_basis_i[2,2] = in_basis[i][2,2] + predictedNaked_vtx[2]  #Local→Global

        shellTemplate_vtx_homo    = cuda.local.array((4), np.float32) 
        predictedNaked_vtx_homo   = cuda.local.array((4), np.float32) 
        in_basis_i_homo           = cuda.local.array((3,4), np.float32) 
        shellTemplate_vtx_tmp = shellTemplate_vtx[i]
        make_homologous4(shellTemplate_vtx_homo   , shellTemplate_vtx_tmp)
        make_homologous4(predictedNaked_vtx_homo  , predictedNaked_vtx)
        make_homologous3_4(in_basis_i_homo        , in_basis_i)
        
        skinning_utils.SkinMeshandBasisLBS_additionalVtx_cuda_sub(shellTemplate_t_vtx     , predictedNaked_t_vtx    , t_basis     , shellTemplate_vtx_homo  , predictedNaked_vtx_homo   , in_basis_i_homo , skinWeights[i] , skeleton_t , True)

        shellTemplate_t_vtx_homo    = cuda.local.array((4), np.float32) 
        predictedNaked_t_vtx_homo   = cuda.local.array((4), np.float32) 
        t_basis_homo                = cuda.local.array((3,4), np.float32) 
        make_homologous4(shellTemplate_t_vtx_homo   , shellTemplate_t_vtx)
        make_homologous4(predictedNaked_t_vtx_homo  , predictedNaked_t_vtx)
        make_homologous3_4(t_basis_homo             , t_basis)

        skinning_utils.SkinMeshandBasisLBS_additionalVtx_cuda_sub(shellTemplate_posed_vtx , predictedNaked_posed_vtx, posed_basis , shellTemplate_t_vtx_homo    , predictedNaked_t_vtx_homo    , t_basis_homo    , skinWeights[i] , skeleton   , True)
        
        out_predictedNaked_posed_vtx[i][0] = predictedNaked_posed_vtx[0]
        out_predictedNaked_posed_vtx[i][1] = predictedNaked_posed_vtx[1]
        out_predictedNaked_posed_vtx[i][2] = predictedNaked_posed_vtx[2]

        if mode == 1:
            posed_basis[0,0] = posed_basis[0,0] - shellTemplate_posed_vtx[0]  #Global→Local
            posed_basis[0,1] = posed_basis[0,1] - shellTemplate_posed_vtx[1]  #Global→Local
            posed_basis[0,2] = posed_basis[0,2] - shellTemplate_posed_vtx[2]  #Global→Local

            posed_basis[1,0] = posed_basis[1,0] - shellTemplate_posed_vtx[0]  #Global→Local
            posed_basis[1,1] = posed_basis[1,1] - shellTemplate_posed_vtx[1]  #Global→Local
            posed_basis[1,2] = posed_basis[1,2] - shellTemplate_posed_vtx[2]  #Global→Local
            
            posed_basis[2,0] = posed_basis[2,0] - shellTemplate_posed_vtx[0]  #Global→Local
            posed_basis[2,1] = posed_basis[2,1] - shellTemplate_posed_vtx[1]  #Global→Local
            posed_basis[2,2] = posed_basis[2,2] - shellTemplate_posed_vtx[2]  #Global→Local
            
            
        elif mode == 2:
            posed_basis[0,0] = posed_basis[0,0] - predictedNaked_posed_vtx[0]  #Global→Local
            posed_basis[0,1] = posed_basis[0,1] - predictedNaked_posed_vtx[1]  #Global→Local
            posed_basis[0,2] = posed_basis[0,2] - predictedNaked_posed_vtx[2]  #Global→Local

            posed_basis[1,0] = posed_basis[1,0] - predictedNaked_posed_vtx[0]  #Global→Local
            posed_basis[1,1] = posed_basis[1,1] - predictedNaked_posed_vtx[1]  #Global→Local
            posed_basis[1,2] = posed_basis[1,2] - predictedNaked_posed_vtx[2]  #Global→Local
            
            posed_basis[2,0] = posed_basis[2,0] - predictedNaked_posed_vtx[0]  #Global→Local
            posed_basis[2,1] = posed_basis[2,1] - predictedNaked_posed_vtx[1]  #Global→Local
            posed_basis[2,2] = posed_basis[2,2] - predictedNaked_posed_vtx[2]  #Global→Local

        """
        out_predictedNaked_posed_vtx[i][0] = predictedNaked_posed_vtx[0]
        out_predictedNaked_posed_vtx[i][1] = predictedNaked_posed_vtx[1]
        out_predictedNaked_posed_vtx[i][2] = predictedNaked_posed_vtx[2]
        """

        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        """
        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        """
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """
        

        #textureTovtx_cuda_sub(disp , texture_clothes , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , texture_clothes , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """
        
        displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
        displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
        displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]

        out_vtx_tmp = cuda.local.array((3), np.float32)
        posed_basis_inv = cuda.local.array((3,3), np.float32)

        getInverse3_3(posed_basis_inv , posed_basis)
        #posed_basis_inv = posed_basis.T        #if Orthogonal matrix
        skinning_utils.my_dot3_3and3(out_vtx_tmp , posed_basis_inv , displace)    #Local→Global
        skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , predictedNaked_posed_vtx)
        
        
        #replace hands and feets from here
        replaced_out_vtx_tmp = cuda.local.array((3), np.float32)
        handfeet_dist_tmp    = cuda.local.array((5,3), np.float32)
        parts_number_tmp = cuda.local.array((1), np.float32)
        #replace_hands_and_feets_cuda_sub(replaced_out_vtx_tmp , handfeet_dist_tmp, parts_number_tmp, out_vtx[i] , uv_cod_list[i] , mask_img , predictedNaked_posed_vtx[i] , posed_basis_inv , mu)
        replace_hands_and_feets_cuda_sub(replaced_out_vtx_tmp , handfeet_dist_tmp, parts_number_tmp, out_vtx[i] , uv_cod_list[i] , mask_img , predictedNaked_posed_vtx[i] , posed_basis_inv , None)

        for j in range(3):
            replaced_out_vtx[i][j] = replaced_out_vtx_tmp[j]  #copy

        for j in range(5):
            for k in range(3):
                handfeet_dist[i][j][k] = handfeet_dist_tmp[j][k]  #copy

        parts_number[i] = parts_number_tmp[0]

@cuda.jit
def Reconstruct_mesh_orthonormal_gpu(out_vtx , replaced_out_vtx , posedsurface, handfeet_dist, parts_number, uv_cod_list , dds_vtx , vtx_num , basis, skinWeights, skeleton , disp_texture , mu_list , gamma , norm_a , mask_img):
    i = cuda.grid(1)
    if i < vtx_num:
        posedsurface_vtx =  cuda.local.array((3), np.float32)
        posed_basis      =  cuda.local.array((3,3), np.float32)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(posedsurface_vtx, posed_basis , dds_vtx[i], basis[i], skinWeights[i], skeleton, True)
        
        posed_basis[0,0] = posed_basis[0,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - posedsurface_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - posedsurface_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - posedsurface_vtx[2]  #Global→Local
        
        posedsurface[i][0] = posedsurface_vtx[0]
        posedsurface[i][1] = posedsurface_vtx[1]
        posedsurface[i][2] = posedsurface_vtx[2]

        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """
        
        disp =  cuda.local.array((3), np.float32)

        #textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , disp_texture , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """

        #if (disp[0] != 0.5 or disp[1] != 0.5 or disp[2] != 0.5) and (mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0):
        if mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0:
            displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
            displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
            displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]
    
            fs = posed_basis
            out_vtx_tmp = cuda.local.array((3), np.float32)
            skinning_utils.my_dot3_3and3(out_vtx_tmp , fs.T , displace)
            skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , posedsurface_vtx)
        else:
            displace[0] = mu[0]
            displace[1] = mu[1]
            displace[2] = mu[2]

            fs = posed_basis
            out_vtx_tmp = cuda.local.array((3), np.float32)
            skinning_utils.my_dot3_3and3(out_vtx_tmp , fs.T , displace)
            skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , posedsurface_vtx)

        replaced_out_vtx[i][0] = out_vtx[i][0]  #copy
        replaced_out_vtx[i][1] = out_vtx[i][1]  #copy
        replaced_out_vtx[i][2] = out_vtx[i][2]  #copy

        #replace hands and feets from here
        """
        handsfeets_flg = False
        Righthand_flg = False
        Lefthand_flg = False
        Rightfeet_flg = False
        Leftfeet_flg = False
        Face_flg = False
        connector_flg = False
        
        for j in range(len(uv_cod_list[i])):
            if mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 60:      #Righthand
                handsfeets_flg = True
                Righthand_flg = True
                parts_number[i] = 0
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 90:    #Lefthand
                handsfeets_flg = True
                Lefthand_flg = True
                parts_number[i] = 1
            elif  mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 120:  #Rightfeet
                handsfeets_flg = True
                Rightfeet_flg = True
                parts_number[i] = 2
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 150:   #Leftfeet
                handsfeets_flg = True
                Leftfeet_flg = True
                parts_number[i] = 3
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 180:   #Face
                handsfeets_flg = True
                Face_flg = True
                parts_number[i] = 4
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 30:     #Other parts → this mean that this point is shared by other parts        
                connector_flg = True
        
        if handsfeets_flg and out_vtx[i][0] != 0 and out_vtx[i][1] != 0 and out_vtx[i][2] != 0:
            out_vtx_tmp  = cuda.local.array((3), np.float32)
            replace_vtx_mu = cuda.local.array((3), np.float32)
            skinning_utils.my_dot3_3and3(out_vtx_tmp , fs.T , mu)
            skinning_utils.my_add3(replace_vtx_mu, out_vtx_tmp , posedsurface_vtx)

            #calculate disance between before replacing and after replacing → used by translating replaces parts
            if   Righthand_flg and connector_flg:
                handfeet_dist[0][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[0][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[0][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Lefthand_flg and connector_flg:
                handfeet_dist[1][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[1][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[1][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Rightfeet_flg and connector_flg:
                handfeet_dist[2][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[2][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[2][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Leftfeet_flg and connector_flg:
                handfeet_dist[3][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[3][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[3][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Face_flg and connector_flg:
                handfeet_dist[4][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[4][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[4][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
                        
            #replace with mu (but no translation)
            replaced_out_vtx[i][0] = replace_vtx_mu[0]
            replaced_out_vtx[i][1] = replace_vtx_mu[1]
            replaced_out_vtx[i][2] = replace_vtx_mu[2]
        """

@cuda.jit
def Reconstruct_mesh_gpu_debug(out_vtx , replaced_out_vtx , posedsurface, handfeet_dist, parts_number, uv_cod_list , dds_vtx , vtx_num , basis, skinWeights, skeleton , disp_texture , mu_list , gamma , norm_a , mask_img , debug_basises):
    i = cuda.grid(1)
    if i < vtx_num:
        posedsurface_vtx =  cuda.local.array((3), np.float32)
        posed_basis      =  cuda.local.array((3,3), np.float32)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(posedsurface_vtx, posed_basis , dds_vtx[i], basis[i], skinWeights[i], skeleton, True)
        
        posed_basis[0,0] = posed_basis[0,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - posedsurface_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - posedsurface_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - posedsurface_vtx[2]  #Global→Local
        
        posedsurface[i][0] = posedsurface_vtx[0]
        posedsurface[i][1] = posedsurface_vtx[1]
        posedsurface[i][2] = posedsurface_vtx[2]

        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)

        #Orthonormal coordinate system
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)
        
        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """
        
        disp =  cuda.local.array((3), np.float32)

        #textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height , width , valid_uv_num_list[i])
        textureTovtx_alignment_cuda_sub(disp , disp_texture , uv_cod_list[i])
        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)

        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """
        if (disp[0] != 0.5 or disp[1] != 0.5 or disp[2] != 0.5) and (mu[0] != 0.0 or mu[1] != 0.0 or mu[2] != 0.0):
            displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0]
            displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]
            displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]
    
            fs = transformed_basis
            out_vtx_tmp = cuda.local.array((3), np.float32)
            skinning_utils.my_dot3_3and3(out_vtx_tmp , fs.T , displace)
            skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , posedsurface_vtx)

            debug_basises[i][0][0] = fs[0][0]
            debug_basises[i][0][1] = fs[0][1]
            debug_basises[i][0][2] = fs[0][2]
            debug_basises[i][1][0] = fs[1][0]
            debug_basises[i][1][1] = fs[1][1]
            debug_basises[i][1][2] = fs[1][2]
            debug_basises[i][2][0] = fs[2][0]
            debug_basises[i][2][1] = fs[2][1]
            debug_basises[i][2][2] = fs[2][2]
        else:
            out_vtx[i][0] = 0
            out_vtx[i][1] = 0
            out_vtx[i][2] = 0
        
        replaced_out_vtx[i][0] = out_vtx[i][0]  #copy
        replaced_out_vtx[i][1] = out_vtx[i][1]  #copy
        replaced_out_vtx[i][2] = out_vtx[i][2]  #copy

        #replace hands and feets from here
        """
        handsfeets_flg = False
        Righthand_flg = False
        Lefthand_flg = False
        Rightfeet_flg = False
        Leftfeet_flg = False
        Face_flg = False
        connector_flg = False
        
        for j in range(len(uv_cod_list[i])):
            if mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 60:      #Righthand
                handsfeets_flg = True
                Righthand_flg = True
                parts_number[i] = 0
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 90:    #Lefthand
                handsfeets_flg = True
                Lefthand_flg = True
                parts_number[i] = 1
            elif  mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 120:  #Rightfeet
                handsfeets_flg = True
                Rightfeet_flg = True
                parts_number[i] = 2
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 150:   #Leftfeet
                handsfeets_flg = True
                Leftfeet_flg = True
                parts_number[i] = 3
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 180:   #Face
                handsfeets_flg = True
                Face_flg = True
                parts_number[i] = 4
            elif mask_img[uv_cod_list[i][j][1] , uv_cod_list[i][j][0]][0] == 30:     #Other parts → this mean that this point is shared by other parts        
                connector_flg = True
        
        if handsfeets_flg and out_vtx[i][0] != 0 and out_vtx[i][1] != 0 and out_vtx[i][2] != 0:
            out_vtx_tmp  = cuda.local.array((3), np.float32)
            replace_vtx_mu = cuda.local.array((3), np.float32)
            skinning_utils.my_dot3_3and3(out_vtx_tmp , fs.T , mu)
            skinning_utils.my_add3(replace_vtx_mu, out_vtx_tmp , posedsurface_vtx)

            #calculate disance between before replacing and after replacing → used by translating replaces parts
            if   Righthand_flg and connector_flg:
                handfeet_dist[0][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[0][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[0][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Lefthand_flg and connector_flg:
                handfeet_dist[1][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[1][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[1][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Rightfeet_flg and connector_flg:
                handfeet_dist[2][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[2][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[2][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Leftfeet_flg and connector_flg:
                handfeet_dist[3][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[3][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[3][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
            elif Face_flg and connector_flg:
                handfeet_dist[4][i][0] = replace_vtx_mu[0] - out_vtx[i][0]
                handfeet_dist[4][i][1] = replace_vtx_mu[1] - out_vtx[i][1]
                handfeet_dist[4][i][2] = replace_vtx_mu[2] - out_vtx[i][2]
                        
            #replace with mu (but no translation)
            replaced_out_vtx[i][0] = replace_vtx_mu[0]
            replaced_out_vtx[i][1] = replace_vtx_mu[1]
            replaced_out_vtx[i][2] = replace_vtx_mu[2]
        """

@cuda.jit
def Reconstruct_mesh_color_gpu(out_vtx , out_clr , uv_cod_list , grid_x_list , grid_y_list , uv_cod_clr_list , grid_x_clr_list , grid_y_clr_list ,  dds_vtx , vtx_num , basis, skinWeights, skeleton , disp_texture , color_texture , mu_list , gamma , valid_uv_list , valid_uv_num_list , height_disp , width_disp , height_clr , width_clr , norm_a ):
    i = cuda.grid(1)
    if i < vtx_num:
        posedsurface_vtx =  cuda.local.array((3), np.float32)
        posed_basis      =  cuda.local.array((3,3), np.float32)

        skinning_utils.SkinMeshandBasisLBS_cuda_sub(posedsurface_vtx, posed_basis , dds_vtx[i], basis[i], skinWeights[i], skeleton, True)
        
        posed_basis[0,0] = posed_basis[0,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[0,1] = posed_basis[0,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[0,2] = posed_basis[0,2] - posedsurface_vtx[2]  #Global→Local

        posed_basis[1,0] = posed_basis[1,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[1,1] = posed_basis[1,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[1,2] = posed_basis[1,2] - posedsurface_vtx[2]  #Global→Local
        
        posed_basis[2,0] = posed_basis[2,0] - posedsurface_vtx[0]  #Global→Local
        posed_basis[2,1] = posed_basis[2,1] - posedsurface_vtx[1]  #Global→Local
        posed_basis[2,2] = posed_basis[2,2] - posedsurface_vtx[2]  #Global→Local
        

        transformed_basis = cuda.local.array((3,3), np.float32)
        posed_basis_mns = cuda.local.array((3,3), np.float32)
        R_a = cuda.local.array((3,3), np.float32)
        #Orthonormal coordinate system
        
        I =  cuda.local.array((3,3), np.float32)
        I[0][0] = 1.0
        I[1][1] = 1.0
        I[2][2] = 1.0

        trace = posed_basis[0][0] + posed_basis[1][1] + posed_basis[2][2]
        skinning_utils.my_mns3_3_b(posed_basis_mns,posed_basis.T ,posed_basis)
        skinning_utils.my_div3_3(R_a , posed_basis_mns , (1 + trace))
        Cayley(transformed_basis , R_a , I)

        """
        U , s , V = np.linalg.svd(posed_basis)
        transformed_basis =  U@V
        """

        disp =  cuda.local.array((3), np.float32)
        
        textureTovtx_cuda_sub(disp , disp_texture , uv_cod_list[i] , grid_x_list[i] , grid_y_list[i] ,height_disp , width_disp , valid_uv_num_list[i])

        mu = mu_list[i]
        #gamma = gamma_list[i]
        displace = cuda.local.array((3), np.float32)
        
        """
        displace[0] = (disp[0] - 0.5) *  (gamma[0] * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma[1] * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma[2] * norm_a) + mu[2]
        """

        displace[0] = (disp[0] - 0.5) *  (gamma * norm_a) + mu[0] 
        displace[1] = (disp[1] - 0.5) *  (gamma * norm_a) + mu[1]  
        displace[2] = (disp[2] - 0.5) *  (gamma * norm_a) + mu[2]

        fs = transformed_basis
        out_vtx_tmp = cuda.local.array((3), np.float32)
        skinning_utils.my_dot3_3and3(out_vtx_tmp , fs.T , displace)
        skinning_utils.my_add3(out_vtx[i] , out_vtx_tmp , posedsurface_vtx)

        #color from here
        clr =  cuda.local.array((3), np.float32)
        textureTovtx_cuda_sub(clr , color_texture , uv_cod_clr_list[i] , grid_x_clr_list[i] , grid_y_clr_list[i] ,height_clr , width_clr , valid_uv_num_list[i])

        out_clr[i][0] = round(clr[0]*255)
        out_clr[i][1] = round(clr[1]*255)
        out_clr[i][2] = round(clr[2]*255)

class AITS_Reconstructor():
    def __init__(self , cfg):
        self.device                      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.asset_path                  = os.path.abspath("../assets")
        self.kintree_path                = os.path.join(self.asset_path , "kintree.bin" )

        self.SMPL_and_Outershell_Corr_Face_path    = os.path.join(self.asset_path , "heavy" , "SMPL_and_Outershell_CorrespondFace.bin")
        self.SMPL_and_Outershell_Bary_path         = os.path.join(self.asset_path , "heavy" , "SMPL_and_Outershell_Barycentric.bin")
        self.smpl_model_dir              = os.path.join(self.asset_path , "heavy")
        
        self.test_type                   = cfg.train.test_type
        self.reconst_mode                = cfg.train.reconst_mode
        self.smpl_folder_path_test       = cfg.train.smpl_folder_path_test
        self.cp_naked_inference          = cfg.train.cp_naked_inference
        self.naked_texture_type          = cfg.train.naked_texture_type
        self.naked_topology              = cfg.train.naked_topology
        self.smpl_gender                 = cfg.train.smpl_gender
        self.save_mode                   = cfg.train.save_mode
        self.Replace_on_the_             = cfg.train.Replace_on_the_

        if self.test_type != "naked":
            self.Canonical_Mesh_folder       = cfg.train.Canonical_Mesh_folder
            self.mu_texture_path             = os.path.join(self.Canonical_Mesh_folder, "base_displacement_texture.npy")
            #self.base_displacement_texture_mask_path = os.path.join(self.Canonical_Mesh_folder, "base_displacement_texture_mask.npy")
            self.gamma_path                  = os.path.join(self.Canonical_Mesh_folder, "gamma.npy")
            self.joint_t_path                = os.path.join(self.Canonical_Mesh_folder, "T_joints.bin")
            self.shellb_template_shaped_vtx  = os.path.join(self.Canonical_Mesh_folder, "shellTemplate_shaped_vtx.ply")
        self.Tshapecoarsejoints_path         = os.path.join(self.asset_path               , "Tshapecoarsejoints.bin")

        self.outershell_model_path         = os.path.join(self.asset_path , "heavy" , "shellb_template.ply")
        self.outershell_sw_surface_path    = os.path.join(self.asset_path , "heavy" , "weight_shellb_template.bin")
        self.outershell_uvskin_path        = os.path.join(self.asset_path , "heavy" , "Shellb_expandUV_768.obj")

        self.SMPL_template_model_path         = os.path.join(self.asset_path , "heavy" , "Template_skin.ply")
        self.SMPL_sw_surface_path             = os.path.join(self.asset_path , "heavy" , "weights_template.bin")
        self.SMPL_uvskin_path                 = os.path.join(self.asset_path , "heavy" , "SMPL_expandUV_512.obj")

        self.beta_path                  = os.path.join(self.Canonical_Mesh_folder , "beta.npy")
        self.use_beta                   = cfg.reconst.use_beta
        
        if self.naked_topology == "Outershell":
            self.texture_resolution = 1024
        elif self.naked_topology == "SMPL":
            self.texture_resolution = 512

        self.smpl_fmt                    = cfg.reconst.smpl_format
        #self.gamma                       = cfg.reconst.fixed_gamma
        self.Orthonormalize              = cfg.reconst.Orthonormalize
        self.norm_a                      = cfg.reconst.norm_a

        self.reconst_dir                 = os.path.abspath(cfg.train.reconst)   
        
        self.one_ring_vtxIDfacelist_dict_path       = os.path.join(self.asset_path , "heavy" , "one_ring_vtxIDfacelist_dict_" + self.naked_topology + ".json")
        self.edge_oneling_neighbor_vtxID_dicts_path = os.path.join(self.asset_path , "heavy" , "edge_oneling_neighbor_vtxID_dicts_" + self.naked_topology + ".json")
        self.present_ringID_lists_path              = os.path.join(self.asset_path , "heavy" , "ring_stairs_" + self.naked_topology + ".json")

        if os.path.isdir(self.reconst_dir) == False:
            os.mkdir(self.reconst_dir)

        if self.save_mode == "display" and os.name == 'posix':                 
            self.mv = MeshViewer(keepalive=True)

    def asset_loader(self):
        vtx_outershell  , _ , _ , self.face_outershell  , self.vtx_num_outershell  , self.face_num_outershell  = load_ply(self.outershell_model_path)
        shellb_template_shaped_vtx  , _ , _ , _  , _  , _  = load_ply(self.shellb_template_shaped_vtx)
        self.vtx_outershell             = cuda.to_device(np.array(vtx_outershell , dtype = np.float32))
        self.shellb_template_shaped_vtx = cuda.to_device(np.array(shellb_template_shaped_vtx , dtype = np.float32))



        vtx_smpl  , _ , _ , self.face_smpl  , self.vtx_num_smpl  , self.face_num_smpl                          = load_ply(self.SMPL_template_model_path)
        self.vtx_smpl = cuda.to_device(np.array(vtx_smpl , dtype = np.float32))

        self.skinWeights_SMPL       = cuda.to_device(skinning_utils.load_skinweight_bin(self.SMPL_sw_surface_path       , vtx_num = self.vtx_num_smpl))
        self.skinWeights_outershell = cuda.to_device(skinning_utils.load_skinweight_bin(self.outershell_sw_surface_path , vtx_num = self.vtx_num_outershell))

        _ , _ , uvs_tmp , faceID2vtxIDset_tmp , faceID2uvIDset_tmp , _ , vtx_num_tmp , _ = load_obj(self.SMPL_uvskin_path)
        vtxID2uvCordinate = get_vtxID2uvCoordinate(uvs_tmp , faceID2vtxIDset_tmp , faceID2uvIDset_tmp , vtx_num_tmp,  uvs_np_format = np.int16 , regular_array_size = 4)
        self.uv_cod_list_smpl = cuda.to_device(np.array(vtxID2uvCordinate))
           
        _ , _ , uvs_tmp , faceID2vtxIDset_tmp , faceID2uvIDset_tmp , _ , vtx_num_tmp , _ = load_obj(self.outershell_uvskin_path)
        vtxID2uvCordinate_outershell = get_vtxID2uvCoordinate(uvs_tmp , faceID2vtxIDset_tmp , faceID2uvIDset_tmp , vtx_num_tmp,  uvs_np_format = np.int16 , regular_array_size = 4)
        self.uv_cod_list_outershell = cuda.to_device(np.array(vtxID2uvCordinate_outershell))

        if self.naked_topology == "Outershell":
            #self.Template_model_vtx  = self.vtx_outershell
            self.Template_model_vtx   = self.shellb_template_shaped_vtx
            self.vtx_num              = self.vtx_num_outershell
            self.face_num             = self.face_num_outershell
            self.face                 = self.face_outershell
            self.uv_cod_list          = self.uv_cod_list_outershell
            self.skinWeights          =  self.skinWeights_outershell
        elif self.naked_topology == "SMPL":
            self.Template_model_vtx  = self.vtx_smpl
            self.vtx_num             = self.vtx_num_smpl
            self.face_num            = self.face_num_smpl
            self.face                = self.face_smpl
            self.uv_cod_list         = self.uv_cod_list_smpl
            self.skinWeights         =  self.skinWeights_SMPL

        if self.test_type != "naked":
            self.texture_mu = np.load(self.mu_texture_path)
            self.gamma      = float(np.load(self.gamma_path))

        with open(self.one_ring_vtxIDfacelist_dict_path, 'r') as f:     
            self.one_ring_vtxIDfacelist_dict = json.load(f)

        with open(self.edge_oneling_neighbor_vtxID_dicts_path, 'r') as f:  
            self.edge_oneling_neighbor_vtxID_dicts = json.load(f)

        with open(self.present_ringID_lists_path, 'r') as f:  
            self.present_ringID_lists = json.load(f)
        
        mask_img = cv2.imread(os.path.join(self.asset_path,"mask_" + self.naked_topology + ".png"))
        self.mask_img = cuda.to_device(mask_img)
        self.mask_img_for_gaussian = cuda.to_device(np.where(mask_img == 255 , 0 , 1)[:,:,0].astype(np.int8))

        #self.base_displacement_texture_mask = np.load(self.base_displacement_texture_mask_path)

        self.bary_cod_cuda        = cuda.to_device(np.fromfile(self.SMPL_and_Outershell_Bary_path      , np.float32))   
        self.correspond_face_cuda = cuda.to_device(np.fromfile(self.SMPL_and_Outershell_Corr_Face_path , np.int32))     

        if self.cp_naked_inference != "smplpytorch":
            minmax_save_path = os.path.join(self.asset_path , "naked_normalize" , self.naked_texture_type + "_" + self.naked_topology)
            self.min_for_naked = np.load(os.path.join(minmax_save_path , r"min.npy")).tolist()
            self.max_for_naked = np.load(os.path.join(minmax_save_path , r"max.npy")).tolist()

    def reconst_preprocess(self):
        ###----------------Load Datas for LBS----------------------------------------------------------###
        self.kintree                = skinning_utils.load_kintree(self.kintree_path)
        self.Tshapecoarsejoints     = skinning_utils.Load_joint(self.Tshapecoarsejoints_path)
        self.joints_smpl            = skinning_utils.LoadStarSkeleton(self.Tshapecoarsejoints,self.kintree,0.5)    #smpl t_pose (position)
        
        if self.test_type != "naked":  
            self.T_joints     = skinning_utils.Load_joint(self.joint_t_path)
            #self.joint_T  = skinning_utils.LoadStarSkeleton(self.T_joints,self.kintree,0.5)   #data depend t_pose (position)
            self.joint_T  = skinning_utils.LoadStarSkeleton(self.T_joints,self.kintree,0.0)   #data depend t_pose (position)
            self.mu_list = textureTovtx_from_alignment_gpu_caller(self.texture_mu, self.uv_cod_list , self.vtx_num ,self.device) 

            ###----------------Calculate translation(Joints(=smpl t_pose) -> Joints_T(=data_depend t_pose))-###
            skeleton_t = np.zeros([24,4,4] , np.float32)
            for i in range(24):
                skeleton_t[i] = np.dot(self.joints_smpl[i] , skinning_utils.InverseTransfo(self.joint_T[i]))  
            
            self.skeleton_t = cuda.to_device(skeleton_t)

        if self.reconst_mode == 0 or self.reconst_mode == 1:
            #----------------initialize basis-------------------------------------------------------#
            basis_list = []
            for i in tqdm.tqdm(range(self.vtx_num)):
                f1 = np.array([1,0,0] , dtype = "float32")
                f2 = np.array([0,1,0] , dtype = "float32")
                f3 = np.array([0,0,1] , dtype = "float32")
                fs = np.array([f1,f2,f3])
                basis_list.append(fs)
            self.basis_list = cuda.to_device(np.array(basis_list , np.float32))
            if self.naked_topology == "SMPL":
                basis_list = []
                for i in tqdm.tqdm(range(self.vtx_num_outershell)):
                    f1 = np.array([1,0,0] , dtype = "float32")
                    f2 = np.array([0,1,0] , dtype = "float32")
                    f3 = np.array([0,0,1] , dtype = "float32")
                    fs = np.array([f1,f2,f3])
                    basis_list.append(fs)
                self.basis_list_outershell = cuda.to_device(np.array(basis_list , np.float32))
        elif self.reconst_mode == 2:
            raise Exception("not implemented")

        if self.reconst_mode == 0 :     #FIXME:numbaに書き換える、
            #----------------LBS(smpl-Tpose → data-Tpose)--------------------------------------------#
            self.basis_list[:,0,:] = self.basis_list[:,0,:] + self.Template_model_vtx[:]  #Local→Global
            self.basis_list[:,1,:] = self.basis_list[:,1,:] + self.Template_model_vtx[:]  #Local→Global
            self.basis_list[:,2,:] = self.basis_list[:,2,:] + self.Template_model_vtx[:]  #Local→Global

            shellTemplate_t_vtx     , t_basis                                = skinning_utils.SkinMeshandBasisLBS_gpu(self.Template_model_vtx   , self.basis_list , self.skinWeights , self.skeleton_t 
                                                                                                                , self.self.vtx_num , True)
            shellTemplate_t_vtx = np.array(shellTemplate_t_vtx)
            self.shellTemplate_t_vtx = np.hstack((shellTemplate_t_vtx , np.ones((self.vtx_num ,1))))
            t_basis_f1 = np.hstack((t_basis[:,0] , np.ones((self.vtx_num ,1))))
            t_basis_f2 = np.hstack((t_basis[:,1] , np.ones((self.vtx_num ,1))))
            t_basis_f3 = np.hstack((t_basis[:,2] , np.ones((self.vtx_num ,1))))
            self.t_basis = np.stack([t_basis_f1,t_basis_f2,t_basis_f3],axis = 1)


        self.GaussianKernel = reconstruct_utils.Make_kernel(kernel_size = 15 , sigma = 4 , device = self.device)       #sigmaの値が大きすぎると溝ができる
        #self.GaussianKernel = reconstruct_utils.Make_kernel(kernel_size = 5 , sigma = 2 , device = self.device)       #sigmaの値が大きすぎると溝ができる
        self.texture_clothes_for_gaussian        = cuda.to_device(np.zeros((768 , 768 ,3),dtype = "float32"))           #FIXME : hard cording
        self.sum_for_gaussian                    = cuda.to_device(np.zeros((768 , 768   ),dtype = "float32"))           #FIXME : hard cording
        self.out_vtx                             = cuda.to_device(np.zeros((self.vtx_num_outershell , 3) , dtype= "float32"))
        self.predictedNaked_posed_vtx            = cuda.to_device(np.zeros((self.vtx_num , 3) , dtype= "float32"))
        self.predictedNaked_posed_outershell_vtx = cuda.to_device(np.zeros((self.vtx_num_outershell , 3) , dtype= "float32"))
        self.shellTemplate_posed_vtx             = cuda.to_device(np.zeros((self.vtx_num , 3) , dtype= "float32"))
        self.posed_basis_list                    = cuda.to_device(np.zeros((self.vtx_num , 3 , 3) , dtype= "float32"))
                
        if self.Replace_on_the_ == "mesh":
            self.replaced_out_vtx                = cuda.to_device(np.zeros((self.vtx_num , 3) , dtype= "float32"))
            self.handfeet_dist                   = cuda.to_device(np.zeros((self.vtx_num , 5 , 3) , dtype= "float32"))   #axis0 → 0 : Rhand , 1 : Lhand , 2 : Rfeet , 3 : Lfeet , 4 : Face
            self.parts_number                    = cuda.to_device(np.full((self.vtx_num), 99 , dtype= "uint8"))         #axis0 → 0 : Rhand , 1 : Lhand , 2 : Rfeet , 3 : Lfeet , 4 : Face , 99 : other
        
    def Interpolate_smpl_to_outershell_cpu(self , smpl_vtxs):                          
        bary_cod        = np.fromfile(self.SMPL_and_Outershell_Bary_path      , np.float32)
        correspond_face = np.fromfile(self.SMPL_and_Outershell_Corr_Face_path , np.int32)
        
        outershell_vtxs = np.zeros((93894 , 3), np.float32)
        for i in range(93894):
            for j in range(3):
                outershell_vtxs[i][j]   = bary_cod[i * 3    ] * smpl_vtxs[correspond_face[i * 3    ]][j] + \
                                        + bary_cod[i * 3 + 1] * smpl_vtxs[correspond_face[i * 3 + 1]][j] + \
                                        + bary_cod[i * 3 + 2] * smpl_vtxs[correspond_face[i * 3 + 2]][j]

        """
        #debug
        PC = trimesh.points.PointCloud(vertices = outershell_vtxs )
        PC.export("./outer.obj")
        """
        return outershell_vtxs
    
    def post_process(self , fileID , pose, smpl_vtxs = None, texture_naked = None , texture_clothes = None , texture_color = None , texture_mask = None):   
        #print("fileID : ",fileID)
        if texture_naked != None:
            texture_naked = cuda.as_cuda_array(texture_naked)
        if texture_clothes != None:
            if self.save_mode == "debug":
                debug_texture_clothes_path     = os.path.join(self.reconst_dir, r"raw_displacement_texture_" + str(fileID).zfill(4) +".png")
                displacement_show(texture_clothes , debug_texture_clothes_path)
            texture_clothes = cuda.as_cuda_array(texture_clothes)
        if texture_color != None:
            texture_color = cuda.as_cuda_array(texture_color)
        if texture_mask != None:
            if self.save_mode == "debug":
                debug_texture_mask = torch.cat((texture_mask , texture_mask  , texture_mask) , axis = 2)
                debug_texture_mask_path = os.path.join(self.reconst_dir, r"output_texture_mask_" + str(fileID).zfill(4) +".png")
                displacement_show(debug_texture_mask , debug_texture_mask_path)

            #texture_mask = base_displacement_texture_mask * texture_mask[:,:,0]
            texture_mask_cuda = cuda.as_cuda_array(texture_mask[:,:,0].to(torch.int8))
        else:
            texture_mask_cuda = self.mask_img_for_gaussian
        #else:
        #   texture_mask = cuda.as_cuda_array(base_displacement_texture_mask)

        """
        if self.save_mode == "debug":
            #debug_texture_mask = texture_mask.to('cpu').detach().numpy().copy()
            debug_texture_mask = texture_mask.copy_to_host()
            debug_texture_mask = np.concatenate((debug_texture_mask[:,:,np.newaxis] , debug_texture_mask[:,:,np.newaxis]  , debug_texture_mask[:,:,np.newaxis]) , axis = 2)
            debug_texture_mask_path = os.path.join(self.reconst_dir, r"applied_texture_mask_" + str(fileID).zfill(4) +".png")
            displacement_show(debug_texture_mask , debug_texture_mask_path)
        """
            
        ### Load texture & Process to Displacement texture ###
        if texture_clothes != None:
            texture_clothes = reconstruct_utils.Mygaussian_gpu(self.texture_clothes_for_gaussian , texture_clothes , self.sum_for_gaussian , texture_mask_cuda , self.GaussianKernel , self.device)
            if self.save_mode == "debug":
                debug_texture_clothes = texture_clothes.copy_to_host()

                debug_texture_clothes_path     = os.path.join(self.reconst_dir, r"gauss_displacement_texture_" + str(fileID).zfill(4) +".png")
                displacement_show(debug_texture_clothes , debug_texture_clothes_path)
            
            if self.Replace_on_the_ == "texture":
                raise Exception("not implemeted") 
                texture_clothes_tmp = mix_texture_with_mu_cpu(texture_clothes.copy_to_host() , self.mask_img.copy_to_host() , self.present_ringID_lists , self.uv_cod_list.copy_to_host() , loop = 5 ,face_flg = False , handsfeets_flg= True)    #in 4d datasets, should true&true? #FIXME:5sec かかっている。numba or pytorchで置き換える?
                print(type(texture_clothes_tmp))
                texture_clothes  = cuda.to_device(texture_clothes_tmp)
                self.mask_img    = cuda.as_cuda_array(self.mask_img)
                self.uv_cod_list = cuda.as_cuda_array(self.uv_cod_list)
                     
        if self.test_type == "naked":
            raise AssertionError("not implemented or not tested") #FIXME
            ### Calculate trainslation(joints_smpl(=smpl t_pose) -> joints(=posed)
            skeleton = np.zeros([24,4,4] ,np.float32)
            joints = skinning_utils.LoadSkeleton(pose , self.Tshapecoarsejoints , self.kintree  )   #pose (Angle)       
            for j in range(24):
                skeleton[j] = np.dot(self.joints_smpl[j] , skinning_utils.InverseTransfo(joints[j]))  
        else:
            ### Calculate trainslation(joint_T(=data depend t_pose) -> joints(=posed)
            skeleton = np.zeros([24,4,4] ,np.float32)
            joints = skinning_utils.LoadSkeleton(pose , self.T_joints , self.kintree )   #pose (Angle)         
            for j in range(24):
                skeleton[j] = np.dot(self.joint_T[j] , skinning_utils.InverseTransfo(joints[j]))  

        threads_per_block = 64  #i.e) block dim
        blocks_per_grid = int(divUp(self.vtx_num,threads_per_block))     #i.e) grid dim  
        #TODO:各gpuコードの中身に共通する部分が多いので、局所的な関数を用意して整理する。
        if self.test_type == "naked":              
            raise AssertionError("not implemented or not tested") #FIXME
            if self.naked_texture_type == "Canonical":
                #Reconstruct_CanonicalNakedmesh_from_CanonicalNakedTexture_gpu[blocks_per_grid,threads_per_block](self.out_vtx ,self.uv_cod_list , self.Template_model_vtx , self.vtx_num , texture_naked , self.min_for_naked , self.max_for_naked)
                Reconstruct_nakedmesh_from_CanonicalNakedTexture_gpu[blocks_per_grid,threads_per_block](self.out_vtx ,self.uv_cod_list , self.Template_model_vtx , self.vtx_num , self.skinWeights , skeleton , texture_naked , self.min_for_naked , self.max_for_naked )
            elif self.naked_texture_type == "Posed":
                Reconstruct_nakedmesh_from_PosedNakedTexture_gpu[blocks_per_grid,threads_per_block](self.predictedNaked_posed_vtx ,self.uv_cod_list , self.Template_model_vtx  , self.vtx_num , self.basis_list , self.skinWeights , skeleton ,texture_naked, self.min_for_naked , self.max_for_naked )
                if self.naked_topology == "Outershell":
                    self.out_vtx = self.predictedNaked_posed_vtx                
                elif self.naked_topology == "SMPL":
                    blocks_per_grid = int(divUp(self.vtx_num_outershell,threads_per_block))     #i.e) grid dim  
                    Dilatation_smpl_to_Outershell_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.predictedNaked_posed_vtx , self.bary_cod_cuda , self.correspond_face_cuda , self.vtx_num_outershell)
        elif self.test_type == "clothes" or self.test_type == "clothes_and_color":
            if self.Orthonormalize:
                if self.reconst_mode == 0:
                    Reconstruct_mesh_orthonormal_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.shellTemplate_posed_vtx, self.handfeet_dist , self.parts_number ,self.uv_cod_list , self.shellTemplate_t_vtx , self.vtx_num ,  self.t_basis , self.skinWeights , skeleton ,texture_clothes , self.mu_list , self.gamma , self.norm_a , self.mask_img)
                elif self.reconst_mode == 1:
                    #Reconstruct_mesh_orthonormal_2path_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.predictedNaked_posed_vtx, self.handfeet_dist , self.parts_number ,uv_cod_list , t_vtx , self.vtx_num ,  self.t_basis , self.skinWeights , skeleton ,texture_clothes , self.mu_list , self.gamma , self.norm_a , self.mask_img)
                    raise AssertionError("not implemented or not tested") #FIXME
                elif self.reconst_mode == 2:
                    #Reconstruct_mesh_orthonormal_2path_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.predictedNaked_posed_vtx, self.handfeet_dist , self.parts_number ,uv_cod_list , t_vtx , self.vtx_num ,  self.t_basis , self.skinWeights , skeleton ,texture_clothes , self.mu_list , self.gamma , self.norm_a , self.mask_img , mode = 2)
                    raise AssertionError("not implemented or not tested") #FIXME
            else:      
                if self.reconst_mode == 0:
                    Reconstruct_mesh_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.shellTemplate_posed_vtx, self.handfeet_dist , self.parts_number ,self.uv_cod_list , self.shellTemplate_t_vtx , self.vtx_num ,  self.t_basis , self.skinWeights , skeleton ,texture_clothes , self.mu_list , self.gamma , self.norm_a , self.mask_img)
                    raise AssertionError("not implemented or not tested") #FIXME
                elif self.reconst_mode == 1:
                    if self.Replace_on_the_ == "mesh":
                        if self.cp_naked_inference == "smplpytorch":
                            if texture_mask != None:
                                raise AssertionError("not implemented or not tested") #FIXME
                                Reconstruct_mesh_2path_from_posedNaked_replace_on_mesh_with_mask_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.predictedNaked_posed_vtx  , self.shellTemplate_posed_vtx , self.handfeet_dist , self.parts_number ,self.uv_cod_list , self.Template_model_vtx , self.vtx_num ,  self.posed_basis_list, self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , smpl_vtxs, self.bary_cod_cuda , self.correspond_face_cuda , self.mu_list , self.gamma , self.norm_a , self.mask_img , texture_mask_cuda )
                            else:
                                Reconstruct_mesh_2path_from_posedNaked_replace_on_mesh_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.predictedNaked_posed_vtx  , self.shellTemplate_posed_vtx , self.handfeet_dist , self.parts_number ,self.uv_cod_list , self.Template_model_vtx , self.vtx_num ,  self.posed_basis_list, self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , smpl_vtxs, self.bary_cod_cuda , self.correspond_face_cuda , self.mu_list , self.gamma , self.norm_a , self.mask_img )
                        elif self.naked_texture_type == "Canonical":       
                            raise AssertionError("not implemented or not tested") #FIXME
                            Reconstruct_mesh_2path_from_CanonicalNakedTexture_replace_on_mesh_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.predictedNaked_posed_vtx, self.handfeet_dist , self.parts_number ,self.uv_cod_list , self.Template_model_vtx , self.vtx_num , self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , texture_naked ,  self.mu_list , self.gamma , self.norm_a , self.mask_img , 1 , self.min_for_naked , self.max_for_naked)
                        elif self.naked_texture_type == "Posed":       
                            raise AssertionError("not implemented or not tested") #FIXME
                    else :  
                        if self.cp_naked_inference == "smplpytorch":
                            raise AssertionError("not implemented or not tested") #FIXME
                            Reconstruct_mesh_2path_from_posedNaked_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.predictedNaked_posed_vtx , self.shellTemplate_posed_vtx , self.uv_cod_list , self.Template_model_vtx , self.vtx_num , self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , smpl_vtxs, self.bary_cod_cuda , self.correspond_face_cuda , self.mu_list , self.gamma , self.norm_a)
                            #Reconstruct_debug_mu_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.predictedNaked_posed_vtx , self.Template_model_vtx , self.vtx_num , self.basis_list , self.skinWeights , self.skeleton_t , skeleton , smpl_vtxs, self.bary_cod_cuda , self.correspond_face_cuda , self.mu_list )
                        elif self.naked_texture_type == "Canonical":   
                            raise AssertionError("not implemented or not tested") #FIXME
                            Reconstruct_mesh_2path_from_CanonicalNakedTexture_gpu[blocks_per_grid,threads_per_block](self.out_vtx ,  self.predictedNaked_posed_vtx,self.uv_cod_list , self.Template_model_vtx  , self.vtx_num , self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , texture_naked, self.mu_list , self.gamma , self.norm_a , 1 , self.min_for_naked , self.max_for_naked )
                        elif self.naked_texture_type == "Posed":      
                            if self.naked_topology == "Outershell":
                                raise AssertionError("not implemented or not tested") #FIXME
                                Reconstruct_mesh_2path_from_PosedNakedTexture_gpu[blocks_per_grid,threads_per_block](self.out_vtx ,  self.predictedNaked_posed_vtx, self.uv_cod_list , self.Template_model_vtx  , self.vtx_num , self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , texture_naked, self.mu_list , self.gamma , self.norm_a , self.min_for_naked , self.max_for_naked )
                            elif self.naked_topology == "SMPL":
                                raise AssertionError("not implemented or not tested") #FIXME
                                Reconstruct_nakedmesh_from_PosedNakedTexture_gpu[blocks_per_grid,threads_per_block](self.predictedNaked_posed_vtx ,self.uv_cod_list , self.Template_model_vtx  , self.vtx_num , self.basis_list , self.skinWeights , skeleton ,texture_naked, self.min_for_naked , self.max_for_naked )
                                blocks_per_grid = int(divUp(self.vtx_num_outershell,threads_per_block))     #i.e) grid dim  
                                Reconstruct_mesh_2path_from_posedNaked_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.predictedNaked_posed_outershell_vtx  , self.uv_cod_list_outershell , self.vtx_outershell , self.vtx_num_outershell , self.basis_list_outershell , self.skinWeights_outershell , self.skeleton_t , skeleton ,texture_clothes , self.predictedNaked_posed_vtx, self.bary_cod_cuda , self.correspond_face_cuda , self.mu_list , self.gamma , self.norm_a)
                elif self.reconst_mode == 2:
                    if self.Replace_on_the_ == "mesh":  #not tested
                        raise AssertionError("not implemented or not tested") #FIXME
                        Reconstruct_mesh_2path_from_CanonicalNakedTexture_replace_on_mesh_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.replaced_out_vtx, self.predictedNaked_posed_vtx, self.handfeet_dist , self.parts_number ,self.uv_cod_list , self.Template_model_vtx , self.vtx_num , self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , texture_naked , self.mu_list , self.gamma , self.norm_a , self.mask_img , 1 , self.min_for_naked , self.max_for_naked)
                    else:                               #not tested
                        raise AssertionError("not implemented or not tested") #FIXME
                        Reconstruct_mesh_2path_from_CanonicalNakedTexture_gpu[blocks_per_grid,threads_per_block](self.out_vtx , self.predictedNaked_posed_vtx, self.uv_cod_list , self.Template_model_vtx  , self.vtx_num , self.basis_list , self.skinWeights , self.skeleton_t , skeleton ,texture_clothes , texture_naked , self.mu_list , self.gamma , self.norm_a , 2 , self.min_for_naked , self.max_for_naked)
            
            if self.Replace_on_the_ == "mesh":      #TODO:gpuコードにする
                replaced_out_vtx = self.replaced_out_vtx.copy_to_host()
                handfeet_dist    = self.handfeet_dist.copy_to_host()
                parts_number     = self.parts_number.copy_to_host()
                #translate hands and feets after replacing them
                #Rhand dist avg
                Rhand_dist_sum = np.sum(handfeet_dist[:,0],axis = 0)
                Rhand_cnt = np.count_nonzero(np.any(handfeet_dist[:,0] ,axis = 1))
                Rhand_dist = Rhand_dist_sum / Rhand_cnt

                #Lhand dist avg
                Lhand_dist_sum = np.sum(handfeet_dist[:,1],axis = 0)
                Lhand_cnt = np.count_nonzero(np.any(handfeet_dist[:,1] ,axis = 1))
                Lhand_dist = Lhand_dist_sum / Lhand_cnt

                #hand dist avg
                Rfeet_dist_sum = np.sum(handfeet_dist[:,2],axis = 0)
                Rfeet_cnt = np.count_nonzero(np.any(handfeet_dist[:,2] ,axis = 1))
                Rfeet_dist = Rfeet_dist_sum / Rfeet_cnt

                #Rhand dist avg
                Lfeet_dist_sum = np.sum(handfeet_dist[:,3],axis = 0)
                Lfeet_cnt = np.count_nonzero(np.any(handfeet_dist[:,3] ,axis = 1))
                Lfeet_dist = Lfeet_dist_sum / Lfeet_cnt

                #Face dist avg
                """
                Face_dist_sum = np.sum(handfeet_dist[:,4],axis = 0)
                Face_cnt = np.count_nonzero(np.any(handfeet_dist[:,4] ,axis = 1))
                Face_dist = Face_dist_sum / Face_cnt
                """
                
                #translate to correct position
                for j in range(self.vtx_num):
                    if np.all(replaced_out_vtx[j] == np.zeros(3)):
                        continue
                    if   parts_number[j] == 1:   #Rhand
                        replaced_out_vtx[j] = replaced_out_vtx[j] - Rhand_dist
                    elif parts_number[j] == 2:   #Lhand
                        replaced_out_vtx[j] = replaced_out_vtx[j] - Lhand_dist
                    elif parts_number[j] == 3:   #Rfeet
                        replaced_out_vtx[j] = replaced_out_vtx[j] - Rfeet_dist
                    elif parts_number[j] == 4:   #Lhand
                        replaced_out_vtx[j] = replaced_out_vtx[j] - Lfeet_dist
                    """
                    elif parts_number[j] == 5:   #Face
                        replaced_out_vtx[j] = replaced_out_vtx[j] - Face_dist
                    """

                replaced_out_vtx = replaced_out_vtx.tolist()
                replaced_out_vtx = reconstruct_utils.interp_on_mesh(replaced_out_vtx, self.vtx_num , self.edge_oneling_neighbor_vtxID_dicts , loop = 3) 

                if self.save_mode == "debug" or self.save_mode == "result":
                    new_faces = delete_invalid_faces_cpu(replaced_out_vtx , self.face_outershell , self.face_num_outershell)
                    reconstIPL_path     = os.path.join(self.reconst_dir, r"IPLMesh_" + str(fileID).zfill(4) +".ply")
                    save_ply(reconstIPL_path ,replaced_out_vtx , nml = None ,rgb = None, face=new_faces.tolist(), vtx_num=self.vtx_num_outershell, face_num=self.face_num_outershell)
            
        if self.save_mode == "debug" or self.save_mode == "result":
            out_vtx = self.out_vtx.copy_to_host()
            new_faces = delete_invalid_faces_cpu(out_vtx , self.face_outershell , self.face_num_outershell)
            reconst_path        = os.path.join(self.reconst_dir,r"mesh_" + str(fileID).zfill(4) +".ply")
            save_ply(reconst_path ,out_vtx , nml = None  ,rgb = None, face=new_faces.tolist(), vtx_num=self.vtx_num_outershell, face_num=self.face_num_outershell)
        elif self.save_mode == "display" and os.name == 'posix':
            display_mesh = Mesh(v= self.out_vtx.copy_to_host(), f=self.face)
            self.mv.set_static_meshes([display_mesh])

        #save posed surface
        if self.save_mode == "debug":
            if self.reconst_mode == 0:
                shellTemplate_posed_path  = os.path.join(self.reconst_dir,r"shellTemplate_posed_"  + str(fileID).zfill(4) +".ply")
                save_ply(shellTemplate_posed_path , self.shellTemplate_posed_vtx , nml = None ,rgb = None, face=self.face_outershell, vtx_num=self.vtx_num_outershell, face_num=self.face_num_outershell)
            if self.reconst_mode == 1 or self.reconst_mode == 2:
                predictedNaked_posed_path = os.path.join(self.reconst_dir,r"predictedNaked_posed_" + str(fileID).zfill(4) +".ply")
                save_ply(predictedNaked_posed_path , self.predictedNaked_posed_vtx.copy_to_host() , nml = None ,rgb = None, face=self.face_outershell, vtx_num=self.vtx_num_outershell, face_num=self.face_num_outershell)
                
                shellTemplate_posed_path  = os.path.join(self.reconst_dir,r"shellTemplate_posed_"  + str(fileID).zfill(4) +".ply")
                shellTemplate_posed_vtx_np = self.shellTemplate_posed_vtx.copy_to_host()
                save_ply(shellTemplate_posed_path , shellTemplate_posed_vtx_np , nml = None ,rgb = None, face=self.face_outershell, vtx_num=self.vtx_num_outershell, face_num=self.face_num_outershell)
                
                display_local_coordinate_system_as_arrow_on_ply(self.posed_basis_list.copy_to_host() , shellTemplate_posed_vtx_np , self.face_outershell , self.vtx_num_outershell , self.face_num_outershell , self.reconst_dir, str(fileID).zfill(4))

        if self.test_type == "clothes_and_color":   
            raise AssertionError("not implemented or not tested") #FIXME
            #texture_color = np.load(texture_color_folder_pathlist[0])   #debug with 1st texture
            print("texture_color_path : " , texture_color_folder_pathlist[i])
            texture_color = np.load(texture_color_folder_pathlist[i])
            height_clr = texture_color.shape[0]
            width_clr = texture_color.shape[1]
            print("color resolution : " ,height_clr," , ",width_clr)
            out_clr = np.zeros((dds_vtx_num , 3) , dtype= "uint8")
            Reconstruct_mesh_color_gpu[blocks_per_grid,threads_per_block](self.out_vtx , out_clr , uv_cod_list , grid_x_list , grid_y_list , uv_cod_color_list , grid_x_color_list , grid_y_color_list , dds_vtx , dds_vtx_num ,  self.t_basis , self.skinWeights , skeleton ,texture_clothes , texture_color , mu_list , gamma , valid_uv_list , valid_uv_num_list , height , width , height_clr , width_clr, self.norm_a)

            debug_time = time.time()

            out_clr = out_clr.tolist()
            save_ply(reconst_path ,self.out_vtx , nml = None ,rgb = out_clr, face=dds_face, vtx_num=dds_vtx_num, face_num=dds_face_num)
            print("debug_time : " , time.time() - debug_time)

        #posed_basis = np.concatenate((Local_posed_basis_f1[:,np.newaxis,:],Local_posed_basis_f2[:,np.newaxis,:],Local_posed_basis_f3[:,np.newaxis,:]) , axis = 1)
        
        #PosedSurface_path = os.path.join(args.PosedSurface_path , "SkinnedSurface_" + os.path.basename(texture_disp_folder_pathlist[i]).split(".")[0].split("_")[-1] + ".ply")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig): 
    Reconstructor = AITS_Reconstructor(cfg) 
    Reconstructor.asset_loader()            
    Reconstructor.reconst_preprocess()   
    
    smpl_folder_path_test     = cfg.train.smpl_folder_path_test 
    GTtexture_folderpath      = cfg.train.GTtexture_folderpath
    if cfg.train.test_type == "naked":
        dataset   = dataloader.Dataloader(cfg.train.test_type, smpl_folder_path = smpl_folder_path_test, texture_disp_folder_path= GTtexture_folderpath, texture_color_folder_path = None , load_smpl_set_flg = False ,duplicate_smpl = True , padding1024_flg = False , use_skeleton_aware = True )
        #dataset_smplparam = dataloader.Dataloader("test_naked", smpl_folder_path = smpl_folder_path_test, texture_disp_folder_path= None , texture_color_folder_path = None , load_smpl_set_flg = False ,duplicate_smpl = True , padding1024_flg = False , use_skeleton_aware = True )
    
    elif cfg.train.test_type == "clothes":
        kintree             = np.fromfile("../assets/kintree.bin" , np.int32)
        Tshapecoarsejoints  = np.fromfile("../assets/Tshapecoarsejoints.bin" , np.float32)           #smpl t_pose (position)
        T_joints            = np.fromfile(os.path.join(cfg.train.Canonical_Mesh_folder , "T_joints.bin") , np.float32)      #data depend t_pose (position)
        
        smplpytorch_processor_instance = smplpytorch_processor("D:/Project/Human/AITS/avatar-in-the-shell/assets/heavy" , gender = Reconstructor.smpl_gender)
        if Reconstructor.use_beta == True:
            beta = np.load(Reconstructor.beta_path)
        else:
            smplpytorch_processor_instance.rescale_smpl_model(Tshapecoarsejoints , T_joints , kintree)
            beta = None
        smplpytorch_processor_instance.init_SMPL_layer()

        dataset   = dataloader.Dataloader("debug", smpl_folder_path = smpl_folder_path_test, texture_disp_folder_path= GTtexture_folderpath, texture_color_folder_path = None , load_smpl_set_flg = True ,duplicate_smpl = True , padding1024_flg = False , use_skeleton_aware = True )
        #dataset_smplparam = dataloader.Dataloader("test_fix", smpl_folder_path = smpl_folder_path_test, texture_disp_folder_path= None , texture_color_folder_path = None , load_smpl_set_flg = True ,duplicate_smpl = True , padding1024_flg = False , use_skeleton_aware = True)

    #for datas_texture , datas_smplparam in tqdm.tqdm(zip(dataset_texture , dataset_smplparam)):
    for i, datas_texture  in tqdm.tqdm(enumerate(dataset)):
        fileID , smpl_input , texture , texture_mask  , _ , trans = datas_texture
        #fileID , _ , texture , _ = datas_texture
        #_ , smpl_input                 = datas_smplparam
        smpl_for_reconst = smpl_input[48:,:]

        #smpl_for_reconst = skinning_utils.fix_smpl_parts(smpl_for_reconst , fix_hands = True , fix_foots = True , fix_wrists = False)
        texture_tensor   = torch.Tensor(texture).to("cuda:0")
        texture_tensor   = texture_tensor.permute((1,2,0)) 

        smpl_verts       = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(smpl_for_reconst , beta)
        predicted_texture_clothes_mask = torch.from_numpy(texture_mask[:,:,np.newaxis].astype(np.float32)).clone().cuda()
        
        if trans is not None:
            smpl_verts = smpl_verts + torch.from_numpy(trans.astype(np.float32)).cuda()
        if cfg.train.test_type == "naked":  
            Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = None, texture_naked = texture_tensor , texture_clothes = None , texture_color = None )   
        if cfg.train.test_type == "clothes" :
            Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = smpl_verts, texture_naked = None , texture_clothes = texture_tensor , texture_color = None  , texture_mask = predicted_texture_clothes_mask)   
    
    ### Average Texture debug
    smpl_for_reconst = np.zeros((24,3) , np.float32)
    texture          = np.full((3, 768, 768) , 0.5 , np.float32)
    fileID           = 999

    #smpl_for_reconst = skinning_utils.fix_smpl_parts(smpl_for_reconst , fix_hands = True , fix_foots = True , fix_wrists = False)
    texture_tensor   = torch.Tensor(texture).to("cuda:0")
    texture_tensor   = texture_tensor.permute((1,2,0)) 

    smpl_verts       = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(smpl_for_reconst , beta)
    Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = smpl_verts, texture_naked = None , texture_clothes = texture_tensor , texture_color = None )   


if __name__ == "__main__":
    main()
