from numba import jit,cuda
import os
import sys
import itertools
import numpy as np
import math
from copy import deepcopy
import torch
import cv2
import glob
from natsort import natsorted

@cuda.jit(device=True)
def my_bilinear_interplation(out , xs, ys, c, x, y):
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
    out[0] =  (1-dx) * (1-dy) * c[0] \
        +  (1-dx) *     dy * c[1] \
        +      dx * (1-dy) * c[2] \
        +      dx *     dy * c[3]

def interp_on_mesh(vtx , vtx_num , edge_oneling_neighbor_vtxID_dicts , loop = 3):
    debug_rgb = []
    for i in range(vtx_num):
        debug_rgb.append([0,0,0])

    for i in range(1,loop+1):
        edge_oneling_neighbor_vtxID_dict = edge_oneling_neighbor_vtxID_dicts[str(i)]
        ##### around edge (1st) ########################################
        for edge in edge_oneling_neighbor_vtxID_dict:
            debug_rgb[int(edge)] = [255,0,0]
            neighbors_list = edge_oneling_neighbor_vtxID_dict[edge]
            if vtx[int(edge)][0] != 0.0 or vtx[int(edge)][1] != 0.0 or vtx[int(edge)][2] != 0.0:
                base_x = vtx[int(edge)][0]
                base_y = vtx[int(edge)][1]
                base_z = vtx[int(edge)][2]
                sum_x = 0
                sum_y = 0
                sum_z = 0

                dist = []
                tmp_vtx = []
                for i in neighbors_list:
                    if vtx[i][0] != 0.0 or vtx[i][1] != 0.0 or vtx[i][2] != 0.0:
                        tmp_vtx.append(vtx[i])
                        d = ((base_x - vtx[i][0])**2 + (base_y - vtx[i][1])**2 + (base_z - vtx[i][2])**2) ** (1/2)
                        dist.append(d)
                #print("dist : " , dist)
                dist_for_gauss = deepcopy(dist)
                dist = np.array(dist)

                dist_for_gauss.append(0.0)
                #print("dist_for_gauss : " , dist_for_gauss)
                dist_for_gauss = np.array(dist_for_gauss)
                avg = np.average(dist_for_gauss)
                var = np.var(dist_for_gauss)
                #print("var : " , var)
                sum_weight = 0.0
                for i in range(dist.shape[0]):
                    d = dist[i]
                    
                    if var != 0.0:
                        weight = math.exp(-(((d - avg)**2)/(2*var)))
                    else:
                        weight = 1.0

                    if weight > 1.0:
                        print("strange")
                        sys.exit()
                    #print("weight : " , weight)
                    sum_weight += weight
                    sum_x += tmp_vtx[i][0] * weight
                    sum_y += tmp_vtx[i][1] * weight
                    sum_z += tmp_vtx[i][2] * weight
                
                #orijinal vtx
                if avg != 0 and var != 0:
                    weight = math.exp(-(((0.0 - avg)**2)/(2*var)))
                else:
                    weight = 1.0
                sum_weight += weight
                sum_x += base_x * weight
                sum_y += base_y * weight
                sum_z += base_z * weight

                if sum_weight != 1.0:
                    vtx[int(edge)][0] = sum_x / sum_weight
                    vtx[int(edge)][1] = sum_y / sum_weight
                    vtx[int(edge)][2] = sum_z / sum_weight      
                else:
                    vtx[int(edge)][0] =  vtx[int(edge)][0]
                    vtx[int(edge)][1] =  vtx[int(edge)][1]
                    vtx[int(edge)][2] =  vtx[int(edge)][2]
            """
            else:
                sum_x = 0
                sum_y = 0
                sum_z = 0
                sum_weight = 0.0
                for i in range(dist.shape[0]):
                    d = dist[i]
                    weight = math.exp(-((d**2)/(2*var)))
                    sum_weight += weight
                    sum_x += tmp_vtx[i][0] * weight
                    sum_y += tmp_vtx[i][1] * weight
                    sum_z += tmp_vtx[i][2] * weight

                if sum_weight != 0.0:
                    vtx[int(edge)][0] = sum_x / sum_weight
                    vtx[int(edge)][1] = sum_y / sum_weight
                    vtx[int(edge)][2] = sum_z / sum_weight      
                else:
                    vtx[int(edge)][0] =  0.0
                    vtx[int(edge)][1] =  0.0
                    vtx[int(edge)][2] =  0.0
            """
    ##############################################################    
    return vtx


PI = math.pi

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y



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
    
    if device == None:
        return Kernel
    else:
        return cuda.to_device(Kernel)

@cuda.jit('void(float32[:,:,:] , float32[:,:,:] , int8[:,:] , float32[:,:] , float32[:,:])')
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
        out_texture[h,w][0] = 0.5
        out_texture[h,w][1] = 0.5
        out_texture[h,w][2] = 0.5


def Mygaussian_gpu(out_texture ,in_texture , sum , mask, kernel , device):
    height = in_texture.shape[0]
    width  = in_texture.shape[0]
    threads_per_block = 16 , 16   #i.e) block dim
    blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim 
    
    if mask is not None:
        Mygaussian_cuda[blocks_per_grid,threads_per_block](in_texture ,out_texture ,mask ,kernel, sum)
    else:                                                   
        raise AssertionError("No mask")
    return out_texture


def make_mask(in_texture_path_dir , save_path):
    in_texture_paths = natsorted(glob.glob(os.path.join(in_texture_path_dir , "*.png")))
    mask_srcs = []
    len_texure = len(in_texture_paths)
    for i , in_texture_path in enumerate(in_texture_paths):
        if i % int(len_texure/10) != 0:
            continue
        #a = cv2.imread(r"D:\Data\Human\HUAWEI\Cape_fitted_00159\data\Texture_All\texture_src\debug_texture_rgb_0002.png")
        in_texture = cv2.imread(in_texture_path)
        mask_srcs.append(cv2.inRange(in_texture , (128,128,128) , (128,128,128)))
    mask_srcs = np.array(mask_srcs)
    mask_src = np.where(np.all(mask_srcs , axis= 0) , 255 , 0).astype(np.uint8)
    img_show = np.dstack([mask_src[:,:,np.newaxis],mask_src[:,:,np.newaxis],mask_src[:,:,np.newaxis]])
    cv2.imshow("show",img_show)
    cv2.waitKey(0)

    #Rhand
    retval, img, mask, rect  = cv2.floodFill(
                        mask_src[:,:201]                #Outershell_topology
                        #mask_src                       #SMPL_topology
                        , mask= None
                        , seedPoint = (50,50)           #Outershell_topology
                        #, seedPoint = (330,20)         #SMPL_topology
                        , newVal=60
                        )

    #Lhand
    retval, img, mask, rect  = cv2.floodFill(mask_src 
                        , mask= None
                        , seedPoint = (305,60)         #Outershell_topology
                        #, seedPoint = (450,20)          #SMPL_topology
                        , newVal=90
                        )

    #Rfeet
    retval, img, mask, rect  = cv2.floodFill(mask_src 
                        , mask= None
                        , seedPoint = (450,113)        #Outershell_topology
                        #, seedPoint = (370,90)          #SMPL_topology
                        , newVal=120
                        )

    #Lfeet
    retval, img, mask, rect  = cv2.floodFill(mask_src 
                        , mask= None
                        , seedPoint = (660,75)         #Outershell_topology
                        #, seedPoint = (475,100)         #SMPL_topology
                        , newVal=150
                        )

    #Head
    retval, img, mask, rect  = cv2.floodFill(mask_src 
                        , mask= None
                        , seedPoint = (440 ,300)       #Outershell_topology
                        #, seedPoint = (150 ,300)        #SMPL_topology
                        , newVal=180
                        )



    img = np.where(img == 0 , 30 , img)

    img = np.dstack([img[:,:,np.newaxis],img[:,:,np.newaxis],img[:,:,np.newaxis]])

    #img[675:690 ,60:80] = 0
    #cv2.imwrite(r"D:\Data\Human\Template-star-0.015-0.05\mask.png" , img)
    cv2.imwrite(save_path , img)
    cv2.imshow("mask" , img)
    cv2.waitKey(0)

if __name__ == "__main__":
    in_texture_path_dir = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Cape_fitted_00159\data\Texture_naked_Canonical_Outershell"
    #in_texture_path_dir = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\4D_datasets\Cape_fitted_00159\data\Texture_naked_Posed_SMPL"
    save_path       = "mask.png"
    make_mask(in_texture_path_dir , save_path)