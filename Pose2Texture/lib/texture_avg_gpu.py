from numba.cuda.simulator import kernel
import numpy as np
import cv2
import tqdm
from numba import jit,cuda
import time
import glob
from natsort import natsorted
import os
import sys

#texture_path = r"displacement_texture.npy"
#texture_float = np.load(texture_path)
"""
height = texture_float.shape[0]
width = texture_float.shape[1]
"""

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

@cuda.jit
def texture_avg_disp_cuda(texture_float,copy_texture,kernel_size,sum):
    """# Thread id in a 1D block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # Block id in a 2D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    h = tx + bx * bw
    w = ty + by * bh"""
    h,w = cuda.grid(2)

    height = texture_float.shape[0]
    width = texture_float.shape[1]

    r = int(kernel_size/2)
    counter = 0
    if texture_float[h,w][0] == 0 and texture_float[h,w][1] == 0 and texture_float[h,w][2] == 0 :
        for j in range(kernel_size):
            y = h + j - r
            if y < 0 or y > height-1 :
                continue
            for i in range(kernel_size):
                x = w + i - r
                if x < 0 or x > width -1:
                    continue
                if texture_float[y,x][0] != 0 or texture_float[y,x][1] != 0 or texture_float[y,x][2] != 0:
                    sum[h,w][0] += texture_float[y,x][0]
                    sum[h,w][1] += texture_float[y,x][1]
                    sum[h,w][2] += texture_float[y,x][2]
                    counter += 1
        if counter > 0:
            copy_texture[h,w][0] = sum[h,w][0] / float(counter)
            copy_texture[h,w][1] = sum[h,w][1] / float(counter)
            copy_texture[h,w][2] = sum[h,w][2] / float(counter)
    else:
        copy_texture[h,w][0] = texture_float[h,w][0]
        copy_texture[h,w][1] = texture_float[h,w][1]
        copy_texture[h,w][2] = texture_float[h,w][2]

@cuda.jit
def texture_avg_sw_cuda(texture_float,copy_texture,kernel_size,sum):
    """# Thread id in a 1D block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # Block id in a 2D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    h = tx + bx * bw
    w = ty + by * bh"""
    h,w = cuda.grid(2)

    height = texture_float.shape[0]
    width = texture_float.shape[1]

    r = int(kernel_size/2)
    for p in range(24):
        counter = 0
        if (texture_float[h,w][p])  == 0:
            for j in range(kernel_size):
                y = h + j - r
                if y < 0 or y > height-1 :
                    continue
                for i in range(kernel_size):
                    x = w + i - r
                    if x < 0 or x > width -1:
                        continue
                    if (texture_float[y,x][p]  != 0):
                        sum[h,w][p]  += texture_float[y,x][p]
                        counter += 1
            if counter > 0:
                copy_texture[h,w][p]  = sum[h,w][p] / float(counter)
        else:
            copy_texture[h,w][p]  = texture_float[h,w][p]
                

    """
    #if np.all(texture_float[h,w] == 0) :
    if (texture_float[h,w][0]  == 0 and texture_float[h,w][1]  == 0 and texture_float[h,w][2]  == 0 and 
        texture_float[h,w][3]  == 0 and texture_float[h,w][4]  == 0 and texture_float[h,w][5]  == 0 and 
        texture_float[h,w][6]  == 0 and texture_float[h,w][7]  == 0 and texture_float[h,w][8]  == 0 and 
        texture_float[h,w][9]  == 0 and texture_float[h,w][10] == 0 and texture_float[h,w][11] == 0 and 
        texture_float[h,w][12] == 0 and texture_float[h,w][13] == 0 and texture_float[h,w][14] == 0 and 
        texture_float[h,w][15] == 0 and texture_float[h,w][16] == 0 and texture_float[h,w][17] == 0 and 
        texture_float[h,w][18] == 0 and texture_float[h,w][19] == 0 and texture_float[h,w][20] == 0 and 
        texture_float[h,w][21] == 0 and texture_float[h,w][22] == 0 and texture_float[h,w][23] == 0     ):
        for j in range(kernel_size):
            y = h + j - r
            if y < 0 or y > height-1 :
                continue
            for i in range(kernel_size):
                x = w + i - r
                if x < 0 or x > width -1:
                    continue
                if (texture_float[y,x][0]  != 0 or texture_float[y,x][1]  != 0 or texture_float[y,x][2]  != 0 or 
                    texture_float[y,x][3]  != 0 or texture_float[y,x][4]  != 0 or texture_float[y,x][5]  != 0 or 
                    texture_float[y,x][6]  != 0 or texture_float[y,x][7]  != 0 or texture_float[y,x][8]  != 0 or 
                    texture_float[y,x][9]  != 0 or texture_float[y,x][10] != 0 or texture_float[y,x][11] != 0 or 
                    texture_float[y,x][12] != 0 or texture_float[y,x][13] != 0 or texture_float[y,x][14] != 0 or 
                    texture_float[y,x][15] != 0 or texture_float[y,x][16] != 0 or texture_float[y,x][17] != 0 or 
                    texture_float[y,x][18] != 0 or texture_float[y,x][19] != 0 or texture_float[y,x][20] != 0 or 
                    texture_float[y,x][21] != 0 or texture_float[y,x][22] != 0 or texture_float[y,x][23] != 0     ):
                    for k in range(24):
                        #sum[h,w][k]  += texture_float[y,x][k]
                        sum[h,w][0]  += texture_float[y,x][0]
                        sum[h,w][1]  += texture_float[y,x][1]
                        sum[h,w][2]  += texture_float[y,x][2]
                        sum[h,w][3]  += texture_float[y,x][3]
                        sum[h,w][4]  += texture_float[y,x][4]
                        sum[h,w][5]  += texture_float[y,x][5]
                        sum[h,w][6]  += texture_float[y,x][6]
                        sum[h,w][7]  += texture_float[y,x][7]
                        sum[h,w][8]  += texture_float[y,x][8]
                        sum[h,w][9]  += texture_float[y,x][9]
                        sum[h,w][10] += texture_float[y,x][10]
                        sum[h,w][11] += texture_float[y,x][11]
                        sum[h,w][12] += texture_float[y,x][12]
                        sum[h,w][13] += texture_float[y,x][13]
                        sum[h,w][14] += texture_float[y,x][14]
                        sum[h,w][15] += texture_float[y,x][15]
                        sum[h,w][16] += texture_float[y,x][16]
                        sum[h,w][17] += texture_float[y,x][17]
                        sum[h,w][18] += texture_float[y,x][18]
                        sum[h,w][19] += texture_float[y,x][19]
                        sum[h,w][20] += texture_float[y,x][20]
                        sum[h,w][21] += texture_float[y,x][21]
                        sum[h,w][22] += texture_float[y,x][22]
                        sum[h,w][23] += texture_float[y,x][23]
                    counter += 1
        if counter > 0:
            copy_texture[h,w][0]  = sum[h,w][0] / float(counter)
            copy_texture[h,w][1]  = sum[h,w][1] / float(counter)
            copy_texture[h,w][2]  = sum[h,w][2] / float(counter)
            copy_texture[h,w][3]  = sum[h,w][3] / float(counter)
            copy_texture[h,w][4]  = sum[h,w][4] / float(counter)
            copy_texture[h,w][5]  = sum[h,w][5] / float(counter)
            copy_texture[h,w][6]  = sum[h,w][6] / float(counter)
            copy_texture[h,w][7]  = sum[h,w][7] / float(counter)
            copy_texture[h,w][8]  = sum[h,w][8] / float(counter)
            copy_texture[h,w][9]  = sum[h,w][9] / float(counter)
            copy_texture[h,w][10] = sum[h,w][10] / float(counter)
            copy_texture[h,w][11] = sum[h,w][11] / float(counter)
            copy_texture[h,w][12] = sum[h,w][12] / float(counter)
            copy_texture[h,w][13] = sum[h,w][13] / float(counter)
            copy_texture[h,w][14] = sum[h,w][14] / float(counter)
            copy_texture[h,w][15] = sum[h,w][15] / float(counter)
            copy_texture[h,w][16] = sum[h,w][16] / float(counter)
            copy_texture[h,w][17] = sum[h,w][17] / float(counter)
            copy_texture[h,w][18] = sum[h,w][18] / float(counter)
            copy_texture[h,w][19] = sum[h,w][19] / float(counter)
            copy_texture[h,w][20] = sum[h,w][20] / float(counter)
            copy_texture[h,w][21] = sum[h,w][21] / float(counter)
            copy_texture[h,w][22] = sum[h,w][22] / float(counter)
            copy_texture[h,w][23] = sum[h,w][23] / float(counter)
    else:
        copy_texture[h,w][0]  = texture_float[h,w][0]
        copy_texture[h,w][1]  = texture_float[h,w][1]
        copy_texture[h,w][2]  = texture_float[h,w][2]
        copy_texture[h,w][3]  = texture_float[h,w][3]
        copy_texture[h,w][4]  = texture_float[h,w][4]
        copy_texture[h,w][5]  = texture_float[h,w][5]
        copy_texture[h,w][6]  = texture_float[h,w][6]
        copy_texture[h,w][7]  = texture_float[h,w][7]
        copy_texture[h,w][8]  = texture_float[h,w][8]
        copy_texture[h,w][9]  = texture_float[h,w][9]
        copy_texture[h,w][10] = texture_float[h,w][10]
        copy_texture[h,w][11] = texture_float[h,w][11]
        copy_texture[h,w][12] = texture_float[h,w][12]
        copy_texture[h,w][13] = texture_float[h,w][13]
        copy_texture[h,w][14] = texture_float[h,w][14]
        copy_texture[h,w][15] = texture_float[h,w][15]
        copy_texture[h,w][16] = texture_float[h,w][16]
        copy_texture[h,w][17] = texture_float[h,w][17]
        copy_texture[h,w][18] = texture_float[h,w][18]
        copy_texture[h,w][19] = texture_float[h,w][19]
        copy_texture[h,w][20] = texture_float[h,w][20]
        copy_texture[h,w][21] = texture_float[h,w][21]
        copy_texture[h,w][22] = texture_float[h,w][22]
        copy_texture[h,w][23] = texture_float[h,w][23]
    """

def texture_avg(texture,data=None,iter = 30,kernel_size = 15):
    height = texture.shape[0]
    width = texture.shape[1]

    threads_per_block = 16 , 16   #i.e) block dim
    blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim  

    if data == "disp" or  data == "color":
        for it in tqdm.tqdm(range(iter)):
            copy_texture = texture.copy()
            sum = np.zeros([height,width,3],dtype=np.float32)
            texture_avg_disp_cuda[blocks_per_grid,threads_per_block](texture,copy_texture,kernel_size,sum)
            texture = copy_texture.copy()
    elif data == "sw":
        print("debug")
        
        iter = 30
        kernel_size = 5
        for it in tqdm.tqdm(range(iter)):
            copy_texture = texture.copy()
            sum = np.zeros([height,width,24],dtype=np.float32)
            texture_avg_sw_cuda[blocks_per_grid,threads_per_block](texture,copy_texture,kernel_size,sum)
            texture = copy_texture.copy()
        """
        iter = 100
        kernel_size = 30
        for it in tqdm.tqdm(range(iter)):
            copy_texture = texture.copy()
            sum = np.zeros([height,width,24],dtype=np.float32)
            texture_avg_sw_cuda[blocks_per_grid,threads_per_block](texture,copy_texture,kernel_size,sum)
            texture = copy_texture.copy()
        """

    if data == None:
        print("select data")
        sys.exit()
    
    return texture


@cuda.jit
def dispToRGB(texture_float,norm_t,max,min):
    h,w = cuda.grid(2)
    if texture_float[h,w][0] != 0 or texture_float[h,w][1] != 0 or texture_float[h,w][2] != 0:
        #texture_float[h,w][0] = int(((texture_float[h,w][0] + norm_t) / (2*norm_t))* 255)
        #texture_float[h,w][1] = int(((texture_float[h,w][1] + norm_t) / (2*norm_t))* 255)
        #texture_float[h,w][2] = int(((texture_float[h,w][2] + norm_t) / (2*norm_t))* 255)
        """
        texture_float[h,w][0] = int(texture_float[h,w][0]* 255)
        texture_float[h,w][1] = int(texture_float[h,w][1]* 255)
        texture_float[h,w][2] = int(texture_float[h,w][2]* 255)
        """
        texture_float[h,w][0] = int(((texture_float[h,w][0] - min) / (max - min) * 255))
        texture_float[h,w][1] = int(((texture_float[h,w][1] - min) / (max - min) * 255))
        texture_float[h,w][2] = int(((texture_float[h,w][2] - min) / (max - min) * 255))
    else:
        texture_float[h,w][0] = 0
        texture_float[h,w][1] = 0
        texture_float[h,w][2] = 0


def displacement_show(texture_float,save_path,norm_t ,max = None , min = None):
    start = time.time()
    height = texture_float.shape[0]
    width = texture_float.shape[1]
    threads_per_block = 16 , 16   #i.e) block dim
    blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim 
    if max == None or min == None: 
        print("max == None or min == None")
        max = texture_float.max()
        min = texture_float.min()
    texture_float_copy = texture_float.copy()
    #print(texture_float_copy.shape)
    #print(texture_float_copy[0])

    dispToRGB[blocks_per_grid,threads_per_block](texture_float_copy,norm_t,max,min)
    #print("process time : ",time.time() - start)

    texture_int = texture_float_copy.astype(np.uint8)
    texture_int = cv2.cvtColor(texture_int, cv2.COLOR_RGB2BGR)
    #cv2.imshow("view",texture_int)
    #cv2.waitKey(0)
    cv2.imwrite(save_path,texture_int)

def displacement_show_sw(texture_float,save_path_root,save_base_name):
    start = time.time()
    height = texture_float.shape[0]
    width = texture_float.shape[1]
    threads_per_block = 16 , 16   #i.e) block dim
    blocks_per_grid =  int(divUp(height,threads_per_block[0])),   int(divUp(width,threads_per_block[1]))     #i.e) grid dim  
    
    for i in range(8):
        norm_t = 0
        texture_float_tmp = np.zeros((height,width,3))
        texture_float_tmp = texture_float[:,:,3*i:3*(i+1)].copy()
        #print(texture_float_tmp)
        dispToRGB[blocks_per_grid,threads_per_block](texture_float_tmp,norm_t)
        #print("process time : ",time.time() - start)

        texture_int = texture_float_tmp.astype(np.uint8)
        texture_int = cv2.cvtColor(texture_int, cv2.COLOR_RGB2BGR)
        #cv2.imshow("view",texture_int)
        #cv2.waitKey(0)
        save_path = os.path.join(save_path_root,save_base_name+"_"+str(i).zfill(4)+".png")
        print("save_sw_path:" , save_path)
        cv2.imwrite(save_path,texture_int)

if __name__ == "__main__":
    #texture_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Iwamoto\data\texture256_6a_trans\displacement_texture_*.npy"))
    #texture_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Cape_fitted_00159\data\texture_globalbasis\displacement_texture_*[0-9][0-9][0-9][0-9].npy"))
    #texture_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Cape_fitted_00159\data\texture_debug_before_normalization\displacement_texture_*[0-9][0-9][0-9][0-9].npy"))
    texture_mu_path = r"D:\Data\Human\HUAWEI\Cape_fitted_00159\data\Texture_All\texture_disp\Mu_displacement_texture.npy"
    #texture_mu_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\Texture_Extrapolation\.texture_color\color_texture_0003.npy"
    #texture_mu_path = r"D:\Project\Human\Avatar-In-The-Shell\avatar-in-the-shell\Pose2Texture\make_displacement\test_uv.npy"
    #texture_gamma_path = r"D:\Data\Human\HUAWEI\Cape_fitted_00159\data\texture_globalbasis\Gamma_displacement_texture.npy"
    #texture_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Female_ballet_dancer\data\texture_base_avgtexture\color_texture_*.npy"))
    #texture_sw_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Iwamoto\data\texture_base_avgtexture\sw_texture_*.npy"))
    #save_folder_path = r"D:\Data\Human\HUAWEI\Cape_fitted_00159\data\texture_rgb_before_normalization"
    #save_folder_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\tmp_rgb"

    #texture_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Iwamoto\data\debug_disp_texture_base_norm\*.npy"))
    #save_folder_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\debug_disp_texture_base_norm_rgb"

    #texture_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Iwamoto\data\color_texture_base_avgtexture\*.npy"))
    #save_folder_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\color_texture_base_avgtexture_rgb"

    ###old version norm
    #texture_paths = natsorted(glob.glob(r"D:\Data\Human\HUAWEI\Iwamoto\data\displacement_texture_base0_0.15\*.npy"))
    #save_folder_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\debug_disp_texture_base_norm_rgb_old"
    
    """
    max = -100000
    min = 100000
    for i, texture_path in enumerate(texture_paths):
        print(texture_path)
        texture_float = np.load(texture_path)
        max_tmp = texture_float.max()
        min_tmp = texture_float.min()
        if max_tmp > max:
            max = max_tmp
        elif min_tmp < min:
            min = min_tmp
    print("max:",max)
    print("min:",min)

    for i, texture_path in enumerate(texture_paths):
        texture_float = np.load(texture_path)
        save_path = os.path.join(save_folder_path,os.path.basename(texture_path).split(".")[0]+".png")
        print("load_path:" , texture_path)
        print("save_path:" , save_path)
        displacement_show(texture_float,save_path,norm_t = 0.18 , max = max , min = min)
    """

    save_folder_path = "."
    texture_float = np.load(texture_mu_path)
    save_path = os.path.join(save_folder_path,os.path.basename(texture_mu_path).split(".")[0]+".png")
    print("load_path:" , texture_mu_path)
    print("save_path:" , save_path)
    displacement_show(texture_float,save_path,norm_t = 0.18 , max = np.max(texture_float) , min = np.min(texture_float))
    """
    texture_float = np.load(texture_gamma_path)
    save_path = os.path.join(save_folder_path,os.path.basename(texture_gamma_path).split(".")[0]+".png")
    print("load_path:" , texture_gamma_path)
    print("save_path:" , save_path)
    displacement_show(texture_float,save_path,norm_t = 0.18)
    """

    """
    for i, texture_path in enumerate(texture_sw_paths):
        texture_float_sw = np.load(texture_path)
        save_path = os.path.join(save_folder_path,os.path.basename(texture_path).split(".")[0]+".png")
        save_sw_basename = os.path.basename(texture_sw_paths[i]).split(".")[0]
        print("load_path:" , texture_path)
        print("save_path:" , save_path)
        displacement_show_sw(texture_float_sw,save_folder_path,save_sw_basename)
    """

    """
    texture_path = r"D:\Project\Human\Pose2Texture\make_displacement\testdisp.npy"
    texture_float = np.load(texture_path)
    save_path = r"test_rgb.png"
    displacement_show(texture_float,save_path,norm_t = 0.18)
    """