import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tqdm 
import time
from numba import jit,cuda
import json
import math
from scipy.linalg import norm
import struct
from copy import deepcopy
import os
import itertools
import pickle   

if __name__ == "__main__":
    from meshes_utils import load_obj,load_ply  , save_ply  , save_obj , get_vtxID2uvIDs
else:
    from lib.meshes_utils import load_obj,load_ply  , save_ply  , save_obj , get_vtxID2uvIDs
    
def dispToRGB(texture_float):
    norm_t = 0.40
    height = texture_float.shape[0]
    width = texture_float.shape[1]
    for h in range(height):
        for w in range(width):
            txr = texture_float[h,w]
            if any(txr):
                #texture_float[h,w] = list(map(int,((txr + norm_t) / (2*norm_t))* 255))
                texture_float[h,w] = list(map(int,txr* 255))
                #print(texture_float[h,w])
            else:
                texture_float[h,w] = np.array([0,0,0])
    return texture_float    

def round_6(n):
    return round(n,6)

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

def load_uv_raw_mapping(uv_path,vtx_num = 93894):   #vtx_num = 93894 is hardcoding for our task
    print("uv_path :" , uv_path)
    f = open(uv_path,"rb")
    uv_bin = f.read()
    uv_list = []
    for i in range(vtx_num):
        uv_list_tmp = [struct.unpack("f",uv_bin[8*i:8*i+4])[0],struct.unpack("f",uv_bin[8*i+4:8*i+8])[0]]
        uv_list.append(uv_list_tmp)
    uv = np.array(uv_list)
    uv = uv.astype(np.float32)
    return uv

def drawing_cpu2(dispfaces,FaceIDTexture , BaryTexture,height,width,data = "float"):
    print("drawing_cpu2")
    start = time.time()
    #make texture
    if data == "float":
        texture_float = np.zeros((height,width,3), dtype=np.float32)
    elif data == "int":
        texture_int = np.zeros((height,width,3), dtype=np.uint8)

    uint_size = 4294967295
    for y in tqdm.tqdm(range(height)):
        for x in range(width):
            if FaceIDTexture[y][x] < uint_size:
                faceID = FaceIDTexture[y][x]
                alpha = BaryTexture[y][x][0]
                beta  = BaryTexture[y][x][1]
                gamma = BaryTexture[y][x][2]
                
                rgb = np.array([dispfaces[faceID][0],dispfaces[faceID][1],dispfaces[faceID][2]])

                color = (rgb[0]*alpha + rgb[1]*beta + rgb[2]*gamma)
                if data == "float":
                    texture_float[y,x] = list(map(round_6,color))
                elif data == "int":
                    texture_int[y,x] = list(map(round,color))

    print(time.time()-start)
    if data == "float":
        return texture_float
    elif data == "int":
        return texture_int

@cuda.jit
def drawing_gpu2(texture_float,rgb,FaceIDTexture , BaryTexture , height , width):
    h,w = cuda.grid(2)
    uint_size = 4294967295
    
    if h < height and w < width:
        if (FaceIDTexture[h][w] < uint_size)  :
            faceID = FaceIDTexture[h][w]
            #if all vertices in face are valid
            if ((rgb[faceID][0][0] != 0.0) or (rgb[faceID][0][1] != 0.0) or (rgb[faceID][0][2] != 0.0)) and \
                ((rgb[faceID][1][0] != 0.0) or (rgb[faceID][1][1] != 0.0) or (rgb[faceID][1][2] != 0.0)) and \
                ((rgb[faceID][2][0] != 0.0) or (rgb[faceID][2][1] != 0.0) or (rgb[faceID][2][2] != 0.0))   :
                alpha = BaryTexture[h][w][0]
                beta  = BaryTexture[h][w][1]
                gamma = BaryTexture[h][w][2]

                #color = (rgb[faceID][0]*alpha + rgb[faceID][1]*beta + rgb[faceID][2]*gamma)
                color = (round(rgb[faceID][0][0]*alpha + rgb[faceID][1][0]*beta + rgb[faceID][2][0]*gamma,6), 
                         round(rgb[faceID][0][1]*alpha + rgb[faceID][1][1]*beta + rgb[faceID][2][1]*gamma,6),
                         round(rgb[faceID][0][2]*alpha + rgb[faceID][1][2]*beta + rgb[faceID][2][2]*gamma,6))
                texture_float[h,w] = color

@cuda.jit
def norm(p0,p1):
    return ((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)**0.5

@cuda.jit
def drawing_gpu3(texture_float,uv_raw,FaceIDTexture , BaryTexture ):
    h,w = cuda.grid(2)
    uint_size = 4294967295
    if FaceIDTexture[h][w] < uint_size:
        faceID = FaceIDTexture[h][w]
        alpha = BaryTexture[h][w][0]
        beta =  BaryTexture[h][w][1]
        gamma = BaryTexture[h][w][2]
        
        dis01 = ((uv_raw[faceID][0][0] - uv_raw[faceID][1][0])**2 + (uv_raw[faceID][0][1] - uv_raw[faceID][1][1])**2)**0.5
        dis12 = ((uv_raw[faceID][1][0] - uv_raw[faceID][1][0])**2 + (uv_raw[faceID][1][1] - uv_raw[faceID][1][1])**2)**0.5
        dis20 = ((uv_raw[faceID][2][0] - uv_raw[faceID][1][0])**2 + (uv_raw[faceID][2][1] - uv_raw[faceID][1][1])**2)**0.5

        """
        dis01 = norm(uv_raw[faceID][0] , uv_raw[faceID][1])
        dis12 = norm(uv_raw[faceID][1] , uv_raw[faceID][2])
        dis20 = norm(uv_raw[faceID][2] , uv_raw[faceID][0])
        """
 
        if dis01 < 0.1 and dis12 < 0.1 and dis20 < 0.1:
            out_uv = (round(uv_raw[faceID][0][0]*alpha + uv_raw[faceID][1][0]*beta + uv_raw[faceID][2][0]*gamma,6), 
                    round(uv_raw[faceID][0][1]*alpha + uv_raw[faceID][1][1]*beta + uv_raw[faceID][2][1]*gamma,6))
        else:
            out_uv = (0.0,0.0)
        texture_float[h,w] = out_uv

@cuda.jit
def barycentric_interpolation_cuda(texture_float,triArr,rgb,ytop,xleft,p):
    h,w = cuda.grid(2)
    # Store the current point as a matrix
    p[h][w][0] = float(w+xleft)
    p[h][w][1] = float(h+ytop)
    p[h][w][2] = float(1)

    #p = np.array([[w+xleft], [h+ytop], [1]])

    # Solve for least squares solution to get barycentric coordinates
    (alpha, beta, gamma) = np.linalg.lstsq(triArr, p[h][w], rcond=-1)[0]

    # The point is inside the triangle if all the following conditions are met; otherwise outside the triangle
    if alpha > -1e-13 and beta > -1e-13 and gamma > -1e-13 :
        # do barycentric interpolation on colors
        color = (rgb[0]*alpha + rgb[1]*beta + rgb[2]*gamma)
        #texture_float[h+ytop,w+xleft] = list(map(round_6,color))
        #texture_float[h+ytop,w+xleft][0] = round_6(color[0])
        #texture_float[h+ytop,w+xleft][1] = round_6(color[1])
        #texture_float[h+ytop,w+xleft][2] = round_6(color[2])
        texture_float[h+ytop,w+xleft][0] = color[0]
        texture_float[h+ytop,w+xleft][1] = color[1]
        texture_float[h+ytop,w+xleft][2] = color[2]


def drawing_gpu(face2rgb):
    #make texture
    width = 1024
    height = 1024
    #texture_int = np.zeros((height,width,3), dtype=np.uint8)
    texture_float = np.zeros((height,width,3), dtype=np.float32)
    time_a = 0
    time_b = 0
    for face in tqdm.tqdm(face2rgb):
        ### make to GPU coding
        uv = [0,0,0]
        rgb = [0,0,0]
        for i in range(3):
            uv[i] = face[i][0]
            #rgb[i] = list(map(dispToRGB,face[i][1]))
            rgb[i] = face[i][1]

            #print(face[i][1])
        #print(rgb)
            
        # Make array of vertices
        # ax bx cx
        # ay by cy
        #  1  1  1
        #triArr = np.asarray([round(uv[0][0]*width),round(uv[1][0]*width),round(uv[2][0]*width), round(uv[0][1]*height),round(uv[1][1]*height),round(uv[2][1]*height), 1,1,1]).reshape((3, 3))
        triArr = np.asarray([uv[0][0]*width,uv[1][0]*width,uv[2][0]*width, uv[0][1]*height,uv[1][1]*height,uv[2][1]*height, 1.0,1.0,1.0]).reshape((3, 3))
        
        # Get bounding box of the triangle
        xleft = round(min(uv[0][0]*width, uv[1][0]*width, uv[2][0]*width))
        xright = round(max(uv[0][0]*width, uv[1][0]*width, uv[2][0]*width))
        ytop = round(min(uv[0][1]*height, uv[1][1]*height, uv[2][1]*height))
        ybottom = round(max(uv[0][1]*height, uv[1][1]*height, uv[2][1]*height))
        
                    
        gpu_height = ybottom - ytop + 2
        gpu_width = xright - xleft + 2

        p = []
        for i in range(gpu_height):
            p_t = []
            for j in range(gpu_width):
                p_t.append(np.array([[1.0], [1.0], [1.0]]))
            p.append(p_t)


        threads_per_block = 16 , 16   #i.e) block dim
        blocks_per_grid =  int(divUp(gpu_height,threads_per_block[0])),   int(divUp(gpu_width,threads_per_block[1]))     #i.e) grid dim  
        
        barycentric_interpolation_cuda[blocks_per_grid,threads_per_block](texture_float,triArr,rgb,ytop,xleft,p)

    return texture_float



def compute_Local_coordinate(Surface_path , one_ring_vtxID_dict = None ,init_flg = False , fixed_neightbor_vtx_list = None):
    ps_vtx , ps_nml , ps_rgb , ps_face , ps_vtx_num , ps_face_num = load_ply(Surface_path)
    f_list = []
    re_compute_list = []
    if init_flg :   #for f2
        fixed_neightbor_vtx_list = np.zeros(ps_vtx_num,dtype = "int32")
    for i in tqdm.tqdm(range(ps_vtx_num)):
        p0 = np.array(ps_vtx[i])      #origin of Local coordinate system
        #neighbor_facelist = one_ring_vtxID_dict[i]
        ###f1 : weighted average of normal vector of incident triangles
        sum_nml = np.array([0.0,0.0,0.0])    
        if init_flg :   #for f2
            min_x = 100000
        """
        for vtx_list in neighbor_facelist:
            #calculate normal and weight of Triangle(p0,p1,p2)
            p1 = np.array(ps_vtx[vtx_list[0]])
            p2 = np.array(ps_vtx[vtx_list[1]])
            nml = np.cross(p1-p0 , p2-p0) 
            if init_flg:    #for f2 , fix vtx with min_x
                if (p1[0] < min_x) :
                    min_x = p1[0]
                    fixed_neightbor_vtx_list[i] = vtx_list[0]
                if (p2[0] < min_x) :
                    min_x = p2[0]
                    fixed_neightbor_vtx_list[i] = vtx_list[1]
                        
            #S = norm(nml) / 2
            #nml = nml / norm(nml)
            #sum_nml += nml * S
            sum_nml += nml
        f1 = sum_nml / norm(sum_nml)
        """
        # f1 from computed normal
        f1 = ps_nml[i]      

        ###f2 from (1,0,0)
        #print(ps_vtx[fixed_neightbor_vtx_list[i]])
        #p_fix = np.array(ps_vtx[fixed_neightbor_vtx_list[i]])  # get f2 from neighbor vtx
        p_fix = np.array([1,0,0])                               # get f2 from fixed vtx
        if np.all(p_fix == f1) :        #Exception when p2 is not computable
            re_compute_list.append(i)
            f_list.append(np.array(f1,np.zeros(3),np.zeros(3)))
            print("p_fix == f1")
            break
            #continue
        p_nml = np.dot(np.dot(f1 , p_fix) , f1)
        p_pln = p_fix - p_nml
        f2 = p_pln / norm(p_pln)

        #f2 from grad
        """
        neighbor_facelist = one_ring_vtxID_dict[i]
        neighbor_vtxlist = set([x for row in neighbor_facelist for x in row]) 
        sum_grad = np.zeros(3)
        for vtx_id in neighbor_vtxlist:
            p = np.array(ps_vtx[vtx_id])
            dis =  p - p0
            sum_grad += dis
        grad = sum_grad / norm(sum_grad)
        p_nml = np.dot(np.dot(f1 , grad) , f1)
        p_pln = grad - p_nml
        f2 = p_pln / norm(p_pln)
        """
        
        ###f3 from cross(f1,f2)
        f3 = np.cross(f1 , f2) 
        f3 = f3 / norm(f3) 
        
        fs = np.array([f1,f2,f3])
        f_list.append(fs)


        ####debug plot#####################################################
        """
        if i == 0:
            fig = plt.figure(figsize=(8, 8)) # 図の設定
            ax = fig.add_subplot(projection='3d') # 3Dプロットの設定
            ax.set_xlabel('x') # x軸ラベル
            ax.set_ylabel('y') # y軸ラベル
            ax.set_zlabel('z') # z軸ラベル
            ax.set_title('quiver(x, y, z, u, v, w)', fontsize=20) # タイトル
            ax.legend() # 凡例
            ax.set_xlim(-2, 2) # x軸の表示範囲
            ax.set_ylim(-2, 2) # y軸の表示範囲
            ax.set_zlim(-2, 2) # z軸の表示範囲


        ###Glocal coordinate system
        ax.scatter(0, 0, 0, label='Global origin' , color = "Black") # 始点
        ax.scatter(1, 0, 0, label='x') # 終点
        ax.scatter(0, 1, 0, label='y') # 終点
        ax.scatter(0, 0, 1, label='z') # 終点
        ax.quiver(0, 0, 0, 1, 0, 0, arrow_length_ratio=0.1) # 矢印プロット
        ax.quiver(0, 0, 0, 0, 1, 0, arrow_length_ratio=0.1) # 矢印プロット
        ax.quiver(0, 0, 0, 0, 0, 1, arrow_length_ratio=0.1) # 矢印プロット

        ###Local coordinate system
        ax.scatter(p0[0], p0[1], p0[2], label='Local origin' , color = "Red") # 始点
        ax.scatter(p0[0] + f1[0], p0[1] + f1[1], p0[2] + f1[2] , label='f1') # 終点
        ax.scatter(p0[0] + f2[0], p0[1] + f2[1], p0[2] + f2[2] , label='f2') # 終点
        ax.scatter(p0[0] + f3[0], p0[1] + f3[1], p0[2] + f3[2] , label='f3') # 終点
        ax.quiver(p0[0], p0[1], p0[2], f1[0], f1[1], f1[2], arrow_length_ratio=0.1) # 矢印プロット
        ax.quiver(p0[0], p0[1], p0[2], f2[0], f2[1], f2[2], arrow_length_ratio=0.1) # 矢印プロット
        ax.quiver(p0[0], p0[1], p0[2], f3[0], f3[1], f3[2], arrow_length_ratio=0.1) # 矢印プロット
        #if i == 35000:
        plt.show()
        """
        ##################################################

    """
    if re_compute_list != []:   #Exception when p2 is not computable
        for j in re_compute_list:
            f1 = f_list[j][0]
            neighbor_facelist = one_ring_vtxID_dict[j]

            neighbor_vtxlist = []
            for vtx_list in neighbor_facelist:
                neighbor_vtxlist.append(vtx_list[0])
                neighbor_vtxlist.append(vtx_list[1])
            neighbor_vtxlist = set(neighbor_vtxlist)
            f2_sum = np.array([0.0,0.0,0.0])  
            for vtx_id in neighbor_vtxlist:
                f2_sum += f_list[vtx_id][1] 
            p_nml = np.dot(np.dot(f1 , f2_sum) , f1)
            p_pln = f2_sum - p_nml
            f2 = p_pln / norm(p_pln)

            ###f3 
            f3 = np.cross(f1 , f2) 
            f3 = f3 / norm(f3)
            
            fs = np.array([f1,f2,f3])
            f_list.append(fs)
    """
    f_list = np.array(f_list)
    
    #Debug as ply
    f1_rgb = []
    f2_rgb = []
    f2_r = []
    f2_g = []
    f2_b = []
    f3_rgb = []
    for fs in  f_list :
        f1_rgb.append([int(((fs[0][0]+1)/2)*255),int(((fs[0][1]+1)/2)*255),int(((fs[0][2]+1)/2)*255)])
        f2_rgb.append([int(((fs[1][0]+1)/2)*255),int(((fs[1][1]+1)/2)*255),int(((fs[1][2]+1)/2)*255)])
        f2_r.append([int(((fs[1][0]+1)/2)*255),int(((fs[1][0]+1)/2)*255),int(((fs[1][0]+1)/2)*255)])
        f2_g.append([int(((fs[1][1]+1)/2)*255),int(((fs[1][1]+1)/2)*255),int(((fs[1][1]+1)/2)*255)])
        f2_b.append([int(((fs[1][2]+1)/2)*255),int(((fs[1][2]+1)/2)*255),int(((fs[1][2]+1)/2)*255)])        
        f3_rgb.append([int(((fs[2][0]+1)/2)*255),int(((fs[2][1]+1)/2)*255),int(((fs[2][2]+1)/2)*255)])
    save_ply("f1.ply",ps_vtx , ps_nml ,f1_rgb , ps_face , ps_vtx_num , ps_face_num)
    save_ply("f2.ply",ps_vtx , ps_nml ,f2_rgb , ps_face , ps_vtx_num , ps_face_num)
    save_ply("f2_x.ply",ps_vtx , ps_nml ,f2_r , ps_face , ps_vtx_num , ps_face_num)
    save_ply("f2_y.ply",ps_vtx , ps_nml ,f2_g , ps_face , ps_vtx_num , ps_face_num)
    save_ply("f2_z.ply",ps_vtx , ps_nml ,f2_b , ps_face , ps_vtx_num , ps_face_num)
    save_ply("f3.ply",ps_vtx , ps_nml ,f3_rgb , ps_face , ps_vtx_num , ps_face_num)
    
    if init_flg :  
        return f_list, fixed_neightbor_vtx_list
    else:
        return f_list
    
def Local_coordinate_from_global_coordinate(Surface_path):
    ps_vtx , ps_nml , ps_rgb , ps_face , ps_vtx_num , ps_face_num = load_ply(Surface_path)
    f_list = []
    for i in tqdm.tqdm(range(ps_vtx_num)):
        f1 = np.array([1,0,0])
        f2 = np.array([0,1,0])
        f3 = np.array([0,0,1])
        fs = np.array([f1,f2,f3])
        f_list.append(fs)
    return f_list

@cuda.jit
def my_dot3_3and3(res,a,b):
    res[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2]
    res[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2]
    res[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2]

@cuda.jit
def calc_disp_on_LocalCoordinateSystem_gpu(out_disp , ft_vtx , ps_vtx , basis , vtx_num):
    i = cuda.grid(1)
    if i < vtx_num:
        fs = basis[i]
        if ft_vtx[i][0] == 0 and ft_vtx[i][1] == 0 and ft_vtx[i][2] == 0:
            out_disp[i][0] = 0.0
            out_disp[i][1] = 0.0
            out_disp[i][2] = 0.0
        else:
            ft_local = cuda.local.array((3), np.float32)
            ps_local = cuda.local.array((3), np.float32)
            my_dot3_3and3(ft_local ,fs , ft_vtx[i])         #Global → Local
            my_dot3_3and3(ps_local ,fs , ps_vtx[i])         #Global → Local
            out_disp[i][0] = ft_local[0] - ps_local[0]
            out_disp[i][1] = ft_local[1] - ps_local[1]
            out_disp[i][2] = ft_local[2] - ps_local[2]
            #out_disp[i] = np.dot(fs, ft_vtx[i]) - np.dot(fs, ps_vtx[i])     

def calc_disp_on_LocalCoordinateSystem_caller(ft_vtx ,ps_vtx ,basis ,vtx_num):
    threads_per_block                   = 64  #i.e) block dim
    blocks_per_grid                     = int(divUp(vtx_num,threads_per_block))     #i.e) grid dim  

    out_disp = np.zeros((vtx_num , 3) , np.float32)
    calc_disp_on_LocalCoordinateSystem_gpu[blocks_per_grid,threads_per_block](out_disp , ft_vtx , ps_vtx , basis , vtx_num)
    return out_disp

def calc_disp_on_LocalCoordinateSystem_cpu(ft_vtx ,ps_vtx ,basis ,vtx_num):
    disp = []
    for i in tqdm.tqdm(range(vtx_num)):
        fs = basis[i]
        if ft_vtx[i] == [0,0,0]:
            dis = np.array([0.0,0.0,0.0])
        else:
            dis = np.dot(fs, ft_vtx[i]) - np.dot(fs, ps_vtx[i])     
        disp.append(dis)
    return disp

def Calc_displacement_on_LocalCoordinateSystem(ps_vtx, ft_vtx , ft_rgb , vtx_num, basis , data_cnt = 0 , ColorTexture_from_raw_flg=False , uv_raw_mapping_path = None ,  ):
    """data_cnt : +1 = disp   , +2 = color """

    disp = []
    color = []
    uv_raw = []
    print("calculate displacement")
    #calculate displacement for every vertex
    if data_cnt == 2 or data_cnt == 3 and ColorTexture_from_raw_flg:
        uv_raw_mapping = load_uv_raw_mapping(uv_raw_mapping_path)

    if data_cnt == 1 or data_cnt == 3 :
        disp = calc_disp_on_LocalCoordinateSystem_caller(ft_vtx , ps_vtx , basis , vtx_num)
        #disp = calc_disp_on_LocalCoordinateSystem_cpu(ft_vtx , ps_vtx , basis , vtx_num)

    if data_cnt == 2 or data_cnt == 3 :         #TODO:calc_disp_callerを参考にcudaコードに変更する
        for i in tqdm.tqdm(range(vtx_num)):
                if ColorTexture_from_raw_flg == False:
                    color_x = round(ft_rgb[i][0] / 255.0 ,6)
                    color_y = round(ft_rgb[i][1] / 255.0 ,6)
                    color_z = round(ft_rgb[i][2] / 255.0 ,6)
                    color.append([color_x,color_y,color_z])
                else:
                    u_raw = uv_raw_mapping[i][0]
                    v_raw = uv_raw_mapping[i][1]
                    uv_raw.append([u_raw,v_raw])

    print("Finish calculate displacement")
    disp_avg = None
    if data_cnt == 1 or data_cnt == 3 :
        disp = np.array(disp)
        disp_avg = np.average(disp,axis=0)

    if data_cnt == 2 or data_cnt == 3 :
        if ColorTexture_from_raw_flg == False:
            color = np.array(color)
        else:
            uv_raw = np.array(uv_raw , dtype= np.float32)

    return (disp , color , disp_avg , uv_raw)     

@cuda.jit
def calc_disp_gpu(out_disp , ft_vtx , ps_vtx , vtx_num):
    i = cuda.grid(1)
    if i < vtx_num:
        if ft_vtx[i][0] == 0 and ft_vtx[i][1] == 0 and ft_vtx[i][2] == 0:
            out_disp[i][0] = 0.0
            out_disp[i][1] = 0.0
            out_disp[i][2] = 0.0
        else:
            out_disp[i][0] = ft_vtx[i][0] - ps_vtx[i][0]
            out_disp[i][1] = ft_vtx[i][1] - ps_vtx[i][1]
            out_disp[i][2] = ft_vtx[i][2] - ps_vtx[i][2]

def calc_disp_caller(ft_vtx ,ps_vtx ,vtx_num):
    threads_per_block                   = 64  #i.e) block dim
    blocks_per_grid                     = int(divUp(vtx_num,threads_per_block))     #i.e) grid dim  

    out_disp = np.zeros((vtx_num , 3) , np.float32)
    calc_disp_gpu[blocks_per_grid,threads_per_block](out_disp , ft_vtx , ps_vtx , vtx_num)
    return out_disp


def calc_disp_cpu(ft_vtx ,ps_vtx ,vtx_num):
    disp = []
    for i in tqdm.tqdm(range(vtx_num)):
        if ft_vtx[i] == [0,0,0]:
            dis = np.array([0.0,0.0,0.0])
        else:
            dis = ft_vtx[i] - ps_vtx[i] 
        disp.append(dis)
    return disp

def Calc_displacement(ps_vtx, ft_vtx, ft_rgb , vtx_num, data_cnt = 0 ):
    """data_cnt : +1 = disp   , +2 = color """

    disp = []
    color = []
    print("calculate displacement")
    #calculate displacement for every vertex
    if data_cnt == 1 or data_cnt == 3 :
        #disp = calc_disp_cpu(ft_vtx , ps_vtx , vtx_num)
        disp = calc_disp_caller(ft_vtx , ps_vtx , vtx_num)

    if data_cnt == 2 or data_cnt == 3 :
        for i in tqdm.tqdm(range(vtx_num)):     #TODO:calc_disp_callerを参考にcudaコードに変更する
                color_x = round(ft_rgb[i][0] / 255.0 ,6)
                color_y = round(ft_rgb[i][1] / 255.0 ,6)
                color_z = round(ft_rgb[i][2] / 255.0 ,6)
                color.append([color_x,color_y,color_z])

    disp_avg = None
    if data_cnt == 1 or data_cnt == 3 :
        disp = np.array(disp)
        disp_avg = np.average(disp,axis=0)

    if data_cnt == 2 or data_cnt == 3 :
        color = np.array(color)
    return (disp , color , disp_avg)   

def UV_expand_assignment(uvskin_path,save_path,height,width):  
    uv_vtx , uv_nml , uv_txr_float , faceID2vtxIDset , uv_faceID2uvIDset , uv_faceID2normalIDset , uv_vtx_num , uv_face_num = load_obj(uvskin_path)
    vtxID2uvIDs = get_vtxID2uvIDs(faceID2vtxIDset , uv_faceID2uvIDset)
    uv_txr = deepcopy(uv_txr_float)
    assing_uv_texture = np.full((height,width,1), -1, dtype=np.int8)
    
    height = height - 1
    width = width - 1

    ### float uv → int uv ########################################
    for vtxID in range(uv_vtx_num):
        uvID_list = vtxID2uvIDs[vtxID]
        for uvID in uvID_list:
            uv_cod = uv_txr_float[uvID]
            new_uv_cod = [round(uv_cod[0]*width),round(uv_cod[1]*height)]       #[x,y]
            uv_txr[uvID] = new_uv_cod
            
    save_obj("aaa.obj" , uv_vtx , uv_nml , uv_txr , faceID2vtxIDset , uv_faceID2uvIDset , uv_faceID2normalIDset , uv_vtx_num , uv_face_num)
    insert_flg = False

    #insert_list = []
    for vtxID in tqdm.tqdm(range(uv_vtx_num)):
        uvID_list = vtxID2uvIDs[vtxID]
        for uvID in uvID_list:
            uv_cod = deepcopy(uv_txr[uvID])
            for vtxID_in in range(vtxID+1,uv_vtx_num):
                uvID_in_list = vtxID2uvIDs[vtxID_in]
                debug_cnt = 0
                if insert_flg:
                    break
                for uvID_in in uvID_in_list:
                    uv_cod_in = deepcopy(uv_txr[uvID_in])
                    if np.all(np.array(uv_cod_in) == np.array(uv_cod)):
                        print()
                        print()
                        print("vtxID : ",vtxID)
                        print("uv_cod : ",uv_cod)
                        print("uv_cod_in : ",uv_cod_in)
                        debug_cnt += 1
                        if debug_cnt == 2:
                            print(debug_cnt)
                            sys.exit()
                        dir = np.array(uv_txr_float[uvID]) - np.array(uv_txr_float[uvID_in])
                        if  dir[0] > 0:
                            dir[0] = 1
                        else :
                            dir[0] = -1

                        if dir[1] > 0:
                            dir[1] = 1
                        else :
                            dir[1] = -1

                        insert_flg = True
                        for uvID_mv , uv_mv in enumerate(uv_txr):
                            if np.all(uv_mv == uv_cod + dir) :
                                insert_flg = False
                                break
                        
                        if insert_flg:
                            uv_txr[uvID][0] = uv_cod[0] + dir[0]
                            uv_txr[uvID][1] = uv_cod[1] + dir[1]
                            break
                        else:
                            if dir[0] == -1:
                                dir[0] = 0
                            if dir[1] == -1:
                                dir[1] = 0

                        print("uv_cod[0] + dir[0] : " , uv_cod[0] + dir[0])
                        print("uv_cod[1] + dir[1] : " , uv_cod[1] + dir[1])
                        for uvID_mv , uv_mv in enumerate(uv_txr):
                            #about x
                            if uv_mv[0] >= uv_cod[0] + dir[0]:
                                uv_txr[uvID_mv][0] += 1
                            
                            #about y
                            if uv_mv[1] >= uv_cod[1] + dir[1]:
                                uv_txr[uvID_mv][1] += 1

                        print("uv_cod[0] + dir[0] : " , uv_cod[0] + dir[0])
                        print("uv_cod[1] + dir[1] : " , uv_cod[1] + dir[1])

                        if  dir[0] == 0:
                            uv_txr[uvID][0] = uv_cod[0] + dir[0]
                        else:
                            uv_txr[uvID][0] = uv_cod[0] + dir[0] 

                        if  dir[1] == 0:
                            uv_txr[uvID][1] = uv_cod[1] + dir[1]
                        else:
                            uv_txr[uvID][1] = uv_cod[1] + dir[1] 
                             
                        print("dir : ",dir)
                        print("uv_txr[uvID_in] : ",uv_txr[uvID])
                        #insert_list.append(uv_txr[uvID_in])
                        #sys.exit()
                        insert_flg = True
                        break

            #debug
            if insert_flg or vtxID % 200 == 0:
                ###debug from here 
                print(vtxID, " / " ,  uv_vtx_num)
                debug_uv_txr = np.array(uv_txr)
                debug_uv_txr = debug_uv_txr.astype(np.int16)
                print(np.max(debug_uv_txr[:,0]) + 1,",",np.max(debug_uv_txr[:,1]) + 1)
                debug_texture = np.zeros((np.max(debug_uv_txr[:,1]) + 1,np.max(debug_uv_txr[:,0]) + 1,1))
                for uvID_mv , uv_mv in enumerate(uv_txr):
                    debug_texture[int(uv_mv[1]),int(uv_mv[0])] += 1
                print("sum : " , np.sum(debug_texture))
                print("duplicate sum : " , np.sum(np.where(debug_texture > 1 , 1 , 0)))
                debug_texture_dupulicate = np.where(debug_texture > 1 , 255 , 0)
                debug_texture = np.where(debug_texture > 0 , 50 , 0)
                print(uv_txr[uvID][0],uv_txr[uvID][1])
                debug_texture[int(uv_txr[uvID][1]),int(uv_txr[uvID][0])] = 255

                cv2.imwrite(os.path.join(save_path , "debug" , str(vtxID) + ".png") , debug_texture)
                cv2.imwrite(os.path.join(save_path , "debug" , "depulicate_" + str(vtxID) + ".png") , debug_texture_dupulicate)
                ###debug by here 

                save_obj(os.path.join(save_path , "Shellb_expandUV.obj") , uv_vtx , uv_nml , uv_txr , faceID2vtxIDset , uv_faceID2uvIDset , uv_faceID2normalIDset , uv_vtx_num , uv_face_num)
                insert_flg = False
                #sys.exit()
    sys.exit()
        
    
    """
    for faceid  in tqdm.tqdm(range(uv_face_num)):
        #uv = [0,0,0]
        #vtx_id = [0,0,0]
        for j in range(3):
            uv = uv_txr[uv_faceID2uvIDset[faceid][j]]   #uv[j] : uv coordinate of j th vtx in face
            vtx_id = uv_face[faceid][j]

            assing_uv_texture[round(uv[0]*width),round(uv[1]*height)] = vtx_id
            debug_assing_uv_texture[round(uv[0]*width),round(uv[1]*height)] += 1
    #debug_assing_uv_texture = np.where(debug_assing_uv_texture == 1 , 1 , 0)
    debug_assing_uv_texture = np.where(debug_assing_uv_texture > 2 , 1 , 0)
    print(np.min(debug_assing_uv_texture))
    cv2.imwrite("debug.png",(debug_assing_uv_texture - np.min(debug_assing_uv_texture))/(np.max(debug_assing_uv_texture) - np.min(debug_assing_uv_texture))*255)
    sys.exit()
    """


def GetFaceIDandbaryTexture(uv_txr,uv_faceID2uvIDset,uv_face_num,height,width):
    """
    FaceIDandBary =  []
    for y in range(height):
        inner_list = []
        for x in range(width):
            inner_list.append(0)
        FaceIDandBary.append(inner_list)
    """

    uint_size = 4294967295
    if uv_face_num > uint_size:
        print("uv_face_num: " , uv_face_num)
        print("uint32 size is 0~4294967295 and your facenum is over 4294967295")
        print("please change FaceIDTexture's dtype to uint64")
        sys.exit()

    FaceIDTexture = np.full((height,width), uint_size , dtype=np.uint32)
    BaryTexture   = np.zeros((height,width,3), dtype=np.float32)
    
    height = height - 1
    width = width - 1
    for faceid  in tqdm.tqdm(range(uv_face_num)):
        #start_a = time.time()
        uv = [0,0,0]

        for j in range(3):
            uv[j] = uv_txr[uv_faceID2uvIDset[faceid][j]]   #uv[j] : uv coordinate of j th vtx in face 
        #print(rgb)
         # Make array of vertices
        # ax bx cx
        # ay by cy
        #  1  1  1
        ###Assign each point of the face to nearest pixel
        triArr = np.asarray([uv[0][0],uv[1][0],uv[2][0], uv[0][1],uv[1][1],uv[2][1], 1,1,1]).reshape((3, 3))
        # Get bounding box of the triangle
        xleft   = int(min(uv[0][0], uv[1][0], uv[2][0]))
        xright  = int(max(uv[0][0], uv[1][0], uv[2][0]))
        ytop    = int(min(uv[0][1], uv[1][1], uv[2][1]))
        ybottom = int(max(uv[0][1], uv[1][1], uv[2][1]))

        ###No assign
        """
        triArr = np.asarray([uv[0][0]*width,uv[1][0]*width,uv[2][0]*width, uv[0][1]*height,uv[1][1]*height,uv[2][1]*height, 1,1,1]).reshape((3, 3))
        # Get bounding box of the triangle
        xleft   = math.ceil(min(uv[0][0]*width , uv[1][0]*width , uv[2][0]*width))
        xright  = math.floor (max(uv[0][0]*width , uv[1][0]*width , uv[2][0]*width))
        ytop    = math.ceil(min(uv[0][1]*height, uv[1][1]*height, uv[2][1]*height))
        ybottom = math.floor (max(uv[0][1]*height, uv[1][1]*height, uv[2][1]*height))
        """

        #time_a = time.time() - start_a
        #print("a:",time_a)            
        #start_b = time.time()

        # loop over each pixel, compute barycentric coordinates and interpolate vertex colors
        for y in range(ytop, ybottom+1):
            for x in range(xleft, xright+1):
                # Store the current point as a matrix
                p = np.array([[x], [y], [1]])

                # Solve for least squares solution to get barycentric coordinates
                (alpha, beta, gamma) = np.linalg.lstsq(triArr, p, rcond=-1)[0]

                # The point is inside the triangle if all the following conditions are met; otherwise outside the triangle
                #if alpha > 0 and beta > 0 and gamma > 0 :
                #if alpha > -1e-10 and beta > -1e-10 and gamma > -1e-10 :
                if alpha > -1e-12 and beta > -1e-12 and gamma > -1e-12 :
                    #FaceIDandBary[y][x] = [faceid,[alpha,beta,gamma]]
                    FaceIDTexture[y][x] = faceid        #id of face which is interpolate this pixel(y,x)
                    BaryTexture[y][x][0] = alpha[0]     #Interpolate weight of vtx1 which is belong to face
                    BaryTexture[y][x][1] = beta[0]      #Interpolate weight of vtx2 which is belong to face
                    BaryTexture[y][x][2] = gamma[0]     #Interpolate weight of vtx3 which is belong to face
    return FaceIDTexture , BaryTexture      

def Find_one_ring_vtx(uvskin_path):
    uv_vtx , uv_nml , uv_txr , uv_face , uv_faceID2uvIDset , uv_faceID2normalIDset , uv_vtx_num , uv_face_num = load_obj(uvskin_path)

    #map generation is needed only first time
    one_ring_vtxID_dict = {}
    #Initialize one_ring_vtxID_dict
    for i in range(uv_vtx_num):
        one_ring_vtxID_dict[i] = []

    # Store neighbor vtx in same face
    for i, fc in enumerate(uv_face):
        vtx_id0 = fc[0]
        vtx_id1 = fc[1]
        vtx_id2 = fc[2]

        #store id0's vtx neightbor id1 & id2 in same face
        one_ring_vtxID_dict[vtx_id0].append([vtx_id1,vtx_id2])

        #store id1's vtx neightbor id0 & id2 in same face
        one_ring_vtxID_dict[vtx_id1].append([vtx_id0,vtx_id2])

        #store id2's vtx neightbor id0 & id1 in same face
        one_ring_vtxID_dict[vtx_id2].append([vtx_id0,vtx_id1])

    #delite depulicate & sort
    for i in range(uv_vtx_num):
        one_ring_vtxID_dict[i] = sorted(one_ring_vtxID_dict[i])
    
    with open('one_ring_vtxIDfacelist_dict.json', 'w') as fp:
        json.dump(one_ring_vtxID_dict, fp)
    
    return one_ring_vtxID_dict

def calc_local_coord_f1_as_nml(vtxs , vtx_num , faces , face_num , edge_oneling_neighbor_vtxID_dict_first = None):
    #edge_list = list(map(int,edge_oneling_neighbor_vtxID_dict_first.keys()))
    vtx_nmls = np.zeros((vtx_num,3) , np.float32)
    debug_nmls = np.zeros((vtx_num,3) , np.uint8)
    vtxs = np.array(vtxs)
    #compute Face normal and add to vtx_nml
    for face_id in tqdm.tqdm(range(face_num)):
        face = faces[face_id]
        #if set(edge_list).isdisjoint(set(face)): #edge_listにfaceを構成する点のいずれも含まれない。
        v0 = vtxs[face[0]]
        v1 = vtxs[face[1]]
        v2 = vtxs[face[2]]

        face_nml = np.cross(v1-v0 , v2-v0)  
        #face_nml = face_nml * np.abs(face_nml)  #with area weight

        #calculate sum of faces normal
        vtx_nmls[face[0]] += face_nml
        vtx_nmls[face[1]] += face_nml
        vtx_nmls[face[2]] += face_nml
    
    for vtx_id in range(vtx_num):
        #if np.any(vtx_nmls[vtx_id] != np.zeros(3 , np.float32)):
        vtx_nmls[vtx_id] = vtx_nmls[vtx_id] / np.linalg.norm(vtx_nmls[vtx_id], ord=2)   #normalize
    
        debug_nmls[vtx_id][0] = round(((vtx_nmls[vtx_id][0] + 1)/2) * 255)
        debug_nmls[vtx_id][1] = round(((vtx_nmls[vtx_id][1] + 1)/2) * 255)
        debug_nmls[vtx_id][2] = round(((vtx_nmls[vtx_id][2] + 1)/2) * 255)

        if np.linalg.norm(vtx_nmls[vtx_id], ord=2) < 0.999999 or np.linalg.norm(vtx_nmls[vtx_id], ord=2) > 1.000001:
            print("f1 is not 1.0 error : " , vtx_id)
            print(np.linalg.norm(vtx_nmls[vtx_id], ord=2))
            sys.exit()

    #return vtx_nmls     # = local_coordinate_f1 
    return vtx_nmls , debug_nmls

def define_local_coord_f2ID_from_Xway(vtxs,vtx_num,one_ring_vtxID_dict , save_dir_path , mode = 0):
    """For define second local coordinate(=f2), initialize used vtxID\n
       There are mode("near_vtx") and mode("near_line")\n

       mode 0 : ("near_vtx")  : find vtx which is closest to vector(1,0,0) and previous vtx by using inner_product. Return the closest ID.\n
       mode 1 : ("near_line") : find line that intersects vector(1,0,0). Return the 2 vtxID at both ends and the end-to-end ratio.
       """

    base_vector = np.array([1,0,0])
    vtxs = np.array(vtxs)

    f2_ID_list = []
    max_vec_to_neighbor_list = []
    if mode == 0:                       #mode : near_vtx
        for vtx_i in range(vtx_num):
            vtx = vtxs[vtx_i]
            neighbor_face_set = one_ring_vtxID_dict[str(vtx_i)]

            #from neighbor 1 vtx
            neighbor = list(set(list(itertools.chain.from_iterable(neighbor_face_set)))) #flatten & del duplicate
            max = -100
            max_nei_i = -1
            for nei_i in neighbor:
                vtx_nei = vtxs[nei_i]
                vec_to_neighbor = vtx_nei - vtx
                vec_to_neighbor = vec_to_neighbor / np.linalg.norm(vec_to_neighbor)
                inner_product_with_base_vector = vec_to_neighbor @ base_vector

                inner_product_with_nei_j = 0.0
                calclated_nei_j_cnt = 0
                for nei_j in neighbor:
                    if nei_j < vtx_i:
                        nei_j_vector = max_vec_to_neighbor_list[nei_j]
                        nei_j_vector = nei_j_vector / np.linalg.norm(nei_j_vector)
                        inner_product_with_nei_j += vec_to_neighbor @ nei_j_vector
                        calclated_nei_j_cnt += 1
                if calclated_nei_j_cnt != 0:
                    inner_product_with_nei_j = inner_product_with_nei_j / calclated_nei_j_cnt

                sum_inner_product = 1.0 * inner_product_with_base_vector + 0.0 * inner_product_with_nei_j

                if max < sum_inner_product:
                    max = sum_inner_product
                    max_nei_i = nei_i
                    max_vec_to_neighbor = vec_to_neighbor
            f2_ID_list.append(max_nei_i)
            max_vec_to_neighbor_list.append(max_vec_to_neighbor)
        save_path = os.path.join(save_dir_path , "local_coord_f2ID_list.txt")
        f = open(save_path, 'wb')
        pickle.dump(f2_ID_list, f)

    elif mode == 1:                         #mode : near_line
        nei_i_set_and_ratio_v1_v2_list = []
        for vtx_i in range(vtx_num):
            vtx = vtxs[vtx_i]
            neighbor_face_set = one_ring_vtxID_dict[str(vtx_i)]
            debug_valid_line_cnt = 0
            max_inner_product = -100
            for j , nei_i_set in enumerate(neighbor_face_set):
                v1 = vtxs[nei_i_set[0]]
                v2 = vtxs[nei_i_set[1]]
                face_nml = np.cross(v1-vtx , v2-vtx)  
                face_nml = face_nml / np.linalg.norm(face_nml)  #normalize

                base_vector_proj = base_vector - (face_nml @ base_vector) * face_nml   #projection to f1_plane
                base_vector_proj = base_vector_proj / np.linalg.norm(base_vector_proj) #normalize

                v12_0 = v2[0] - v1[0]
                v12_1 = v2[1] - v1[1]
                v12_2 = v2[2] - v1[2]

                v12_0 = v12_0 / np.linalg.norm(v12_0)  #normalize
                v12_1 = v12_1 / np.linalg.norm(v12_1)  #normalize
                v12_2 = v12_2 / np.linalg.norm(v12_2)  #normalize

                left = [[base_vector_proj[0] , -v12_0],
                        [base_vector_proj[1] , -v12_1]]
                        #[base_vector_proj[2] , -v12_2]]
                
                right = [v1[0] - vtx[0],v1[1] - vtx[1]]     #,v1[2] - vtx[2]]
                ratio_v0_proj , ratio_v1_v2  = np.linalg.solve(left,right)

                #if ratio_v0_proj > 0.0 and ratio_v1_v2 > 0.0 :
                inner_product = base_vector_proj @ base_vector
                if max_inner_product < inner_product:
                    max_inner_product = inner_product
                    valid_nei_i_set   = nei_i_set
                    valid_ratio_v1_v2 = ratio_v1_v2

                debug_valid_line_cnt += 1
            if debug_valid_line_cnt == 0 :
                print("debug_valid_line_cnt error")
                print(vtx_i , " : ",debug_valid_line_cnt)
                sys.exit()

            nei_i_set_and_ratio_v1_v2_list.append(valid_nei_i_set + [valid_ratio_v1_v2])
            
        save_path = os.path.join(save_dir_path , "nei_i_set_and_ratio_v1_v2_list.txt")
        f = open(save_path, 'wb')
        pickle.dump(nei_i_set_and_ratio_v1_v2_list, f)

def calc_local_coord_f2_and_f3_from_f2ID_list(vtxs, vtx_num, f1_list , f2_ID_list_or_nei_i_set_and_ratio_v1_v2_list , mode = 0):
    """Define second local coordinate(=f2) and third(=f3), by using pre defined list in define_local_coord_f2ID_from_Xway()\n

       "mode" : Use same mode you used when run define_local_coord_f2ID_from_Xway()
       "f2_ID_list_or_nei_i_set_and_ratio_v1_v2_list" : mode == 0 → f2_ID_list , mode == 1 → nei_i_set_and_ratio_v1_v2_list
    """

    if mode == 0:
        f2_ID_list = f2_ID_list_or_nei_i_set_and_ratio_v1_v2_list
        assert isinstance(f2_ID_list[0] ,int) , "You are using \"nei_i_set_and_ratio_v1_v2_list\" even though you set mode to 0. Use \"local_coord_f2ID_list\"."
    elif mode == 1:
        nei_i_set_and_ratio_v1_v2_list = f2_ID_list_or_nei_i_set_and_ratio_v1_v2_list
        print(type(nei_i_set_and_ratio_v1_v2_list[0] ))
        print(isinstance(nei_i_set_and_ratio_v1_v2_list[0] ,list) )
        assert isinstance(nei_i_set_and_ratio_v1_v2_list[0] ,list) , "You are using \"local_coord_f2ID_list\" even though you set mode to 1. Use \"nei_i_set_and_ratio_v1_v2_list\"."

    vtxs = np.array(vtxs)
    f2_list = []
    f3_list = []
    for vtx_i in range(vtx_num):
        ###Calculate f2
        f1 = f1_list[vtx_i]

        if mode == 0:
            f2_i   = f2_ID_list[vtx_i]
            f2_org = vtxs[f2_i] - vtxs[vtx_i]
        elif mode == 1:
            nei_i_set_and_ratio_v1_v2 = nei_i_set_and_ratio_v1_v2_list[vtx_i]
            v1 = vtxs[nei_i_set_and_ratio_v1_v2[0]]
            v2 = vtxs[nei_i_set_and_ratio_v1_v2[1]]
            segment_v12 = v1 + (v2 - v1) * nei_i_set_and_ratio_v1_v2[2]
            f2_org = segment_v12 - vtxs[vtx_i]

        f2_proj = f2_org - (f1 @ f2_org) * f1   #projection to f1_plane

        if np.any(f2_proj != 0.0):
            f2_proj = f2_proj / np.linalg.norm(f2_proj) #normalize

        if np.linalg.norm(f2_proj, ord=2) < 0.999999 or np.linalg.norm(f2_proj, ord=2) > 1.000001:
            print("f2 is not 1.0 error : " , vtx_i)
            print(np.linalg.norm(f2_proj, ord=2))
            sys.exit()

        f2_list.append(f2_proj)

        ###Calculate f3 from f1 and f2
        f3 = np.cross(f1 , f2_proj)
        if np.any(f3 != 0.0):
            f3 = f3 / np.linalg.norm(f3)

        if np.linalg.norm(f3, ord=2) < 0.999999 or np.linalg.norm(f3, ord=2) > 1.000001:
            print("f3 is not 1.0 error : " , vtx_i)
            print(np.linalg.norm(f3, ord=2))
            sys.exit()

        f3_list.append(f3)

    f2_list = np.array(f2_list)
    f3_list = np.array(f3_list)
    return f2_list , f3_list

def calc_local_coordinate_for_2path(vtxs , vtx_num , faces , face_num , f2_ID_list_or_nei_i_set_and_ratio_v1_v2_list , mode = 0):
    """For define second local coordinate(=f2),There are mode("near_vtx") and mode("near_line")\n
       mode 0 : ("near_vtx") : find vtx which is closest to vector(1,0,0) and previous vtx by using inner_product. Return the closest ID.\n
       mode 1 : ("near_line") : find line that intersects vector(1,0,0). Return the 2 vtxID at both ends and the end-to-end ratio."""

    f1_list , _ = calc_local_coord_f1_as_nml(vtxs , vtx_num , faces , face_num)
    f2_list , f3_list = calc_local_coord_f2_and_f3_from_f2ID_list(vtxs , vtx_num , f1_list , f2_ID_list_or_nei_i_set_and_ratio_v1_v2_list  , mode)
    local_coordinate_system = np.concatenate((f1_list[:,np.newaxis,:],f2_list[:,np.newaxis,:],f3_list[:,np.newaxis,:]) , axis=1)
    return local_coordinate_system

if __name__ == "__main__":
    ### Find_one_ring_vtx
    uvskin_path = r"D:\Project\Human\AITS\avatar-in-the-shell\assets\heavy\SMPL_expandUV_512.obj"
    one_ring_vtxID_dict = Find_one_ring_vtx(uvskin_path)

    ### UV_expand_assignment
    #uvskin_path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Whole\Template-star-0.015-0.05\Basic_data\uv_smpl_coordinate512_manual\uv_smpl_neutral_skin13.obj"
    #save_path   = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Whole\Template-star-0.015-0.05\Basic_data\uv_smpl_coordinate512_manual"
    #UV_expand_assignment(uvskin_path,save_path,512,512)  #make expand texture
    

    #vtx , nml , rgb , face , vtx_num , face_num = load_ply(r"D:\Downloads\untitled.ply")
    #vtx , nml , rgb , face , vtx_num , face_num = load_ply(r"D:\Data\Human\HUAWEI\Iwamoto\data\predicted_naked_meshes\mesh_0208.ply")
    
    #with open(r"D:\Data\Human\Template-star-0.015-0.05\Basic_data\uv_coordinate0422\edge_oneling_neighbor_vtxID_dicts\edge_oneling_neighbor_vtxID_dict_first.json", 'r') as f:  #hard coording
    #    edge_oneling_neighbor_vtxID_dict_first = json.load(f)
    #nml , debug_nml = calc_local_coord_f1_as_nml(vtx , vtx_num , face , face_num , edge_oneling_neighbor_vtxID_dict_first)
    #debug_nml = debug_nml.tolist()
    #nml = nml.tolist()
    #save_ply("debug.ply", vtx , nml , debug_nml , face , vtx_num , face_num)
    #sys.exit()
    
    #vtx , nml , rgb , face , vtx_num , face_num = load_ply(r"D:\Data\Human\Template-star-0.015-0.05\shellb_template.ply")
    #with open(r"D:\Data\Human\Template-star-0.015-0.05\Basic_data\uv_coordinate0422\one_ring_vtxIDfacelist_dict.json", 'r') as f:     #one_ring_vtxIDfacelist_dict was made by make_displacementTextureTools.py/Find_one_ring_vtx()
    #    one_ring_vtxIDfacelist_dict = json.load(f)
    #save_dir_path = r"D:\Data\Human\Template-star-0.015-0.05\Basic_data\uv_coordinate0422"
    #define_local_coord_f2ID_from_Xway(vtx , vtx_num ,one_ring_vtxIDfacelist_dict , save_dir_path , mode = 1)
    #sys.exit()

    #define_local_coord_f2ID_from_Xway_and_line(vtx , vtx_num ,one_ring_vtxIDfacelist_dict , r"D:\Data\Human\Template-star-0.015-0.05\Basic_data\uv_coordinate0422\nei_i_set_and_ratio_v1_v2_list.txt")
    #sys.exit()

    
    #f = open(r"D:\Data\Human\Template-star-0.015-0.05\Basic_data\uv_coordinate0422\f2_ID_list.txt","rb")
    #nei_i_set_and_ratio_v1_v2_list = pickle.load(f)
    #f2_ID_list = pickle.load(f)

    #calc_local_coord_f2_and_f3_from_f2ID_list(vtx , vtx_num , f2_ID_list , nml)
    #calc_local_coordinate_for_2path(vtx , vtx_num , face , face_num , f2_ID_list)
    

    
    #uvskin_path = r"D:\Project\Human\Pose2Texture\make_displacement\uvskin.obj"
    #baseuv_savepath = r"D:\Project\Human\Pose2Texture\make_displacement\baseuvskin_huawei.obj"
    #uvskin_path = baseuv_savepath
    #FittedMesh_path = r"D:\Data\Human\ARTICULATED\I_jumping\FittedMesh_obj\FittedMesh_1.obj"
    #texture = make_displacementTexture(uvskin_path,FittedMesh_path,1024,1024,0.18)
    
    #FaceIDandbaryTexture = GetFaceIDandbaryTexture(uvskin_path,1024,1024)
    #FaceIDTexture , BaryTexture = GetFaceIDandbaryTexture(uvskin_path,1024,1024)
    #print(FaceIDandbaryTexture[100])

    #FittedMesh_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\FittedMesh_ply\FittedMesh_0100.ply"
    #save_disp_path = r"test_disp\testdisp.npy"
    #save_color_path = r"test_color\testcolor.npy"
    #texture_disp , texture_color = make_Textures(uvskin_path,FittedMesh_path,FaceIDTexture , BaryTexture,1024,1024,0.15,"obj","ply")
    #texture_avg_gpu.texture_avg(texture_disp,save_disp_path,"float")
    #texture_avg_gpu.texture_avg(texture_color,save_color_path,"float")
    
    #uvskin_path = r"D:\Data\Human\Template-star-0.015\uvskin.obj"
    #surface_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\FittedMesh_ply_pose\SkinnedSurface_0001.ply"
    #surface_path = r"D:\Data\Human\Template-star-0.015\TetraSurface.ply"
    

    #compute_Local_coordinate(surface_path,one_ring_vtxID_dict ,True)

    #texture_color = cv2.cvtColor(texture_color, cv2.COLOR_RGB2BGR)
    #cv2.imshow("test",texture_color)
    #cv2.waitKey(0)
    #cv2.imwrite("test.png",texture_color)

    