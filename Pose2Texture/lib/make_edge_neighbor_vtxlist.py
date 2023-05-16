from cv2 import seamlessClone
import numpy as np
import sys
from pathlib import Path
import os

def parentpath(path='.', f=0):
    return Path(path).resolve().parents[f]

sys.path.append(os.path.join(parentpath(__file__,1) , r"Mylib"))

from meshes_utils import load_obj , save_ply

import math
import tqdm
from meshes_utils import get_vtxID2uvIDs
import json
import itertools
from copy import deepcopy
import shutil
import cv2

#uv = cv2.imread("uv_texture.png")
uvskin_path = r"D:\Project\Human\AITS\avatar-in-the-shell\assets\heavy\SMPL_expandUV_512.obj"               #hardcording
save_dir_path = r"debug"                                                                                    #hardcording

mask_img = cv2.imread(os.path.join(r"D:\Project\Human\AITS\avatar-in-the-shell\assets\mask_SMPL.png"))      #hardcording

if os.path.isdir(save_dir_path) == False:
    os.mkdir(save_dir_path)
else:
    raise AssertionError("please delete " + save_dir_path + " folder manually")
    #shutil.rmtree(save_dir_path)
    os.mkdir(save_dir_path)

debug_dir_path = os.path.join(save_dir_path , "debug")
if os.path.isdir(debug_dir_path) == False:
    os.mkdir(debug_dir_path)
else:
    shutil.rmtree(debug_dir_path)
    os.mkdir(debug_dir_path)

vtxs , nmls , uvs , faceID2vtxIDset , faceID2uvIDset , faceID2normalIDset , vtx_num , face_num = load_obj(uvskin_path)
get_vtxID2uvIDs = get_vtxID2uvIDs(faceID2vtxIDset , faceID2uvIDset)
#FaceIDTexture , BaryTexture = make_displacementTexture_tools.GetFaceIDandbaryTexture(uvskin_path,height,width)  


height = width = 512                                                                                        #hardcording
#height = width = 768                                                                                       #hardcording

uint_size = 4294967295
if face_num > uint_size:
    print("face_num: " , face_num)
    print("uint32 size is 0~4294967295 and your facenum is over 4294967295")
    print("please change FaceIDTexture's dtype to uint64")
    sys.exit()

FilledTexture = np.zeros((height,width,1), dtype=np.float32)

height = height - 1
width = width - 1
for faceid in tqdm.tqdm(range(face_num)):
    #start_a = time.time()
    uv = [0,0,0]

    for j in range(3):
        uv[j] = uvs[faceID2uvIDset[faceid][j]]   #uv[j] : uv coordinate of j th vtx in face 
    #print(rgb)
        # Make array of vertices
    # ax bx cx
    # ay by cy
    #  1  1  1

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
            #if alpha > -1e-13 and beta > -1e-13 and gamma > -1e-13 :
            if alpha > 0 and beta > 0 and gamma > 0 :
                FilledTexture[y][x][0] = 1.0     #Interpolate weight of vtx1 which is belong to face

#np.save("filled_uv.npy",FilledTexture)
#cv2.imshow("uv",FilledTexture)
#cv2.imwrite("filled_uv.png",FilledTexture*255)
#cv2.waitKey(-1)


"""
uv_cod_list = []
#grid_x_list = []
#grid_y_list = []
uv_cod_clr_list = []
grid_x_clr_list = []
grid_y_clr_list = []
for i in range(vtx_num):
    uv_cod = uvs[get_vtxID2uvIDs[i]]
    #x_floor = math.floor(uv_cod[0]*width)
    #x_ceil  = x_floor + 1                   #math.ceil(uv_cod[0]*width)
    #y_floor = math.floor(uv_cod[1]*height)
    #y_ceil  = y_floor + 1                   #math.ceil(uv_cod[1]*height)
    #grid_x = np.array([x_floor,x_floor,x_ceil,x_ceil])
    #grid_y = np.array([y_floor,y_ceil,y_floor,y_ceil])
    #grid_x = np.array([x_floor,x_ceil])
    #grid_y = np.array([y_floor,y_ceil])
    uv_cod_list.append(uv_cod)
    #grid_x_list.append(grid_x)
    #grid_y_list.append(grid_y)


#grid_x_list = np.array(grid_x_list)
#grid_y_list = np.array(grid_y_list)
uv_cod_list = np.array(uv_cod_list)
"""

uv_cod_list = []
edge_vtxID_list = []
debug_color = np.zeros((vtx_num,3) , np.uint8)
connector_vtxID_list = []
for i in range(vtx_num):
    connector_flg = False
    uv_cods = []
    if len(get_vtxID2uvIDs[i]) > 1 :
        edge_vtxID_list.append(i)
        #debug_color[i] = np.array([255,0,0])
    for j in range(len(get_vtxID2uvIDs[i])):
        uv_cod = uvs[get_vtxID2uvIDs[i][j]]
        #if nonzero_id_mu[int(uv_cod[1]),int(uv_cod[0])] == 1:   #does its points has valid value?   (used by old version)
        if True:
            uv_cods.append(uv_cod)
        else:
            uv_cods.append([0,0])
        if mask_img[int(uv_cod[1]) , int(uv_cod[0])][0] == 30:     #Other parts → this mean that this point is shared by other parts        
            connector_flg = True
    if connector_flg and len(get_vtxID2uvIDs[i]) > 1:
        connector_vtxID_list.append(i)
        debug_color[i] = np.array([255,0,0])
    
    for k in range(j+1,4):
        uv_cods.append([0,0])
    uv_cods = np.array(uv_cods)
    uv_cods = uv_cods.astype(np.int16)
    
    uv_cod_list.append(uv_cods)

uv_cod_list = np.array(uv_cod_list)

# find edge
"""
valid_uv_num_list = []
edge_vtxID_list = []
debug_color = []
for i in range(vtx_num):
    #grid_x = grid_x_list[i]
    #grid_y = grid_y_list[i]
    uv_cod = uv_cod_list[i]
    #grid_uv = np.array([FilledTexture[grid_y[0],grid_x[0]],FilledTexture[grid_y[1],grid_x[0]],FilledTexture[grid_y[0],grid_x[1]],FilledTexture[grid_y[1],grid_x[1]]])
    
    #valid_uv_list = []
    #valid_uv = np.any(grid_uv,1)
    #valid_uv = np.nonzero(np.any(grid_uv,1))
    #valid_uv_list.append(valid_uv.tolist())
    #valid_uv_num = len(valid_uv) 

    #valid_uv_num = np.count_nonzero(valid_uv) 
    #valid_uv_num_list.append(valid_uv_num)

    vtx_value = np.zeros(3)

    if valid_uv_num != 4 :
        edge_vtxID_list.append(i)
        debug_color.append([255,0,0])
    else:
        debug_color.append([0,0,0])
"""

debug_color = debug_color.tolist()
save_ply("edge_debug.ply",vtxs , None , debug_color , faceID2vtxIDset , vtx_num , face_num)

with open(r"D:\Project\Human\AITS\avatar-in-the-shell\assets\heavy\one_ring_vtxIDfacelist_dict_SMPL.json", 'r') as f:     #hardcording
    one_ring_vtxIDfacelist_dict = json.load(f)


# edge_vtxID_list : エッジのリスト
# connector_vtxID_list  : 他の関節との境目になってるエッジのリスト
# edge_oneling_neighbor_vtxID_dict : 各エッジについて、隣り合う点をまとめたリスト 

used_vtxID_list = deepcopy(connector_vtxID_list)
present_ringID_list = deepcopy(connector_vtxID_list)

edge_oneling_neighbor_vtxID_dicts = {}
present_ringID_lists              = {}
present_ringID_lists[0]           = present_ringID_list
debug_save_path = os.path.join(debug_dir_path , "edge_and_nbr_debug" + str(0)+ ".ply")
save_ply(debug_save_path ,vtxs , None , debug_color , faceID2vtxIDset , vtx_num , face_num)

cnt = 1
while(1):
    #one_ring_vtxIDfacelist_dict : any point → neighbor points
    edge_oneling_neighbor_vtxID_dict ={}
    #現在のリングについて、隣り合う点のリストを作る(なお、重複は削除する)
    for vtxID in present_ringID_list:   
        neighbor_list = list(set(list(itertools.chain.from_iterable(one_ring_vtxIDfacelist_dict[str(vtxID)]))))
        edge_oneling_neighbor_vtxID_dict[vtxID] = neighbor_list 


    value_zero_keylist = []
    edge_oneling_neighbor_vtxID_dict2 = deepcopy(edge_oneling_neighbor_vtxID_dict)  #for文にずれが生じないように(removeがループ内にあるので)
    for k in edge_oneling_neighbor_vtxID_dict:  #リングの各点についてのループ
        #edge_oneling_neighbor_vtxID_dict[k] = list(set(edge_oneling_neighbor_vtxID_dict[k]))
        for nb in edge_oneling_neighbor_vtxID_dict[k]:  #リングの各点の隣り合う点についてのループ
            if nb in used_vtxID_list:   #隣り合う点の中に、エッジが含まれていたら、隣り合う点のリストから削除
                edge_oneling_neighbor_vtxID_dict2[k].remove(nb)
        if len(edge_oneling_neighbor_vtxID_dict2[k]) == 0:
            value_zero_keylist.append(k)
    if cnt == 1:
        edge_oneling_neighbor_vtxID_dict_first = deepcopy(edge_oneling_neighbor_vtxID_dict2)
    #past_vtxID_list = set(list(edge_oneling_neighbor_vtxID_dict2.keys()))
    used_vtxID_list = list(set(used_vtxID_list + present_ringID_list))
    present_ringID_list = list(set(list(itertools.chain.from_iterable(edge_oneling_neighbor_vtxID_dict2.values()))))   #次のリングのリスト
    present_ringID_lists[cnt] = sorted(present_ringID_list)
    edge_oneling_neighbor_vtxID_dicts[cnt] = edge_oneling_neighbor_vtxID_dict2

    #debug
    for j in present_ringID_list:
        debug_color[j] = [0,255,0]
    debug_save_path = os.path.join(debug_dir_path , "edge_and_nbr_debug" + str(cnt)+ ".ply")
    save_ply(debug_save_path ,vtxs , None , debug_color , faceID2vtxIDset , vtx_num , face_num)

    cnt += 1
    print(cnt , " : " , len(present_ringID_list))
    if len(present_ringID_list) == 0:
        break

print("value_zero_keylist")
print(value_zero_keylist)

save_path= os.path.join(save_dir_path , "ring_stairs.json")
with open(save_path,"w") as f:
    json.dump(present_ringID_lists , f)

save_path= os.path.join(save_dir_path , "edge_oneling_neighbor_vtxID_dicts.json")
with open(save_path,"w") as f:
    json.dump(edge_oneling_neighbor_vtxID_dicts , f)
"""
for v in value_zero_keylist:                    #隣り合うのがエッジのみなエッジのリスト
    tmp_list = []
    #print(edge_oneling_neighbor_vtxID_dict[v])
    for k in edge_oneling_neighbor_vtxID_dict[v]:  #隣り合うエッジの隣り合うリストについてループを回す
        #print("debug : " , edge_oneling_neighbor_vtxID_dict[k])
        for nb2 in edge_oneling_neighbor_vtxID_dict[k]:
            if (nb2 in edge_vtxID_list) == False:   #その点がエッジでないなら,
                tmp_list.append(nb2)    #隣り合うのがエッジのみなエッジのリストに追加
        tmp_list = list(set(tmp_list))  #重複削除
    if len(tmp_list) == 0:
        print("aaa")
        #sys.exit()
    print("tmp_list : ",tmp_list)
    edge_oneling_neighbor_vtxID_dict2[v] = tmp_list

value_zero_keylist = []
for k in edge_oneling_neighbor_vtxID_dict2:
    if len(edge_oneling_neighbor_vtxID_dict2[k]) == 0:
        value_zero_keylist.append(k)

print("value_zero_keylist")
print(value_zero_keylist)
"""

save_path= os.path.join(save_dir_path , r"edge_oneling_neighbor_vtxID_dict_first.json")
with open(save_path,"w") as f:
    json.dump(edge_oneling_neighbor_vtxID_dict_first , f)

"""
for edge_oneling_neighbor_vtxID_dict in edge_oneling_neighbor_vtxID_dicts:
    debug_nb_list = set(list(itertools.chain.from_iterable(edge_oneling_neighbor_vtxID_dict.values())))
    for j in debug_nb_list:
        debug_color[j] = [0,40*i,0]

for j in value_zero_keylist:
    print("j : ", j)
    debug_color[j] = [255,255,255]

save_ply("edge_and_nbr_debug.ply",vtxs , None , debug_color , faceID2vtxIDset , vtx_num , face_num)
"""
