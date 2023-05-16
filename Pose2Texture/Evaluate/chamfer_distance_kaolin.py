#from pytorch3d.utils import ico_sphere
import torch
import kaolin

from natsort import natsorted 
import argparse
import tqdm

import os
import glob
import sys
import numpy as np
import time 

import matplotlib
import matplotlib.pyplot as plt
import pymeshlab

import trimesh
import shutil

sys.path.append(os.path.abspath(r"..")) 
sys.path.append(os.path.abspath(r".")) 
from lib.meshes_utils import save_ply

def if_not_exists_makedir(path ,comment = None , delete = False , confirm_delete = True):
    if not os.path.exists(path):
        os.makedirs(path)
        if comment is not None:
            print(comment)
    elif delete == False :
        print("folder will be overwritten")
    elif delete == True :
        if confirm_delete == True:
            print()
            print("Can I delete this folder? ")
            print(path)
            print()
            print("yes → enter 'y'")
            print()
            res = input()

            if res == "y":
                shutil.rmtree(path)
                os.makedirs(path)
            else :
                print("Please empty the folder")
                sys.exit()
        else:
            shutil.rmtree(path)
            os.makedirs(path)

def load_ply(path):
    """
    return -> vtx , nml , rgb , face , vtx_num , face_num
    """
    f = open(path,"r")
    ply = f.read()
    lines=ply.split("\n")

    rgb_flg = False
    nml_flg = False
    face_flg = False
    vtx_num = 0
    face_num = 0
    i = 0
    while(1):
        if "end_header" in lines[i]:
            #print("finish reading header")
            break
        if "element vertex" in lines[i]:
            vtx_num = int(lines[i].split(" ")[-1])         
        if "element face" in lines[i]:
            face_num = int(lines[i].split(" ")[-1])
            face_flg = True
        if "red" in lines[i] or "green" in lines[i] or "blue" in lines[i]:
            rgb_flg = True
        if "nx" in lines[i] or "ny" in lines[i] or "nz" in lines[i]:
            nml_flg = True
        i += 1
        if i == 100:
            print("load header error")
            sys.exit()
        header_len = i + 1
    #print("vtx :" , vtx_num ,"  face :" , face_num ,"  nml_flg: " , nml_flg,"  rgb_flg :" , rgb_flg,"  face_flg :" , face_flg)

    vtx = []
    nml = []
    rgb = []
    face = []

    if nml_flg and rgb_flg and face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))
            rgb.append(list(map(int,lines[i].split(" ")[6:9])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))
        

    elif nml_flg and rgb_flg            :
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))
            rgb.append(list(map(int,lines[i].split(" ")[6:9])))
        

    elif nml_flg           and  face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))
        
    elif            rgb_flg and face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            rgb.append(list(map(int,lines[i].split(" ")[3:6])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))

    elif nml_flg                       :
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            nml.append(list(map(float,lines[i].split(" ")[3:6])))

    elif            rgb_flg            :
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))
            rgb.append(list(map(int,lines[i].split(" ")[3:6])))

    elif                       face_flg:
        for i in range(header_len,vtx_num+header_len):
            vtx.append(list(map(float,lines[i].split(" ")[0:3])))

        for i in range(header_len+vtx_num,face_num+header_len+vtx_num):
            face.append(list(map(int,lines[i].split(" ")[1:4])))
    f.close()
    return vtx , nml , rgb , face , vtx_num , face_num

def load_obj(path):
    """
    return -> vtxs , nmls , uvs , faceID2vtxIDset , faceID2uvIDset , faceID2normalIDset , vtx_num , face_num
    """
    f = open(path,"r")
    obj = f.read()
    lines=obj.split("\n")

    vtx_num = 0
    face_num = 0

    vtxs = []
    nmls = []
    uvs = []
    faceID2vtxIDset = []
    faceID2uvIDset  = []
    faceID2normalIDset = []
    #i = 0
    for i in range(len(lines)):
        #if lines[i] == '':
        #    break
        line = lines[i].split(" ")
        if "v" == line[0]:
            vtxs.append(list(map(float,line[1:4])))
            vtx_num += 1
        if "vt" == line[0]:
            uvs.append(list(map(float,line[1:3])))
        if "vn" == line[0]:
            nmls.append(list(map(float,line[1:4])))
        if "f" == line[0]:
            face_num += 1
            if line[1].split("/")[0] != '' and line[2].split("/")[0] != '' and line[3].split("/")[0] != '':
                faceID2vtxIDset   .append([int(line[1].split("/")[0])-1,int(line[2].split("/")[0])-1,int(line[3].split("/")[0])-1]) 
            if line[1].split("/")[1] != '' and line[2].split("/")[1] != '' and line[3].split("/")[1] != '':
                faceID2uvIDset    .append([int(line[1].split("/")[1])-1,int(line[2].split("/")[1])-1,int(line[3].split("/")[1])-1]) 
            if line[1].split("/")[2] != '' and line[2].split("/")[2] != '' and line[3].split("/")[2] != '':
                faceID2normalIDset.append([int(line[1].split("/")[2])-1,int(line[2].split("/")[2])-1,int(line[3].split("/")[2])-1]) 
    #i+=1
    f.close()
    return vtxs , nmls , uvs , faceID2vtxIDset , faceID2uvIDset , faceID2normalIDset , vtx_num , face_num

def load_testID(test_folder_path):
    f = open(test_folder_path,"r")
    test_folder = f.read()
    test_folder_list = test_folder.split("\n")
    f.close()
    return list(map(int,test_folder_list))

def compute_normal_consistency(normal_similarity ,vertices_normal_GT , normal_pred_tmp, normal_hadamard , normal_dot , normal_norm1, normal_norm2 , dist_type_idx):
    normal_hadamard[dist_type_idx]   = vertices_normal_GT[dist_type_idx] * normal_pred_tmp
    normal_dot[dist_type_idx]        = np.sum(normal_hadamard[dist_type_idx] , axis = 1)    
    normal_norm1[dist_type_idx]      = np.linalg.norm(vertices_normal_GT[dist_type_idx] , axis=1)
    normal_norm2[dist_type_idx]      = np.linalg.norm(normal_pred_tmp , axis=1)
    normal_similarity[dist_type_idx] = normal_dot[dist_type_idx] / (normal_norm1[dist_type_idx] * normal_norm2[dist_type_idx])
    return normal_similarity

def compute_normal_consistency_point2faces(normal_similarity , dist_type_np , vertices_normal_GT , face_normal_pred  , index_np , normal_hadamard ,  normal_dot , normal_norm1, normal_norm2):
    dist_type_idx = np.nonzero(np.where(dist_type_np==0 , 1 , 0))

    normal_pred_tmp = np.zeros((len(dist_type_idx[0]),3) , np.float32)
    normal_pred_tmp = face_normal_pred[index_np[0][dist_type_idx]]

    normal_similarity = compute_normal_consistency(normal_similarity ,vertices_normal_GT , normal_pred_tmp, normal_hadamard , normal_dot , normal_norm1, normal_norm2 , dist_type_idx)
    return normal_similarity

def compute_normal_consistency_point2vertices(normal_similarity , dist_type_np , vertices_normal_GT , face_pred, vertices_normal_pred  , index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2):
    for i in range(3):
        d_type = i + 1
        dist_type_idx = np.nonzero(np.where(dist_type_np==d_type , 1 , 0))

        normal_pred_tmp = np.zeros((len(dist_type_idx[0]),3) , np.float32)
        normal_pred_tmp = vertices_normal_pred[face_pred[index_np[0][dist_type_idx]][:,i]]
 
        normal_similarity = compute_normal_consistency(normal_similarity ,vertices_normal_GT , normal_pred_tmp, normal_hadamard , normal_dot , normal_norm1, normal_norm2 , dist_type_idx)
    return normal_similarity

def compute_normal_consistency_point2edges(normal_similarity , dist_type_np , vertices_normal_GT , face_pred , vertices_normal_pred, index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2):
    for i in range(3):
        d_type = i + 4
        
        e = i + 1
        if e == 3:
            e = 0
    
        dist_type_idx = np.nonzero(np.where(dist_type_np==d_type , 1 , 0))
        normal_pred_tmp = np.zeros((len(dist_type_idx[0]),3) , np.float32)
        normal_pred_tmp = (vertices_normal_pred[face_pred[index_np[0][dist_type_idx]][:,i]] + vertices_normal_pred[face_pred[index_np[0][dist_type_idx]][:,e]]) / 2
     
        normal_similarity = compute_normal_consistency(normal_similarity ,vertices_normal_GT , normal_pred_tmp, normal_hadamard , normal_dot , normal_norm1, normal_norm2 , dist_type_idx)

    return normal_similarity

def display_closest_edge(vertices1 , vertices2 , face2 , v1v2_closest_f2idxs , dist_type , distance_color , save_path , sampling = 10):
    new_vtx  = []
    new_face = []
    new_rgb  = []
    j = 0

    face2 = face2.tolist()
    vertices2 = np.array(vertices2)
    for i , v1 in enumerate(vertices1):
        if i%sampling != 0:
            continue
        new_vtx.append(v1)
        if dist_type[i] == 0:
            vtx_tmp = np.mean(vertices2[face2[v1v2_closest_f2idxs[i]]] , axis= 0)                    #p2f
            new_vtx.append(vtx_tmp)
            new_vtx.append(vtx_tmp)
        elif dist_type[i] == 1:
            vtx_tmp = vertices2[face2[v1v2_closest_f2idxs[i]][0]]                                   #p2p
            new_vtx.append(vtx_tmp)
            new_vtx.append(vtx_tmp)
        elif dist_type[i] == 2:
            vtx_tmp = vertices2[face2[v1v2_closest_f2idxs[i]][1]]                                   #p2p
            new_vtx.append(vtx_tmp)
            new_vtx.append(vtx_tmp)
        elif dist_type[i] == 3:
            vtx_tmp = vertices2[face2[v1v2_closest_f2idxs[i]][2]]                                   #p2p
            new_vtx.append(vtx_tmp)
            new_vtx.append(vtx_tmp)
        elif dist_type[i] == 4:
            vtx_tmp = (vertices2[face2[v1v2_closest_f2idxs[i]][0]]  + vertices2[face2[v1v2_closest_f2idxs[i]][1]]) / 2  #p2e
            new_vtx.append(vtx_tmp)
            new_vtx.append(vtx_tmp)
        elif dist_type[i] == 5:
            vtx_tmp = (vertices2[face2[v1v2_closest_f2idxs[i]][1]]  + vertices2[face2[v1v2_closest_f2idxs[i]][2]]) / 2  #p2e
            new_vtx.append(vtx_tmp)
            new_vtx.append(vtx_tmp)
        elif dist_type[i] == 6:
            vtx_tmp = (vertices2[face2[v1v2_closest_f2idxs[i]][2]]  + vertices2[face2[v1v2_closest_f2idxs[i]][0]]) / 2  #p2e
            new_vtx.append(vtx_tmp)
            new_vtx.append(vtx_tmp)
        
        new_rgb.append(distance_color[i,:3])
        new_rgb.append(distance_color[i,:3])
        new_rgb.append(distance_color[i,:3])
        new_face.append([j , j + 1 , j + 2])
        j += 3
    save_ply(save_path ,new_vtx , None ,new_rgb , new_face , len(new_vtx) , len(new_face))

def save_color_mesh(distance , verts , face , save_path , cmap_type = "jet" , cmap_inverse = False , vmin=0 , vmax=1e-1):
    if type(distance) == torch.Tensor:
        disance_np = distance.to('cpu').detach().numpy().copy()
    else:
        disance_np = distance
    
    norm = matplotlib.colors.Normalize(vmin=vmin , vmax=vmax , clip = True)   #1mm(0.001m) ~ 10cm(0.1m) → to [0-1]
    norm_distance = norm(disance_np)
    
    if cmap_type =="jet" :
        if cmap_inverse :
            distance_color = plt.cm.jet_r(norm_distance)  
        else:
            distance_color = plt.cm.jet(norm_distance)                                                    #→RGBA : "jet" color map(R is biggest & B is smallest))
    elif cmap_type =="viridis" :
        if cmap_inverse :
            distance_color = plt.cm.viridis_r(norm_distance)                                                    
        else:     
            distance_color = plt.cm.viridis(norm_distance)                                                    
    elif cmap_type =="gray" :
        if cmap_inverse :
            distance_color = plt.cm.gray_r(norm_distance) 
        else:
            distance_color = plt.cm.gray(norm_distance) 
    
    ms = pymeshlab.MeshSet()
    new_mesh = pymeshlab.Mesh(vertex_matrix = verts ,face_matrix = face , v_color_matrix = distance_color)  
    ms.add_mesh(new_mesh)
    ms.save_current_mesh(save_path , binary = False) 

    distance_color[:,:3] = (distance_color[:,:3] * 255)
    distance_color = distance_color.astype(np.uint8)

    return distance_color

def load_factory(path , device):
    '''
    return verts_tensor , face_tensor , face_vertices , vertices_normal , face_normal
    '''
    #Load GT Mesh
    ext = os.path.splitext(path)[-1]
    #verts = torch.Tensor(np.load(GT_path)["scan_pc"])   #from cape npz
    if ext == ".ply":
        verts , _ , _ , face , _ , _ =  load_ply(path)
        verts_tensor = torch.tensor(verts)
        face_tensor  = torch.tensor(face)
        face         = np.array(face)
    elif ext == ".obj":
        """
        verts_tensor, face_tensor , _ , _ , _ , _ , _ , _ = kaolin.io.obj.import_mesh(GT_path) 
        verts = verts_tensor.to('cpu').detach().numpy().copy()
        face  = face_tensor.to('cpu').detach().numpy().copy()
        """
        verts , _ , _ , face , _ , _ , _ , _ = load_obj(path) 
        verts_tensor = torch.tensor(verts)
        face_tensor  = torch.tensor(face)
        face         = np.array(face)

    verts_tensor  = verts_tensor.unsqueeze(0).to(device)       #.to('cpu')
    face_tensor   = face_tensor.to(device)                     #.to('cpu')
    face_vertices  = kaolin.ops.mesh.index_vertices_by_faces(verts_tensor,face_tensor)

    #compute vertices & face normal 
    face_normal_tensor    = kaolin.ops.mesh.face_normals(face_vertices , True)
    face_normal = face_normal_tensor[0].to('cpu').detach().numpy().copy()

    ms = pymeshlab.MeshSet()
    new_mesh = pymeshlab.Mesh(vertex_matrix = verts ,face_matrix = face , f_normals_matrix = face_normal)
    ms.add_mesh(new_mesh)
    ms.compute_normal_per_vertex()
    curr_mesh = ms.mesh(0)
    vertices_normal = curr_mesh.vertex_normal_matrix()

    return verts , verts_tensor , face , face_vertices , vertices_normal , face_normal

def chamfer(GT_path, pred_path , GT_to_Pred_save_path , Pred_to_GT_save_path, device = None):
    verts_GT  , verts_GT_tensor   , face_GT   , face_vertices_GT   , vertices_normal_GT   , face_normal_GT   = load_factory(GT_path   , device)
    verts_pred, verts_pred_tensor , face_pred , face_vertices_pred , vertices_normal_pred , face_normal_pred = load_factory(pred_path , device)

    print("start computing chamfer distance")

    #######################################################
    ### GTMesh's point 2 PredinctedMesh's face distance
    #######################################################

    distance_gt2pred, index, dist_type   = kaolin.metrics.trianglemesh.point_to_mesh_distance(verts_GT_tensor, face_vertices_pred)
    #indexはdist_typeに依らず、faceのindexを返す
    dist_type_np = dist_type.to('cpu').detach().numpy().copy()[0]
    index_np     = index.to('cpu').detach().numpy().copy()   

    distance_gt2pred     = torch.sqrt(distance_gt2pred)
    loss_chamfer_gt2pred = torch.nanmean(distance_gt2pred[0])
    
    ###debug color ###
    distance_color_gt2pred = save_color_mesh(distance_gt2pred[0] , verts_GT , face_GT , GT_to_Pred_save_path + "_dist.ply" , vmin=0.0 , vmax=5e-2)
    display_closest_edge(verts_GT , verts_pred , face_pred , index_np[0] , dist_type_np , distance_color_gt2pred , GT_to_Pred_save_path + "_closest_edge.ply" )

    #compute normal
    normal_hadamard   = np.zeros((len(distance_gt2pred[0]) , 3) , np.float32)
    normal_similarity = np.zeros( len(distance_gt2pred[0])      , np.float32)
    normal_dot        = np.zeros( len(distance_gt2pred[0])      , np.float32)
    normal_norm1      = np.zeros( len(distance_gt2pred[0])      , np.float32)
    normal_norm2      = np.zeros( len(distance_gt2pred[0])      , np.float32)
   
    normal_similarity = compute_normal_consistency_point2faces   (normal_similarity , dist_type_np , vertices_normal_GT , face_normal_pred                 , index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2)
    normal_similarity = compute_normal_consistency_point2vertices(normal_similarity , dist_type_np , vertices_normal_GT , face_pred , vertices_normal_pred , index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2)
    normal_similarity = compute_normal_consistency_point2edges   (normal_similarity , dist_type_np , vertices_normal_GT , face_pred , vertices_normal_pred , index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2)
    
    normal_similarity_gt2pred = normal_similarity
    loss_normal_gt2pred = np.nanmean(normal_similarity)

    normal_similarity_gt2pred = np.where(normal_similarity_gt2pred > 0.0 , normal_similarity_gt2pred , 0.0)     #add threshold

    ###debug color ###
    save_color_mesh(normal_similarity_gt2pred , verts_GT , face_GT , GT_to_Pred_save_path + "_nml.ply" , cmap_inverse= True , vmin = 0.0 , vmax = 1.0)
    #######################################################
    ### PredinctedMesh's point 2 GTMesh's face distance ###
    #######################################################

    distance_pred2gt, index, dist_type   = kaolin.metrics.trianglemesh.point_to_mesh_distance(verts_pred_tensor , face_vertices_GT)
    #indexはdist_typeに依らず、faceのindexを返す
    dist_type_np = dist_type.to('cpu').detach().numpy().copy()[0]
    index_np     = index.to('cpu').detach().numpy().copy()   

    distance_pred2gt     = torch.sqrt(distance_pred2gt)
    loss_chamfer_pred2gt = torch.nanmean(distance_pred2gt[0])
    
    ###debug color ###
    distance_color_pred2gt = save_color_mesh(distance_pred2gt[0] , verts_pred , face_pred , Pred_to_GT_save_path + "_dist.ply" , vmin=0.0 , vmax=5e-2)
    display_closest_edge(verts_pred , verts_GT , face_GT , index_np[0] , dist_type_np , distance_color_pred2gt , Pred_to_GT_save_path + "_closest_edge.ply" )
    
    #compute normal
    normal_hadamard   = np.zeros((len(distance_pred2gt[0]) , 3) , np.float32)
    normal_similarity = np.zeros( len(distance_pred2gt[0])      , np.float32)
    normal_dot        = np.zeros( len(distance_pred2gt[0])      , np.float32)
    normal_norm1      = np.zeros( len(distance_pred2gt[0])      , np.float32)
    normal_norm2      = np.zeros( len(distance_pred2gt[0])      , np.float32)
   
    normal_similarity = compute_normal_consistency_point2faces   (normal_similarity , dist_type_np , vertices_normal_pred , face_normal_GT               , index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2)
    normal_similarity = compute_normal_consistency_point2vertices(normal_similarity , dist_type_np , vertices_normal_pred , face_GT , vertices_normal_GT , index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2)
    normal_similarity = compute_normal_consistency_point2edges   (normal_similarity , dist_type_np , vertices_normal_pred , face_GT , vertices_normal_GT , index_np , normal_hadamard , normal_dot , normal_norm1 , normal_norm2)
    
    normal_similarity_pred2gt = normal_similarity
    loss_normal_pred2gt = np.nanmean(normal_similarity)

    normal_similarity_pred2gt = np.where(normal_similarity_pred2gt > 0.0 , normal_similarity_pred2gt , 0.0) #add threshold

    ###debug color ###
    save_color_mesh(normal_similarity_pred2gt , verts_pred , face_pred , Pred_to_GT_save_path + "_nml.ply", cmap_inverse= True , vmin = 0.0 , vmax = 1.0)

    #summary_distance_gt2pred  = (np.where(normal_similarity_gt2pred > 0.1 , distance_gt2pred.to('cpu').detach().numpy().copy()[0] / normal_similarity_gt2pred , distance_gt2pred.to('cpu').detach().numpy().copy()[0] / 0.1 )) #arc_cons (0.1) = 84.26082952°
    #summary_distance2_pred2gt = (np.where(normal_similarity_pred2gt > 0.1 , distance_pred2gt.to('cpu').detach().numpy().copy()[0] / normal_similarity_pred2gt , distance_pred2gt.to('cpu').detach().numpy().copy()[0] / 0.1 )) #arc_cons (0.1) = 84.26082952°
    
    #summary_distance_gt2pred  = np.where(normal_similarity_gt2pred > 0.0 , distance_gt2pred.to('cpu').detach().numpy().copy()[0] / (normal_similarity_gt2pred + 1.0) , np.nan) #1.1 → Exclude values that are too small
    #summary_distance2_pred2gt = np.where(normal_similarity_pred2gt > 0.0 , distance_pred2gt.to('cpu').detach().numpy().copy()[0] / (normal_similarity_pred2gt + 1.0) , np.nan) #1.1 → Exclude values that are too small

    #summary_distance_gt2pred  = distance_gt2pred.to('cpu').detach().numpy().copy()[0] / (normal_similarity_gt2pred + 1.1) #1.1 → Exclude values that are too small
    #summary_distance2_pred2gt = distance_pred2gt.to('cpu').detach().numpy().copy()[0] / (normal_similarity_pred2gt + 1.1) #1.1 → Exclude values that are too small

    summary_distance_gt2pred = distance_gt2pred.to('cpu').detach().numpy().copy()[0] * ((1 - normal_similarity_gt2pred ** 2) ** (1/2))
    summary_distance_pred2gt = distance_pred2gt.to('cpu').detach().numpy().copy()[0] * ((1 - normal_similarity_pred2gt ** 2) ** (1/2))

    save_color_mesh(summary_distance_gt2pred  , verts_GT   , face_GT   , GT_to_Pred_save_path + "_summary.ply" , cmap_inverse= False , vmin = 0.0 , vmax = 1e-2)
    save_color_mesh(summary_distance_pred2gt  , verts_pred , face_pred , Pred_to_GT_save_path + "_summary.ply" , cmap_inverse= False , vmin = 0.0 , vmax = 1e-2)


    loss_chamfer_summary = 0.5 * np.nanmean(summary_distance_gt2pred) + 0.5* np.nanmean(summary_distance_pred2gt)

    return (loss_chamfer_gt2pred * 1000, loss_normal_gt2pred , distance_gt2pred[0] , normal_similarity_gt2pred \
          , loss_chamfer_pred2gt * 1000, loss_normal_pred2gt , distance_pred2gt[0] , normal_similarity_pred2gt \
          , loss_chamfer_summary * 1000, summary_distance_gt2pred , summary_distance_pred2gt )       

def main():
    parser = argparse.ArgumentParser(description='chamfer distance')
    parser.add_argument(
        '--pred_dir_path',
        type=str,
        help='')
    
    parser.add_argument(
        '--pred_glob_name',
        type=str,
        help='')

    parser.add_argument(
        '--gt_dir_path',
        type=str,
        help='')
    
    parser.add_argument(
        '--gt_header',
        default= "",
        type=str,
        help='')
    
    parser.add_argument(
        '--gt_ext',
        default= ".obj",
        type=str,
        help='')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.pred_dir_path == None:
        print("select path of predicted folder")
        sys.exit()
    else:
        pred_dir_path = args.pred_dir_path

    if args.pred_glob_name == None:
        print("select path of predicted folder")
        sys.exit()
    else:
        pred_glob_name = args.pred_glob_name
    

    if args.gt_header == None:
        print("select path of gt_header")
        sys.exit()
    else:
        gt_header = args.gt_header

    if args.gt_ext == None:
        print("select path of gt_ext")
        sys.exit()
    else:
        gt_ext = args.gt_ext

    if args.gt_dir_path == None:
        print("select path of predicted folder")
        sys.exit()
    else:
        gt_dir_path = args.gt_dir_path

    #pred_paths = natsorted(glob.glob(os.path.join(pred_dir_path , "[0-9][0-9][0-9][0-9].ply")))  #don't forget "/mnt/d/"
    #pred_paths = natsorted(glob.glob(os.path.join(pred_dir_path , "mesh_[0-9][0-9][0-9][0-9].ply")))  #don't forget "/mnt/d/"
    #pred_paths = natsorted(glob.glob(pred_dir_path +"/IPLMesh_[0-9][0-9][0-9][0-9].ply"))  #don't forget "/mnt/d/"
    #pred_paths = natsorted(glob.glob(os.path.join(pred_dir_path  , "ICPmesh_[0-9][0-9][0-9][0-9].ply")))  #don't forget "/mnt/d/"
    pred_paths = natsorted(glob.glob(os.path.join(pred_dir_path  , pred_glob_name)))  #don't forget "/mnt/d/"

    gt_paths = []
    ids = []
    print("data_num = " , len(pred_paths))
    # go through all predicted meshes => make GT paths
    for pred_path in pred_paths:
        #print(pred_path)

        basename = os.path.basename(pred_path)        #ours / pop / snarf
        id = basename.split(".")[0].split("_")[-1]    #ours / pop / snarf

        #file_name = "mesh_" + str(id).zfill(4) + ".ply"
        #file_name = str(id).zfill(4) + ".obj"
        file_name = gt_header + str(id).zfill(4) + gt_ext

        gt_paths.append(os.path.join(gt_dir_path, file_name))
        ids.append(id)
        
    if len(gt_paths) != len(pred_paths):
        print("2 meshes num is different")
        print(len(gt_paths) , " : " , len(pred_paths) )
        sys.exit()
        
    p2f_chamfer_save_path = os.path.join(args.pred_dir_path , "p2f")
    if_not_exists_makedir(p2f_chamfer_save_path , confirm_delete = False)
    
    result_csv = os.path.join(p2f_chamfer_save_path,"chamfer_result.csv")  #save_path
    
    gt2pred_hist_dir_path = os.path.join(p2f_chamfer_save_path, "gt2pred_hist")
    pred2gt_hist_dir_path = os.path.join(p2f_chamfer_save_path, "pred2gt_hist")
    if_not_exists_makedir(gt2pred_hist_dir_path , confirm_delete = False)
    if_not_exists_makedir(pred2gt_hist_dir_path , confirm_delete = False)

    gt2pred_summary_hist_dir_path = os.path.join(p2f_chamfer_save_path, "gt2pred_summary_hist")
    pred2gt_summary_hist_dir_path = os.path.join(p2f_chamfer_save_path, "pred2gt_summary_hist")
    if_not_exists_makedir(gt2pred_summary_hist_dir_path , confirm_delete = False)
    if_not_exists_makedir(pred2gt_summary_hist_dir_path , confirm_delete = False)

    GT_to_Pred_RGB_save_dir_path = os.path.join(p2f_chamfer_save_path, "GT_to_Pred_RGB")
    Pred_to_GT_RGB_save_dir_path = os.path.join(p2f_chamfer_save_path, "Pred_to_GT_RGB")
    if_not_exists_makedir(GT_to_Pred_RGB_save_dir_path , confirm_delete = False)
    if_not_exists_makedir(Pred_to_GT_RGB_save_dir_path , confirm_delete = False)
    

    f = open(result_csv,"w")
    f.write("id" + "," + "s2m(gt2pred) (mm)" + "," + "n(gt2pred)" + "," + "s2m(gt2pred) (mm)" + "," + "n(gt2pred)" + "," + "chamfer_summary (mm)" +"\n")

    sum_chamfer_gt2pred = 0
    sum_chamfer_pred2gt = 0
    sum_normal_gt2pred = 0
    sum_normal_pred2gt = 0
    sum_chamfer_summary = 0

    cnt_gt2pred = 0
    cnt_pred2gt = 0
    cnt_summary = 0
    all_distance_gt2pred_lists = []
    all_distance_pred2gt_lists = []
    all_distance_summary_gt2pred_lists = []
    all_distance_summary_pred2gt_lists = []

    all_normal_gt2pred_lists = []
    all_normal_pred2gt_lists = []
    #for i, (gt_path , pred_path) in tqdm.tqdm(enumerate(zip(gt_paths,pred_paths))):
    for (gt_path , pred_path ,id) in tqdm.tqdm(zip(gt_paths,pred_paths,ids)):
        print("pred_path : " , pred_path)
        print("gt_path : " , gt_path)
        #id = str(i).zfill(4)
        id = str(id).zfill(4)
        if os.path.isfile(pred_path) == False or os.path.isfile(gt_path) == False:
            print("no that file!!")
            continue

        GT_to_Pred_RGB_save_path = os.path.join(GT_to_Pred_RGB_save_dir_path , "GTvtx_to_PredFace_RGB_"+ id)
        Pred_to_GT_RGB_save_path = os.path.join(Pred_to_GT_RGB_save_dir_path , "PredFace_to_GTvtx_RGB_"+ id)

        loss_chamfer_gt2pred , loss_normal_gt2pred  , distance_for_std_gt2pred , normal_for_std_gt2pred , \
        loss_chamfer_pred2gt , loss_normal_pred2gt  , distance_for_std_pred2gt , normal_for_std_pred2gt , \
        loss_chamfer_summary , loss_summary_gt2pred , loss_summary_pred2gt                                  = chamfer(gt_path, pred_path , GT_to_Pred_RGB_save_path , Pred_to_GT_RGB_save_path , device)
        all_distance_gt2pred_lists.append(distance_for_std_gt2pred)
        all_normal_gt2pred_lists.append(normal_for_std_gt2pred)
        all_distance_summary_gt2pred_lists.append(loss_summary_gt2pred)

        all_distance_pred2gt_lists.append(distance_for_std_pred2gt)
        all_normal_pred2gt_lists.append(normal_for_std_pred2gt)
        all_distance_summary_pred2gt_lists.append(loss_summary_pred2gt)

        #print("loss_chamfer_gt2pred : " , loss_chamfer_gt2pred , " mm")
        #print("loss_normal_gt2pred  : " , loss_normal_gt2pred )

        #print("loss_chamfer_pred2gt : " , loss_chamfer_pred2gt , " mm")
        #print("loss_normal_pred2gt  : " , loss_normal_pred2gt )

        print("loss_chamfer_summary : " , loss_chamfer_summary , " mm")

        loss_chamfer_gt2pred_str = str(loss_chamfer_gt2pred.to('cpu').detach().numpy().copy().item())
        loss_chamfer_pred2gt_str = str(loss_chamfer_pred2gt.to('cpu').detach().numpy().copy().item())
        loss_chamfer_summary_str = str(loss_chamfer_summary)

        loss_normal_gt2pred_str = str(loss_normal_gt2pred)
        loss_normal_pred2gt_str = str(loss_normal_pred2gt)

        f.write(id + "," + loss_chamfer_gt2pred_str + "," +  loss_normal_gt2pred_str + "," + loss_chamfer_pred2gt_str + "," +  loss_normal_pred2gt_str + "," +  loss_chamfer_summary_str + "\n")

        #save hist as npz
        disance_np_gt2pred = distance_for_std_gt2pred.to('cpu').detach().numpy().copy()
        disance_np_pred2gt = distance_for_std_pred2gt.to('cpu').detach().numpy().copy()

        counts_gt2pred, bins_gt2pred = np.histogram(disance_np_gt2pred , bins = 100 , range = (0 , 1e-1))
        new_bins_gt2pred = bins_gt2pred[:-1] + 0.0005
        gt2pred_save_path = os.path.join(gt2pred_hist_dir_path , id + ".npz")
        np.savez(gt2pred_save_path , counts = counts_gt2pred , bins = new_bins_gt2pred)

        counts_pred2gt, bins_pred2gt = np.histogram(disance_np_pred2gt , bins = 100 , range = (0 , 1e-1))
        new_bins_pred2gt = bins_pred2gt[:-1] + 0.0005
        pred2gt_save_path = os.path.join(pred2gt_hist_dir_path , id + ".npz")
        np.savez(pred2gt_save_path , counts = counts_pred2gt , bins = new_bins_pred2gt)

        counts_summary_gt2pred, bins_summary_gt2pred = np.histogram(loss_summary_gt2pred , bins = 100 , range = (0 , 1e-1))
        new_bins_summary_gt2pred = bins_summary_gt2pred[:-1] + 0.0005
        gt2pred_summary_save_path = os.path.join(gt2pred_summary_hist_dir_path , id + ".npz")
        np.savez(gt2pred_summary_save_path , counts = counts_summary_gt2pred , bins = new_bins_summary_gt2pred)

        counts_summary_pred2gt, bins_summary_pred2gt = np.histogram(loss_summary_pred2gt , bins = 100 , range = (0 , 1e-1))
        new_bins_summary_pred2gt = bins_summary_pred2gt[:-1] + 0.0005
        pred2gt_summary_save_path = os.path.join(pred2gt_summary_hist_dir_path , id + ".npz")
        np.savez(pred2gt_summary_save_path , counts = counts_summary_pred2gt , bins = new_bins_summary_pred2gt)

        #plt.figure() 
        #plt.plot(new_bins, counts)
        #plt.show()

        if torch.any(torch.isnan(loss_chamfer_gt2pred)) == False:
            sum_chamfer_gt2pred += loss_chamfer_gt2pred
            sum_normal_gt2pred  += loss_normal_gt2pred
            cnt_gt2pred += 1
        if torch.any(torch.isnan(loss_chamfer_pred2gt)) == False:
            sum_chamfer_pred2gt += loss_chamfer_pred2gt
            sum_normal_pred2gt  += loss_normal_pred2gt
            cnt_pred2gt += 1
        if np.any(np.isnan(loss_chamfer_summary)) == False:
            sum_chamfer_summary += loss_chamfer_summary
            cnt_summary += 1
        print()
    avg_chamfer_gt2pred = sum_chamfer_gt2pred / cnt_gt2pred
    avg_normal_gt2pred  = sum_normal_gt2pred /  cnt_gt2pred

    avg_chamfer_pred2gt = sum_chamfer_pred2gt / cnt_pred2gt
    avg_normal_pred2gt  = sum_normal_pred2gt /  cnt_pred2gt

    avg_chamfer_summary = sum_chamfer_summary / cnt_summary

    all_distance_gt2pred = torch.cat(all_distance_gt2pred_lists)
    all_normal_gt2pred   = np.concatenate(all_normal_gt2pred_lists)

    all_distance_pred2gt = torch.cat(all_distance_pred2gt_lists)
    all_normal_pred2gt   = np.concatenate(all_normal_pred2gt_lists)

    all_distance_summary_gt2pred = np.concatenate(all_distance_summary_gt2pred_lists)
    all_distance_summary_pred2gt = np.concatenate(all_distance_summary_pred2gt_lists)

    #save hist as npz
    all_distance_np_gt2pred = all_distance_gt2pred.to('cpu').detach().numpy().copy()
    counts_gt2pred, bins_gt2pred = np.histogram(all_distance_np_gt2pred , bins = 100 , range = (0 , 1e-1))
    new_bins_gt2pred = bins_gt2pred[:-1] + 0.0005
    gt2pred_save_path = os.path.join(gt2pred_hist_dir_path , "total" + ".npz")
    np.savez(gt2pred_save_path , counts = counts_gt2pred , bins = new_bins_gt2pred)

    all_distance_np_pred2gt = all_distance_pred2gt.to('cpu').detach().numpy().copy()
    counts_pred2gt, bins_pred2gt = np.histogram(all_distance_np_pred2gt , bins = 100 , range = (0 , 1e-1))
    new_bins_pred2gt = bins_pred2gt[:-1] + 0.0005
    pred2gt_save_path = os.path.join(pred2gt_hist_dir_path , "total" + ".npz")
    np.savez(pred2gt_save_path , counts = counts_pred2gt , bins = new_bins_pred2gt)

    counts_summary_gt2pred, bins_summary_gt2pred = np.histogram(all_distance_summary_gt2pred , bins = 100 , range = (0 , 1e-1))
    new_bins_summary_gt2pred = bins_summary_gt2pred[:-1] + 0.0005
    gt2pred_summary_save_path = os.path.join(gt2pred_summary_hist_dir_path , "total" + ".npz")
    np.savez(gt2pred_summary_save_path , counts = counts_summary_gt2pred , bins = new_bins_summary_gt2pred)

    counts_summary_pred2gt, bins_summary_pred2gt = np.histogram(all_distance_summary_pred2gt , bins = 100 , range = (0 , 1e-1))
    new_bins_summary_pred2gt = bins_summary_pred2gt[:-1] + 0.0005
    pred2gt_summary_save_path = os.path.join(pred2gt_summary_hist_dir_path , "total" + ".npz")
    np.savez(pred2gt_summary_save_path , counts = counts_summary_pred2gt , bins = new_bins_summary_pred2gt)

    #compute std
    distance_std_gt2pred = np.nanstd(all_distance_np_gt2pred)
    normal_std_gt2pred   = np.nanstd(all_normal_gt2pred)

    distance_std_pred2gt = np.nanstd(all_distance_np_pred2gt)
    normal_std_pred2gt   = np.nanstd(all_normal_pred2gt)

    #print("avg_chamfer : ",avg_chamfer_gt2pred , " mm  ,   std : " , distance_std_gt2pred)
    #print("avg_normal : " ,avg_normal_gt2pred , " std : " , normal_std_gt2pred)

    avg_chamfer_str_gt2pred = str(avg_chamfer_gt2pred.to('cpu').detach().numpy().copy().item())
    avg_normal_str_gt2pred = str(avg_normal_gt2pred)
    distance_std_str_gt2pred = str(distance_std_gt2pred)
    normal_std_str_gt2pred  = str(normal_std_gt2pred)

    #print("avg_chamfer : ",avg_chamfer_pred2gt , " mm  ,   std : " , distance_std_pred2gt)
    #print("avg_normal : " ,avg_normal_pred2gt , " std : " , normal_std_pred2gt)

    avg_chamfer_str_pred2gt = str(avg_chamfer_pred2gt.to('cpu').detach().numpy().copy().item())
    avg_normal_str_pred2gt = str(avg_normal_pred2gt)
    distance_std_str_pred2gt = str(distance_std_pred2gt)
    normal_std_str_pred2gt  = str(normal_std_pred2gt)

    print("avg_chmafer : " , avg_chamfer_summary  , " mm ")
    avg_chamfer_str_summary = str(avg_chamfer_summary)

    f.write("average" + "," + avg_chamfer_str_gt2pred  + "," + avg_normal_str_gt2pred + "," + avg_chamfer_str_pred2gt  + "," + avg_normal_str_pred2gt + "," + avg_chamfer_str_summary + "\n")
    f.write("std    " + "," + distance_std_str_gt2pred + "," + normal_std_str_gt2pred + "," + distance_std_str_pred2gt + "," + normal_std_str_pred2gt + "\n")
    f.close()

    """
    avg_chamfer = avg_chamfer.detach().numpy().copy()
    avg_normal = avg_normal.detach().numpy().copy()
    np.savetxt(args.pred_dir_path,"chamfer_result.npz", avg_chamfer)
    np.savetxt(args.pred_dir_path,"normal_result.npz", avg_normal)
    """
    

if __name__ == "__main__":
    main()



    