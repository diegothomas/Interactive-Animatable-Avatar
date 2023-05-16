import numpy as np
import pandas as pd
import json
import math
from os.path import exists, join
import time
import tqdm
import sys

# SMPL requrement
import chumpy as ch
import smpl_webuser
import smpl_webuser.serialization
import smpl_webuser.verts
import smpl_webuser.lbs
from smpl_webuser.lbs import verts_core
from smpl_webuser.lbs import global_rigid_transformation
from smpl_webuser.verts import verts_decorated
from smpl_webuser.serialization import load_model
from smpl_webuser.serialization import save_model

import smpl_utils
import ply, obj
import argparse
import pickle as pickle
import os
import subprocess
import struct

root_path = os.path.abspath("../..")

def ComputeNormales (V, F):
    normales = np.zeros(V.shape)
    for f in F:
        v1 = V[f[1],:] - V[f[0], :]
        v2 = V[f[2],:] - V[f[0], :]
        n = np.cross(v1,v2)
        #print(v1, v2, n)
        n = n/np.linalg.norm(n)
        normales[f[0],:] = normales[f[0],:] + n
        normales[f[1],:] = normales[f[1],:] + n
        normales[f[2],:] = normales[f[2],:] + n
            
    for i in range(V.shape[0]):
        if np.linalg.norm(normales[i,:]) > 0.0:
            normales[i,:] = normales[i,:]/np.linalg.norm(normales[i,:])
    
    return normales


def OpenposeToSMPL(pose):
    pose_list = np.zeros((24,3))

    pose_list[0,:] = (5.0*pose[8,:] + pose[1,:])/6.0
    pose_list[1,:] = pose[12,:]
    pose_list[2,:] = pose[9,:]
    pose_list[3,:] = (2.0*pose[8,:] + pose[1,:])/3.0
    pose_list[4,:] = pose[13,:]
    pose_list[5,:] = pose[10,:]
    pose_list[6,:] = (pose[8,:] + pose[1,:])/2.0
    pose_list[7,:] = pose[14,:]
    pose_list[8,:] = pose[11,:]
    pose_list[9,:] = (pose[8,:] + 2.0*pose[1,:])/3.0
    pose_list[10,:] = pose[19,:]
    pose_list[11,:] = pose[22,:]
    pose_list[12,:] = (pose[1,:] + pose[0,:])/2.0
    pose_list[13,:] = (pose[1,:] + pose[5,:])/2.0
    pose_list[14,:] = (pose[1,:] + pose[2,:])/2.0
    pose_list[15,:] = pose[0,:]
    pose_list[16,:] = pose[5,:]
    pose_list[17,:] = pose[2,:]
    pose_list[18,:] = pose[6,:]
    pose_list[19,:] = pose[3,:]
    pose_list[20,:] = pose[7,:]
    pose_list[21,:] = pose[4,:]
    
    arm1 = pose[7,:] - pose[6,:]
    pose_list[22,:] = pose[7,:] + 0.3*arm1
    arm2 = pose[4,:] - pose[3,:]
    pose_list[23,:] = pose[4,:] + 0.3*arm2

    return pose_list

def FitSMPL(init_pose_path, scanpath, outputpath, tmppath, verbose, idx , opt_target , init_flg  , init_trans_path = None , m = None , first_flg = False):
    # <========= Load 3D skeleton
    print("init_pose_path : ", init_pose_path)
    print("scan_path      : ", scanpath)
    if not os.path.exists(scanpath):
        print("Not extist : " , scanpath)
        return False
    
    if init_flg == True:
        trans      = scan_trans = np.array([0.0, 0.0, 0.0])
        pose       = np.fromfile(init_pose_path  , dtype = np.float32)
        init_trans = np.fromfile(init_trans_path , dtype =np.float32)
        init_trans = init_trans.tolist()
        #if copy init_trans from blender and export mesh as obj from blender 
        #blender export an obj file rotated -90 degrees around the x-axis : scan_trans = Rx(-90) * init_trans
        scan_trans[0] = init_trans[0]
        scan_trans[1] = init_trans[2]
        scan_trans[2] = -init_trans[1]

        #here , we use smpl trans 
        trans = -scan_trans
        #trans.tofile("D:/Human_data/RenderPeople/4D-people/rp_aliyah_4d_004_dancing_BLD/data/smplparams_centered/trans_"+ str(idx).zfill(4) +".bin")
    else:
        data = pickle.load(open(init_pose_path, "rb"))
        pose  = np.array(data['pose'])
        betas = np.array(data['betas'])
        trans = np.array(data['trans'])

    '''trans = trans.astype(np.float32)
    trans.tofile("D:/Human_data/RenderPeople/4D-people/rp_aliyah_4d_004_dancing_BLD/data/smplparams_centered/trans_"+ str(idx).zfill(4) +".bin")
    return True'''

    # <========= Load 3D scan
    V , F = obj.load_obj_tmp(scanpath)
    #[V, _, F, _] = obj.load_obj_text(scanpath)
    #V = V*0.01
    #[V, F] = ply.load_ply(scanpath)
    #V = (1.0/data['body_scale'])*(V - data['global_body_translation'])
    # add global orient
    N = ComputeNormales(V, F)

    ply.save_ply_nmle(outputpath+"input.ply", np.transpose(V), np.transpose(N), np.transpose(F))
   
    # <========= LOAD SMPL MODEL 
    # Init upright t-pose
    initial_model = m.copy()
    if init_flg == False:
        for i in range(initial_model.betas.shape[0]):
            initial_model.betas[i] = betas[i]
  
    ##### SAVE T-SHAPE ###############
    _, J = verts_core(initial_model.pose, initial_model.r, initial_model.J, initial_model.weights, initial_model.kintree_table, want_Jtr = True)        
    J = np.array(J).astype(np.float32)
    
    if init_flg == False and first_flg == True:
        smpl_utils.save_smpl_obj(os.path.join(os.path.dirname(outputpath) ,"T-shape.obj"), initial_model)
        J.tofile(os.path.join(os.path.dirname(outputpath) , "T_joints.bin"))
        if (verbose):
            ply.save_ply(os.path.join(os.path.dirname(outputpath) , "T_joints.ply") , J.T)

    ###################################

    for i in range(initial_model.pose.shape[0]):
        initial_model.pose[i] = pose[i]
        #initial_model.pose[i] = data['body_pose'][0,i]
    initial_model.trans[0] = trans[0]
    initial_model.trans[1] = trans[1]
    initial_model.trans[2] = trans[2]
    
    
    itr = 10
    if (verbose):
        ply.save_ply(outputpath+"init_s.ply", np.transpose(initial_model.J_transformed.r))
    
    # <========= FIT SMPL BODY MODEL
    print('\n\n=> (1) Find pose that fits voxels and joints3D (optional)\n\n')
    trans_model = initial_model.copy()

    for it in range(itr):
        """
        # <===== Compute correspondences
        '''if it == 0:
            V = V/1.1
            ply.save_ply_nmle(outputpath+"input.ply", np.transpose(V), np.transpose(N), np.transpose(F))'''
        ply.save_ply(outputpath+"model.ply", np.transpose(trans_model.r), np.transpose(trans_model.f))

        #For each point in the smpl model, record the coordinates of the nearest point on the scan as corr.bin. The 0~3th are in xyz, and the 4th is Exist (=1.0),Not(=0.0).
        #And , for each point on the scan, find the closest point on the smpl model and record it as cross_corr.bin. The 0th point is the ID of the model, and the 1st point is Exist(=1.0) or Not(=0.0).
        command = ['build/Release/matchsmpl', '--path_model', outputpath+"model.ply", '--path_scan',  outputpath+"input.ply", '--path_out', outputpath] 
        #command = ['build/Release/matchsmpl', '--path_model', outputpath+"input.ply", '--path_scan',  outputpath+"input.ply", '--path_out', outputpath]
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)

        time.sleep(1) # Sleep for 3 seconds
        """
        #return
        '''if it == 0:
            V = V*1.1
            ply.save_ply_nmle(outputpath+"input.ply", np.transpose(V), np.transpose(N), np.transpose(F))'''

        #read binary file
        """
        vertices_corr = np.zeros((trans_model.r.shape[0], 3))
        weights_corr = np.zeros((trans_model.r.shape[0]))
        print(vertices_corr.shape, weights_corr.shape)
        file = open(outputpath+"corr.bin", "rb")
        for i in range(trans_model.r.shape[0]):
            #print(trans_model.r[i,:])
            #print(struct.unpack('f', file.read(4)))
            vertices_corr[i,0] = struct.unpack('f', file.read(4))[0]#float(file.read(4))
            vertices_corr[i,1] = struct.unpack('f', file.read(4))[0]#float(file.read(4))
            vertices_corr[i,2] = struct.unpack('f', file.read(4))[0]#float(file.read(4))
            weights_corr[i] = struct.unpack('f', file.read(4))[0]#float(file.read(4))
        file.close()
        #print(trans_model.r[-1,:])
        #print(vertices_corr[-1, :])
        #print(weights_corr)
        """

        #return
         # <==== Get cross correspondences
        indices_cross_corr = np.zeros(V.shape[0])
        cross_weights_corr = np.zeros(V.shape[0])
        print(indices_cross_corr.shape, cross_weights_corr.shape)
        #c_file = open(outputpath+"cross_corr.bin", "rb")
        for i in range(V.shape[0]):
            #id = int(struct.unpack('f', c_file.read(4))[0])
            #w = struct.unpack('f', c_file.read(4))[0]
            #print(V.shape[0], i, id, w)
            indices_cross_corr[i] = i               #POP_cape has same index with smpl's
            cross_weights_corr[i] = 1
            #print(V.shape[0], i, int(struct.unpack('f', c_file.read(4))[0]), struct.unpack('f', c_file.read(4)))
            #indices_cross_corr[i] = int(struct.unpack('f', c_file.read(4))[0])#float(file.read(4))
            #cross_weights_corr[i] = struct.unpack('f', c_file.read(4))[0]#float(file.read(4))
        #c_file.close()
        #print(indices_cross_corr[0:200])
        #print(max(indices_cross_corr))
        #continue

        weights_corr = np.ones(len(V))

        print('\n===> Iteration %d' % it)
        smpl_utils.optimize_on_vertices(
            model=trans_model,
            vertices=V,
            weights_corr=weights_corr,
            vertices_cross_corr=V,
            indices_cross_corr=indices_cross_corr,
            weights_cross_corr=cross_weights_corr,
            opt_target=opt_target,
            viz=False)


        smpl_utils.optimize_on_vertices(
            model=trans_model,
            vertices=V,
            weights_corr=weights_corr,
            vertices_cross_corr=V,
            indices_cross_corr=indices_cross_corr,
            weights_cross_corr=cross_weights_corr,
            opt_target="trans",
            viz=False)
    
    if (verbose):
        smpl_utils.save_smpl_obj(outputpath+"fitted.obj", trans_model)
    
    data = {'pose': trans_model.pose, 'trans': trans_model.trans, 'betas': trans_model.betas, 'J': trans_model.J_transformed.r}
    output = open(outputpath+'fitted_model.pkl', 'wb')
    pickle.dump(data, output)
    output.close()
    #with open(outputpath+"fitted_model.json", 'wb') as handle:
    #    pickle.dump(data, handle)
    
    pose_out = np.array(trans_model.pose).astype(np.float32)
    pose_out.tofile(outputpath)

    """
    root_j = np.array(trans_model.J_transformed.r[6]).astype(np.float32)    #spine_02
    print("root_j : " , root_j)
    root_j.tofile(root_save_path)
    """

    return True
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit SMPL body to voxels predictions.')
    parser.add_argument('--path', type=str, default=r"E:/RenderPeople/4D-people/rp_regina_4d_001_walking_BLD",
                        help='Path to the 3D skeleton that corresponds to the 3D scan.')
    parser.add_argument('--scan_path', type=str, default=r"",
                        help='Path to the 3D skeleton that corresponds to the 3D scan.')    
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Flag to record intermediate results')
    parser.add_argument('--size', type=int, default=210,
                        help='nb of scans')
    parser.add_argument('--init_flg', type=str, default="False",
                        help='True → initialize with 1st pose , False → following poses')
    parser.add_argument('--start_id', type=int, default=-1,
                        help='if set num , start from that id')
    parser.add_argument('--resampled_flg', type=str, default="False",
                        help='True → use meshes in resampled_meshes , False → use meshes in src_meshes')
    args = parser.parse_args()
    path = args.path
    scan_dir_path = args.scan_path
    if args.init_flg == "True":
        init_flg = True
    if args.init_flg == "False":
        init_flg = False

    if args.start_id == -1:
        start_id = 1
    else :
        start_id = args.start_id

    m = load_model("../../assets/heavy/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")

    #m = load_model("Z:\Human\b20-kitamura\AvatarInTheShell_datasets\SMPLmodel\SMPL_python_v.1.0.0\smpl\models\basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    
    ## init → optimize pose and betas , trans parameter , AfterInit → optimize only pose and trans parameter
    if init_flg == True:
        prev_id = idx = args.start_id

        init_pose_path     = os.path.join(path , "data/smplparams_pop/pose_"  + str(prev_id).zfill(4) + "_init.bin")    #copied from blender
        init_trans_path    = os.path.join(path , "data/smplparams_pop/trans_" + str(prev_id).zfill(4) + "_init.bin")    #copied from blender
        tmp_path           = os.path.join(path , "tmp/")
        output_path        = os.path.join(path , "data/smplparams_pop/pose_"  + str(idx).zfill(4) + ".bin" )
        scan_path          = os.path.join(scan_dir_path ,"smpl_" + str(idx).zfill(4) + ".obj"      )

        opt_target = "pose_and_betas"
        #opt_target = "pose"

        if (FitSMPL(init_pose_path, scan_path, output_path, tmp_path, args.verbose, idx , opt_target , init_flg , init_trans_path , m)):
            prev_id = idx
    elif init_flg == False:
        start_id = prev_id = args.start_id
        first_flg = True
        for idx in tqdm.tqdm(range(start_id,args.size)):
            tmp_path       = os.path.join(path , "tmp/")
            init_pose_path     = os.path.join(path , "data/smplparams_pop/pose_" + str(prev_id).zfill(4) + ".binfitted_model.pkl") #output of first process(init = True)
            output_path        = os.path.join(path , "data/smplparams_pop/pose_" + str(idx).zfill(4) + ".bin")

            scan_path          = os.path.join(scan_dir_path ,"smpl_" + str(idx).zfill(4) + ".obj")

            opt_target = "pose"
            if (FitSMPL(init_pose_path, scan_path, output_path, tmp_path, args.verbose, idx , opt_target , init_flg , None , m , first_flg)):
                prev_id = idx
                first_flg = False

