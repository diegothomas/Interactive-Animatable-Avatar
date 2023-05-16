from math import ceil
import os
from platform import java_ver
from re import A
from string import Template
import struct
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy
import time
import numba
from numba import jit,cuda
from scipy import interpolate
import sys

if __name__== "__main__":
    import meshes_utils

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

def rodrigues2matrix(Rodrigues):
    Rotation = np.identity(3)
    
    theta = np.sqrt(Rodrigues[0]*Rodrigues[0]+Rodrigues[1]*Rodrigues[1]+Rodrigues[2]*Rodrigues[2])  #magnitude of the rotation
    if theta == 0:
        return Rotation
    else:
        u = np.array([Rodrigues[0]/theta ,Rodrigues[1]/theta,Rodrigues[2]/theta]) 
        u_skew = np.array([[0.0,-u[2],u[1]],[u[2],0.0,-u[0]],[-u[1],u[0],0.0]])

        Rotation = np.cos(theta) * np.identity(3) + (1.0 - np.cos(theta))*np.dot(np.array([u]).T,np.array([u])) + np.sin(theta) * u_skew
        return Rotation

def load_smpl(skeleton_path):
    #load smpl pose parameter
    f = open(skeleton_path,"rb")
    skel_bin = f.read()
    smpl = []
    for i in range(24):
        joint = []
        for j in range(3):
            joint.append(struct.unpack("f",skel_bin[4*(3*i+j):4*(3*i+j)+4])[0])
        smpl.append(joint)
    f.close()
    return np.array(smpl)

def load_kintree(kintree_path):
    #load kintree
    f = open(kintree_path,"rb")
    kintree_bin = f.read()
    kintree = []
    for i in range(48):
        kintree.append(struct.unpack("i",kintree_bin[4*i:4*i+4])[0])
    f.close()
    return np.array(kintree)

def load_skinweight_bin(sw_path,vtx_num = 35010):   #vtx_num = 35010 is hardcoding for our task
    """f = open(sw_path,"rb")
    sw_bin = f.read()
    sw_lists = []

    for i in range(vtx_num):
        sw_list = []
        for j in range(24):
            sw_list.append(struct.unpack("f",sw_bin[4*(24*i+j):4*(24*i+j)+4])[0])
        sw_lists.append(sw_list)
    sw = np.array(sw_lists)
    sw = sw.astype(np.float32)
    """
    skin_weight = np.fromfile(sw_path, dtype=np.float32)
    sw = skin_weight.reshape([vtx_num, 24])
    sw = np.round(sw,6)
    return sw

def load_skinweight_npz(sw_path):   #vtx_num = 35010 is hardcoding for our task
    sw_dict = np.load(sw_path)
    sw_pose = sw_dict['pose']
    sw_trans = sw_dict['transl']
    return sw_pose , sw_trans

def load_skinweight(sw_folder_path,testfolder_list=None,Include = False):
    if testfolder_list == None:
        testfolder_list=[]
    sw_paths = glob.glob(sw_folder_path + "/*.bin")
    sw_list = []
    for i,sw_path in enumerate(sw_paths):
        print(i)
        if Include == False:
            sw_list.append(load_skinweight_bin(sw_path))
        elif Include == True and testfolder_list != []:
            if i in testfolder_list:
                sw_list.append(load_skinweight_bin(sw_path))
        elif Include == True and testfolder_list == []:
            print("You set input(\"Include\") to True , but testfolder_list = None")
            print("Please set include to False or use testfolder_list")
            sys.exit()
    sws = np.array(sw_list)
    sws = sws.astype(np.float32)              
    return sws

def Load_joint(path):
    #load joint
    f = open(path,"rb")
    joint_bin = f.read()
    joints = []
    for i in range(24):
        joint = []
        for j in range(3):
            joint.append(struct.unpack("f",joint_bin[4*(3*i+j):4*(3*i+j)+4])[0])
        joints.append(joint)
    f.close()
    joints = np.array(joints)
    return joints

"""
def LoadStarSkeleton(joint_t_path , kinTree_Table ,angle_star = 0.5):
    #load joint_t
    f = open(joint_t_path,"rb")
    joint_t_bin = f.read()
    joints = []
    for i in range(24):
        joint = []
        for j in range(3):
            joint.append(struct.unpack("f",joint_t_bin[4*(3*i+j):4*(3*i+j)+4])[0])
        joints.append(joint)
    f.close()
    joints = np.array(joints)
   
    joints_T = np.zeros((24,4,4))
    for i in range(24):
        joints_T[i] = np.vstack((np.hstack((np.identity(3) , np.array([joints[i]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]])))   #Initialization joint_T as 4*4 matrix

    #Take the A pose
    Angles = np.zeros([24,3])

    #rotate from A-pose tp T-pose
    Angles[1,2] = angle_star
    Angles[2,2] = -angle_star

    rotation = rodrigues2matrix(Angles[0])  #Convert to Rotation matrix

    kinematics = np.zeros([24,4,4])
    kinematics[0] = np.vstack((np.hstack((rotation , np.array([joints[0] - [0.0 ,0.0, 0.0]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 

    jnt0_loc = kinematics[0]
    joints_T[0] = jnt0_loc

    for i in range(1,24):
        rotation = rodrigues2matrix(Angles[i])  #Convert to Rotation matrix

        transfo = np.vstack((np.hstack((rotation , np.array([joints[kinTree_Table[2*i+1]] - joints[kinTree_Table[2*i]]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 
        
        kinematics[kinTree_Table[2*i+1]] = np.dot(kinematics[kinTree_Table[2*i]] , transfo)
        jnt_loc = kinematics[kinTree_Table[2*i+1]]
        joints_T[kinTree_Table[2*i+1]] = jnt_loc
   
    return joints_T
"""

def LoadStarSkeleton(joints, kinTree_Table ,angle_star = 0.5):
    joints_T = np.zeros((24,4,4))
    for i in range(24):
        joints_T[i] = np.vstack((np.hstack((np.identity(3) , np.array([joints[i]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]])))   #Initialization joint_T as 4*4 matrix

    #Take the A pose
    Angles = np.zeros([24,3])

    #rotate from A-pose to T-pose
    Angles[1,2] = angle_star
    Angles[2,2] = -angle_star

    rotation = rodrigues2matrix(Angles[0])  #Convert to Rotation matrix

    kinematics = np.zeros([24,4,4])
    kinematics[0] = np.vstack((np.hstack((rotation , np.array([joints[0] - [0.0 ,0.0, 0.0]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 

    joints_T[0] = kinematics[0]

    for i in range(1,24):
        rotation = rodrigues2matrix(Angles[i])  #Convert to Rotation matrix
        transfo = np.vstack((np.hstack((rotation , np.array([joints[kinTree_Table[2*i+1]] - joints[kinTree_Table[2*i]]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 
        
        kinematics[kinTree_Table[2*i+1]] = np.dot(kinematics[kinTree_Table[2*i]] , transfo)
        joints_T[kinTree_Table[2*i+1]]   = kinematics[kinTree_Table[2*i+1]]
    return joints_T

"""
def LoadSkeleton(pose_path , joint_t_path , kinTree_Table , smpl_fmt = "bin"):
    #load joint_t
    f = open(joint_t_path,"rb")
    joint_t_bin = f.read()
    joints_T = []
    for i in range(24):
        joint_T = []
        for j in range(3):
            joint_T.append(struct.unpack("f",joint_t_bin[4*(3*i+j):4*(3*i+j)+4])[0])
        joints_T.append(joint_T)
    f.close()
    joints_T = np.array(joints_T)

    if smpl_fmt == "bin":
        #load pose
        f = open(pose_path,"rb")
        pose_bin = f.read()
        Angles = []
        for i in range(24):
            Angle = []
            for j in range(3):
                Angle.append(struct.unpack("f",pose_bin[4*(3*i+j):4*(3*i+j)+4])[0])
            Angles.append(Angle)
        f.close()
        Angles = np.array(Angles)

    elif smpl_fmt == "npz":
        smpl_dict = np.load(pose_path)
        Angles = smpl_dict['pose'].reshape(24,3)
        #smpl_trans = smpl_dict['transl']

    print(Angles.shape)

    Angles[10] = np.array([0.0 , 0.0 , 0.0])
    Angles[11] = np.array([0.0 , 0.0 , 0.0])
    Angles[22] = np.array([0.0 , 0.0 , 0.0])
    Angles[23] = np.array([0.0 , 0.0 , 0.0])

    rotation = rodrigues2matrix(Angles[0])

    joints = np.zeros([24,4,4])
    kinematics = np.zeros([24,4,4])
    kinematics[0] = np.vstack((np.hstack((rotation , np.array([joints_T[0] - [0.0 , 0.0 , 0.0]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 

    jnt0_loc = kinematics[0]
    joints[0] = jnt0_loc


    for i in range(1,24):
        rotation = rodrigues2matrix(Angles[i])  #
        transfo = np.vstack((np.hstack((rotation , np.array([joints_T[kinTree_Table[2*i+1]] - joints_T[kinTree_Table[2*i]]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 
        
        kinematics[kinTree_Table[2*i+1]] = np.dot(kinematics[kinTree_Table[2*i]] , transfo)
        jnt_loc = kinematics[kinTree_Table[2*i+1]]
        joints[kinTree_Table[2*i+1]] = jnt_loc
    return joints
"""

def fix_smpl_parts(Angles, fix_hands = False , fix_foots = False , fix_wrists = False ):
    if Angles.shape == (72,):
        if fix_hands:
            Angles[22*3 + 0] = 0.0
            Angles[22*3 + 1] = 0.0
            Angles[22*3 + 2] = 0.0

            Angles[23*3 + 0] = 0.0
            Angles[23*3 + 1] = 0.0
            Angles[23*3 + 2] = 0.0

        if fix_foots:
            Angles[10*3 + 0] = 0.0
            Angles[10*3 + 1] = 0.0
            Angles[10*3 + 2] = 0.0

            Angles[11*3 + 0] = 0.0
            Angles[11*3 + 1] = 0.0
            Angles[11*3 + 2] = 0.0

        if fix_wrists:
            Angles[20*3 + 0] = 0.0
            Angles[20*3 + 1] = 0.0
            Angles[20*3 + 2] = 0.0

            Angles[21*3 + 0] = 0.0
            Angles[21*3 + 1] = 0.0
            Angles[21*3 + 2] = 0.0
    elif Angles.shape == (24,3):
        if fix_hands:
            Angles[22][0] = 0.0
            Angles[22][1] = 0.0
            Angles[22][2] = 0.0

            Angles[23][0] = 0.0
            Angles[23][1] = 0.0
            Angles[23][2] = 0.0

        if fix_foots:
            Angles[10][0] = 0.0
            Angles[10][1] = 0.0
            Angles[10][2] = 0.0

            Angles[11][0] = 0.0
            Angles[11][1] = 0.0
            Angles[11][2] = 0.0

        if fix_wrists:
            Angles[20][0] = 0.0
            Angles[20][1] = 0.0
            Angles[20][2] = 0.0

            Angles[21][0] = 0.0
            Angles[21][1] = 0.0
            Angles[21][2] = 0.0
    else:
        print("Angles.shape : " , Angles.shape)
        raise AssertionError("Angles.shape is strange")
    return Angles

def LoadSkeleton(Angles , joints_T , kinTree_Table , fix_hands = False , fix_foots = False , fix_wrists = False):
    if fix_hands:
        Angles[22] = np.array([0.0 , 0.0 , 0.0])
        Angles[23] = np.array([0.0 , 0.0 , 0.0])
    if fix_foots:
        Angles[10] = np.array([0.0 , 0.0 , 0.0])
        Angles[11] = np.array([0.0 , 0.0 , 0.0])
    if fix_wrists:
        Angles[20] = np.array([0.0 , 0.0 , 0.0])
        Angles[21] = np.array([0.0 , 0.0 , 0.0])

    rotation = rodrigues2matrix(Angles[0])

    joints = np.zeros([24,4,4])
    kinematics = np.zeros([24,4,4])
    kinematics[0] = np.vstack((np.hstack((rotation , np.array([joints_T[0] - [0.0 , 0.0 , 0.0]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 

    joints[0] = kinematics[0]

    for i in range(1,24):
        rotation = rodrigues2matrix(Angles[i])  #
        transfo = np.vstack((np.hstack((rotation , np.array([joints_T[kinTree_Table[2*i+1]] - joints_T[kinTree_Table[2*i]]]).T)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 
        
        kinematics[kinTree_Table[2*i+1]] = np.dot(kinematics[kinTree_Table[2*i]] , transfo)
        joints[kinTree_Table[2*i+1]] = kinematics[kinTree_Table[2*i+1]]
    return joints

"""
def transposeMatrix(m):
    return list(map(list,zip(*m)))

def getMatrixMinor(m,i,j):
    print()
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]
    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors
"""

def InverseTransfo(joint):
    rot = joint[:3,:3]
    trans = joint[:3,3]
    
    trans = np.array([trans])
    #rot_inv = np.linalg.inv(rot)
    rot_inv = rot.T
    #rot_inv = getMatrixInverse(rot)
    #if np.any(np.abs(rot_inv - rot_T) > 0.1):
    #    print("rot_inv : " , rot_inv)
    #    print("rot T   : " , rot.T)
    #    sys.exit()
    trans_inv = np.dot(-rot_inv , trans.T)

    res = np.vstack((np.hstack((rot_inv , trans_inv)) , np.array([[0.0, 0.0, 0.0, 1.0]]))) 
    return res

@cuda.jit
def InverseTranfo_gpu(joint_out,joint):
    rot = joint[:3,:3]
    trans = joint[:3,3]
    rot_inv = rot.T
    trans_inv_tmp = cuda.local.array((3,1), np.float32)
    mns_rot_inv =  cuda.local.array((3,3), np.float32)
    my_mns3_3(mns_rot_inv , rot_inv)
    my_dot3_3and3(trans_inv_tmp ,mns_rot_inv , trans.T)
    my_homo(joint_out , rot_inv , trans_inv_tmp)

def SkinMeshLBS(vertices, skinWeights, skeleton, vtx_num, inverse = False):
    vertices_copy = copy.deepcopy(vertices)
    for v in range(vtx_num):
        if np.all(vertices[v] == 0.0):
            continue
        point = np.hstack((vertices[v] , np.array([1.0])))
        vtx  = np.zeros(4)

        transfo = np.zeros((4,4))
        sum_weights = 0.0
        for j in range(24):
            if skinWeights[v][j] > 0.0:
                if inverse:
                    transfo = transfo + skinWeights[v][j] * InverseTransfo(skeleton[j])
                else:
                    transfo = transfo + skinWeights[v][j] * skeleton[j]
                sum_weights += skinWeights[v][j]
        transfo = transfo / sum_weights
        """
        if inverse:
            transfo = InverseTransfo(transfo)
        """
        vtx = np.dot(transfo , point)
        vertices_copy[v] = vtx[0:3]
    return vertices_copy

def SkinMeshLBS_gpu(vtx, skinWeights, skeleton, vtx_num, inverse = False):
    vtx = np.array(vtx)
    vtx = np.hstack((vtx , np.ones((vtx_num ,1))))
    transfo = np.zeros((vtx_num,4,4))

    threads_per_block = 64  #i.e) block dim
    blocks_per_grid = int(divUp(vtx_num,threads_per_block))     #i.e) grid dim  
    vtx_out = np.zeros((vtx_num , 3) , dtype= "float32")
    
    SkinMeshLBS_cuda[blocks_per_grid,threads_per_block](vtx_out, vtx_num, vtx, skinWeights, skeleton, inverse , transfo)
    return vtx_out

@cuda.jit
def SkinMeshLBS_cuda(vtx_out, vtx_num , vtx, skinWeights, skeleton, inverse , transfo):
    i = cuda.grid(1)

    if i < vtx_num:
        point_vtx = vtx[i]

        transfo = cuda.local.array((4,4), np.float32)
        transfo_sum = cuda.local.array((4,4), np.float32)
        transfo_tmp = cuda.local.array((4,4), np.float32)

        sum_weights = 0.0
        for j in range(24):
            if skinWeights[i][j] > 0.0:
                if inverse:
                    trans_inv =  cuda.local.array((4,4), np.float32)
                    #InverseTranfo_gpu(trans_inv , skeleton[j])
                    #InverseTranfo_gpu
                    rot = skeleton[j][:3,:3]
                    trans = skeleton[j][:3,3]
                    rot_inv = rot.T
                    trans_inv_tmp = cuda.local.array((3,1), np.float32)
                    mns_rot_inv =  cuda.local.array((3,3), np.float32)
                    my_mns3_3(mns_rot_inv , rot_inv)
                    my_dot3_3and3(trans_inv_tmp ,mns_rot_inv , trans.T)
                    my_homo(trans_inv , rot_inv , trans_inv_tmp)

                    my_mul4_4(transfo_tmp ,  trans_inv ,skinWeights[i][j])
                    my_add4_4(transfo_sum , transfo , transfo_tmp)
                else:
                    my_mul4_4(transfo_tmp ,  skeleton[j] ,skinWeights[i][j])
                    my_add4_4(transfo_sum , transfo , transfo_tmp)
                sum_weights += skinWeights[i][j]
                transfo = transfo_sum
        my_div4_4(transfo , transfo_sum , sum_weights)

        vtx_out_tmp =  cuda.local.array(4, np.float32)
        my_dot4_4and4(vtx_out_tmp , transfo , point_vtx)

        vtx_out[i][0] = vtx_out_tmp[0]
        vtx_out[i][1] = vtx_out_tmp[1]
        vtx_out[i][2] = vtx_out_tmp[2]

@cuda.jit
def SkinMeshLBS_cuda_sub(vtx_out, vtx, skinWeights, skeleton, inverse):
    point_vtx = vtx

    transfo = cuda.local.array((4,4), np.float32)
    transfo_sum = cuda.local.array((4,4), np.float32)
    transfo_tmp = cuda.local.array((4,4), np.float32)

    sum_weights = 0.0
    for j in range(24):
        if skinWeights[j] > 0.0:
            if inverse:
                trans_inv =  cuda.local.array((4,4), np.float32)
                #InverseTranfo_gpu(trans_inv , skeleton[j])
                #InverseTranfo_gpu
                rot = skeleton[j][:3,:3]
                trans = skeleton[j][:3,3]
                rot_inv = rot.T
                trans_inv_tmp = cuda.local.array((3,1), np.float32)
                mns_rot_inv =  cuda.local.array((3,3), np.float32)
                my_mns3_3(mns_rot_inv , rot_inv)
                my_dot3_3and3(trans_inv_tmp ,mns_rot_inv , trans.T)
                my_homo(trans_inv , rot_inv , trans_inv_tmp)

                my_mul4_4(transfo_tmp ,  trans_inv ,skinWeights[j])
                my_add4_4(transfo_sum , transfo , transfo_tmp)
            else:
                my_mul4_4(transfo_tmp ,  skeleton[j] ,skinWeights[j])
                my_add4_4(transfo_sum , transfo , transfo_tmp)
            sum_weights += skinWeights[j]
            transfo = transfo_sum
    my_div4_4(transfo , transfo_sum , sum_weights)

    vtx_out_tmp =  cuda.local.array(4, np.float32)
    my_dot4_4and4(vtx_out_tmp , transfo , point_vtx)

    vtx_out[0] = vtx_out_tmp[0]
    vtx_out[1] = vtx_out_tmp[1]
    vtx_out[2] = vtx_out_tmp[2]

def SkinMeshandBasisLBS_cpu(vertices, t_basis, skinWeights, skeleton, vtx_num, inverse):
    #vtx_out = np.zeros((vtx_num , 3))
    #t_basis_out = np.zeros((3 , vtx_num , 3))
    vertices_copy = copy.deepcopy(vertices)
    t_basis_copy = copy.deepcopy(t_basis)
    for i in range(vtx_num):
        if np.all(vertices[i] == 0.0):
            continue
            #vtx_out[i] = vtx[i]
            #t_basis_out[0][i] = t_basis[0][i]
        point_vtx = np.hstack((vertices[i] , np.array([1.0])))
        point_f1 = np.hstack((t_basis[i][0] , np.array([1.0])))
        point_f2 = np.hstack((t_basis[i][1] , np.array([1.0])))
        point_f3 = np.hstack((t_basis[i][2] , np.array([1.0])))
        
        transfo = np.zeros((4,4))
        sum_weights = 0.0
        for j in range(24):
            if skinWeights[i][j] > 0.0:
                if inverse:
                    transfo = transfo + skinWeights[i][j] * InverseTransfo(skeleton[j])
                else:
                    transfo = transfo + skinWeights[i][j] * skeleton[j]
                sum_weights += skinWeights[i][j]
        transfo = transfo / sum_weights
        #if inverse:
        #    transfo = InverseTransfo(transfo)
        vtx_out_tmp  = np.zeros(4)
        vtx_out_tmp = np.dot(transfo , point_vtx)
        vertices_copy[i] = vtx_out_tmp[0:3]
        vtx_out_tmp  = np.zeros(4)
        vtx_out_tmp = np.dot(transfo , point_f1)
        t_basis_copy[i][0] = vtx_out_tmp[0:3]
        vtx_out_tmp  = np.zeros(4)
        vtx_out_tmp = np.dot(transfo , point_f2)
        t_basis_copy[i][1] = vtx_out_tmp[0:3]
        vtx_out_tmp  = np.zeros(4)
        vtx_out_tmp = np.dot(transfo , point_f3)
        t_basis_copy[i][2] = vtx_out_tmp[0:3]
    return vertices_copy , t_basis_copy

def SkinMeshandBasisLBS_gpu(vtx, basis, skinWeights, skeleton, vtx_num, inverse = False , additional_vtx = None):
    vtx = np.array(vtx)
    vtx = np.hstack((vtx , np.ones((vtx_num ,1))))
    basis_f1 = np.hstack((basis[:,0] , np.ones((vtx_num ,1))))
    basis_f2 = np.hstack((basis[:,1] , np.ones((vtx_num ,1))))
    basis_f3 = np.hstack((basis[:,2] , np.ones((vtx_num ,1))))
    basis = np.stack([basis_f1,basis_f2,basis_f3],axis = 1)
    if additional_vtx is not None:
        additional_vtx = np.array(additional_vtx)
        additional_vtx = np.hstack((additional_vtx , np.ones((vtx_num ,1))))

    threads_per_block = 64  #i.e) block dim
    blocks_per_grid = int(divUp(vtx_num,threads_per_block))     #i.e) grid dim  
    vtx_out = np.zeros((vtx_num , 3) , dtype= "float32")
    basis_out = np.zeros((vtx_num , 3 , 3), dtype= "float32")
    if additional_vtx is not None:
        additional_vtx_out = np.zeros((vtx_num , 3) , dtype= "float32")
        SkinMeshandBasisLBS_additional_vtx_cuda[blocks_per_grid,threads_per_block](vtx_out, basis_out , additional_vtx_out, vtx_num, vtx, basis, additional_vtx,  skinWeights, skeleton, inverse )
        return vtx_out , basis_out , additional_vtx_out
    else:
        SkinMeshandBasisLBS_cuda[blocks_per_grid,threads_per_block](vtx_out, basis_out , vtx_num, vtx, basis, skinWeights, skeleton, inverse )
        return vtx_out , basis_out
    
@cuda.jit(device=True)
def my_mul4(res,a,b):
    res[0] = a[0]*b
    res[1] = a[1]*b
    res[2] = a[2]*b
    res[3] = a[3]*b

@cuda.jit(device=True)
def my_mul4_4(res,a,b):
    for i in range(4):
        for j in range(4):
            res[i][j] = a[i][j]*b

@cuda.jit(device=True)
def my_div3_3(res,a,b):
    for i in range(3):
        for j in range(3):
            res[i][j] = a[i][j]/b

@cuda.jit(device=True)
def my_div4_4(res,a,b):
    for i in range(4):
        for j in range(4):
            res[i][j] = a[i][j]/b

@cuda.jit(device=True)
def my_add3(res,a,b):
    for i in range(3):
        res[i] = a[i] + b[i]

@cuda.jit(device=True)
def my_add3_nonzero(res,a,b):
    for i in range(3):
        if (a[i] != 0.0) and (b[i] != 0.0):
            res[i] = a[i] + b[i]
        else:
            res[i] = 0.0

@cuda.jit(device=True)
def my_add3_3(res,a,b):
    for i in range(3):
        for j in range(3):
            res[i][j] = a[i][j] + b[i][j]

@cuda.jit(device=True)
def my_add4_4(res,a,b):
    for i in range(4):
        for j in range(4):
            res[i][j] = a[i][j] + b[i][j]

@cuda.jit(device=True)
def my_dot3_3and3_3(res,a,b):
    for i in range(3):
        for j in range(3):
            res[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j] 

@cuda.jit(device=True)
def my_dot3_3and3(res,a,b):
    res[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2]
    res[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2]
    res[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2]

@cuda.jit(device=True)
def my_dot4_4and4(res,a,b):
    res[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2] + a[0][3]*b[3]
    res[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2] + a[1][3]*b[3]
    res[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2] + a[2][3]*b[3]
    res[3] = a[3][0]*b[0] + a[3][1]*b[1] + a[3][2]*b[2] + a[3][3]*b[3]

@cuda.jit(device=True)
def my_homo(res,rot,trans):
    for i in range(3):
        for j in range(3):
            res[i][j] = rot[i][j]
    for i in range(3):
        res[i][3] = trans[i][0]
    for j in range(3):
        res[3][j] = 0.0
    res[3][3] = 1.0
    

@cuda.jit(device=True)
def my_mns3_3(res,a):
    for i in range(3):
        for j in range(3):
            res[i][j] = -a[i][j]
 
@cuda.jit(device=True)
def my_mns3_3_b(res,a,b):
    for i in range(3):
        for j in range(3):
            res[i][j] = a[i][j] - b[i][j]

@cuda.jit
def SkinMeshandBasisLBS_cuda(vtxs_out, basis_out , vtx_num , vtxs, basis, skinWeights, skeleton, inverse ):
    i = cuda.grid(1)

    if i < vtx_num:
        vtx = vtxs[i]
        point_f1 = basis[i][0]
        point_f2 = basis[i][1]
        point_f3 = basis[i][2]

        transfo = cuda.local.array((4,4), np.float32)
        transfo_sum = cuda.local.array((4,4), np.float32)
        transfo_tmp = cuda.local.array((4,4), np.float32)

        sum_weights = 0.0
        for j in range(24):
            if skinWeights[i][j] > 0.0:
                if inverse:
                    trans_inv =  cuda.local.array((4,4), np.float32)
                    #InverseTranfo_gpu(trans_inv , skeleton[j])
                    #InverseTranfo_gpu
                    rot = skeleton[j][:3,:3]
                    trans = skeleton[j][:3,3]
                    rot_inv = rot.T
                    trans_inv_tmp = cuda.local.array((3,1), np.float32)
                    mns_rot_inv =  cuda.local.array((3,3), np.float32)
                    my_mns3_3(mns_rot_inv , rot_inv)
                    my_dot3_3and3(trans_inv_tmp ,mns_rot_inv , trans.T)
                    my_homo(trans_inv , rot_inv , trans_inv_tmp)

                    my_mul4_4(transfo_tmp ,  trans_inv ,skinWeights[i][j])
                    my_add4_4(transfo_sum , transfo , transfo_tmp)
                else:
                    my_mul4_4(transfo_tmp ,  skeleton[j] ,skinWeights[i][j])
                    my_add4_4(transfo_sum , transfo , transfo_tmp)
                sum_weights += skinWeights[i][j]
                transfo = transfo_sum
        my_div4_4(transfo , transfo_sum , sum_weights)

        vtx_out_tmp =  cuda.local.array(4, np.float32)
        basis_out_tmp =  cuda.local.array(4, np.float32)
        
        my_dot4_4and4(vtx_out_tmp , transfo , vtx)

        vtxs_out[i][0] = vtx_out_tmp[0]
        vtxs_out[i][1] = vtx_out_tmp[1]
        vtxs_out[i][2] = vtx_out_tmp[2]

        my_dot4_4and4(basis_out_tmp, transfo , point_f1)
        basis_out[i][0][0] = basis_out_tmp[0]
        basis_out[i][0][1] = basis_out_tmp[1]
        basis_out[i][0][2] = basis_out_tmp[2]
        my_dot4_4and4(basis_out_tmp, transfo , point_f2)
        basis_out[i][1][0] = basis_out_tmp[0]
        basis_out[i][1][1] = basis_out_tmp[1]
        basis_out[i][1][2] = basis_out_tmp[2]
        my_dot4_4and4(basis_out_tmp, transfo , point_f3)
        basis_out[i][2][0] = basis_out_tmp[0]
        basis_out[i][2][1] = basis_out_tmp[1]
        basis_out[i][2][2] = basis_out_tmp[2]


@cuda.jit
def SkinMeshandBasisLBS_additional_vtx_cuda(vtxs_out, basis_out , additional_vtxs_out, vtx_num , vtxs, basis, additional_vtxs, skinWeights, skeleton, inverse ):
    i = cuda.grid(1)

    if i < vtx_num:
        vtx = vtxs[i]
        point_f1 = basis[i][0]
        point_f2 = basis[i][1]
        point_f3 = basis[i][2]
        additional_vtx = additional_vtxs[i]

        transfo = cuda.local.array((4,4), np.float32)
        transfo_sum = cuda.local.array((4,4), np.float32)
        transfo_tmp = cuda.local.array((4,4), np.float32)

        sum_weights = 0.0
        for j in range(24):
            if skinWeights[i][j] > 0.0:
                if inverse:
                    trans_inv =  cuda.local.array((4,4), np.float32)
                    #InverseTranfo_gpu(trans_inv , skeleton[j])
                    #InverseTranfo_gpu
                    rot = skeleton[j][:3,:3]
                    trans = skeleton[j][:3,3]
                    rot_inv = rot.T
                    trans_inv_tmp = cuda.local.array((3,1), np.float32)
                    mns_rot_inv =  cuda.local.array((3,3), np.float32)
                    my_mns3_3(mns_rot_inv , rot_inv)
                    my_dot3_3and3(trans_inv_tmp ,mns_rot_inv , trans.T)
                    my_homo(trans_inv , rot_inv , trans_inv_tmp)

                    my_mul4_4(transfo_tmp ,  trans_inv ,skinWeights[i][j])
                    my_add4_4(transfo_sum , transfo , transfo_tmp)
                else:
                    my_mul4_4(transfo_tmp ,  skeleton[j] ,skinWeights[i][j])
                    my_add4_4(transfo_sum , transfo , transfo_tmp)
                sum_weights += skinWeights[i][j]
                transfo = transfo_sum
        my_div4_4(transfo , transfo_sum , sum_weights)

        vtx_out_tmp =  cuda.local.array(4, np.float32)
        basis_out_tmp =  cuda.local.array(4, np.float32)
        additional_vtx_out_tmp =  cuda.local.array(4, np.float32)
        
        my_dot4_4and4(vtx_out_tmp , transfo , vtx)
        my_dot4_4and4(additional_vtx_out_tmp , transfo , additional_vtx)

        vtxs_out[i][0] = vtx_out_tmp[0]
        vtxs_out[i][1] = vtx_out_tmp[1]
        vtxs_out[i][2] = vtx_out_tmp[2]

        additional_vtxs_out[i][0] = additional_vtx_out_tmp[0]
        additional_vtxs_out[i][1] = additional_vtx_out_tmp[1]
        additional_vtxs_out[i][2] = additional_vtx_out_tmp[2]

        my_dot4_4and4(basis_out_tmp, transfo , point_f1)
        basis_out[i][0][0] = basis_out_tmp[0]
        basis_out[i][0][1] = basis_out_tmp[1]
        basis_out[i][0][2] = basis_out_tmp[2]
        my_dot4_4and4(basis_out_tmp,transfo , point_f2)
        basis_out[i][1][0] = basis_out_tmp[0]
        basis_out[i][1][1] = basis_out_tmp[1]
        basis_out[i][1][2] = basis_out_tmp[2]
        my_dot4_4and4(basis_out_tmp,transfo , point_f3)
        basis_out[i][2][0] = basis_out_tmp[0]
        basis_out[i][2][1] = basis_out_tmp[1]
        basis_out[i][2][2] = basis_out_tmp[2]

@cuda.jit(device=True)
def SkinMeshandBasisLBS_cuda_sub(vtx_out, t_basis_out , vtx, t_basis, skinWeights, skeleton, inverse ):
    point_vtx = vtx
    point_f1 = t_basis[0]
    point_f2 = t_basis[1]
    point_f3 = t_basis[2]

    transfo = cuda.local.array((4,4), np.float32)
    transfo_sum = cuda.local.array((4,4), np.float32)
    transfo_tmp = cuda.local.array((4,4), np.float32)

    sum_weights = 0.0
    for j in range(24):
        if skinWeights[j] > 0.0:
            if inverse:
                trans_inv =  cuda.local.array((4,4), np.float32)
                #InverseTranfo_gpu(trans_inv , skeleton[j])
                #InverseTranfo_gpu
                rot = skeleton[j][:3,:3]
                trans = skeleton[j][:3,3]
                rot_inv = rot.T
                trans_inv_tmp = cuda.local.array((3,1), np.float32)
                mns_rot_inv =  cuda.local.array((3,3), np.float32)
                my_mns3_3(mns_rot_inv , rot_inv)
                my_dot3_3and3(trans_inv_tmp ,mns_rot_inv , trans.T)
                my_homo(trans_inv , rot_inv , trans_inv_tmp)

                my_mul4_4(transfo_tmp ,  trans_inv ,skinWeights[j])
                my_add4_4(transfo_sum , transfo , transfo_tmp)
            else:
                my_mul4_4(transfo_tmp ,  skeleton[j] ,skinWeights[j])
                my_add4_4(transfo_sum , transfo , transfo_tmp)
            sum_weights += skinWeights[j]
            transfo = transfo_sum
    my_div4_4(transfo , transfo_sum , sum_weights)

    vtx_out_tmp =  cuda.local.array(4, np.float32)
    my_dot4_4and4(vtx_out_tmp , transfo , point_vtx)

    vtx_out[0] = vtx_out_tmp[0]
    vtx_out[1] = vtx_out_tmp[1]
    vtx_out[2] = vtx_out_tmp[2]

    my_dot4_4and4(vtx_out_tmp, transfo , point_f1)
    t_basis_out[0][0] = vtx_out_tmp[0]
    t_basis_out[0][1] = vtx_out_tmp[1]
    t_basis_out[0][2] = vtx_out_tmp[2]
    my_dot4_4and4(vtx_out_tmp,transfo , point_f2)
    t_basis_out[1][0] = vtx_out_tmp[0]
    t_basis_out[1][1] = vtx_out_tmp[1]
    t_basis_out[1][2] = vtx_out_tmp[2]
    my_dot4_4and4(vtx_out_tmp,transfo , point_f3)
    t_basis_out[2][0] = vtx_out_tmp[0]
    t_basis_out[2][1] = vtx_out_tmp[1]
    t_basis_out[2][2] = vtx_out_tmp[2]


@cuda.jit(device=True)
def SkinMeshandBasisLBS_additionalVtx_cuda_sub(vtx_out , vtx_out2 , basis_out , vtx , vtx2 , in_basis, skinWeights, skeleton, inverse ):
    point_vtx = vtx
    point_vtx2 = vtx2
    point_f1 = in_basis[0]
    point_f2 = in_basis[1]
    point_f3 = in_basis[2]

    transfo = cuda.local.array((4,4), np.float32)
    transfo_sum = cuda.local.array((4,4), np.float32)
    transfo_tmp = cuda.local.array((4,4), np.float32)

    sum_weights = 0.0
    for j in range(24):
        if skinWeights[j] > 0.0:
            if inverse:
                trans_inv =  cuda.local.array((4,4), np.float32)
                #InverseTranfo_gpu(trans_inv , skeleton[j])
                #InverseTranfo_gpu
                rot = skeleton[j][:3,:3]
                trans = skeleton[j][:3,3]
                rot_inv = rot.T
                trans_inv_tmp = cuda.local.array((3,1), np.float32)
                mns_rot_inv =  cuda.local.array((3,3), np.float32)
                my_mns3_3(mns_rot_inv , rot_inv)
                my_dot3_3and3(trans_inv_tmp ,mns_rot_inv , trans.T)
                my_homo(trans_inv , rot_inv , trans_inv_tmp)

                my_mul4_4(transfo_tmp ,  trans_inv ,skinWeights[j])
                my_add4_4(transfo_sum , transfo , transfo_tmp)
            else:
                my_mul4_4(transfo_tmp ,  skeleton[j] ,skinWeights[j])
                my_add4_4(transfo_sum , transfo , transfo_tmp)
            sum_weights += skinWeights[j]
            transfo = transfo_sum
    my_div4_4(transfo , transfo_sum , sum_weights)

    vtx_out_tmp =  cuda.local.array(4, np.float32)
    vtx_out2_tmp=  cuda.local.array(4, np.float32)
    
    my_dot4_4and4(vtx_out_tmp  , transfo , point_vtx)
    my_dot4_4and4(vtx_out2_tmp , transfo , point_vtx2)

    vtx_out[0] = vtx_out_tmp[0]
    vtx_out[1] = vtx_out_tmp[1]
    vtx_out[2] = vtx_out_tmp[2]

    vtx_out2[0] = vtx_out2_tmp[0]
    vtx_out2[1] = vtx_out2_tmp[1]
    vtx_out2[2] = vtx_out2_tmp[2]
    
    my_dot4_4and4(vtx_out_tmp, transfo , point_f1)
    basis_out[0][0] = vtx_out_tmp[0]
    basis_out[0][1] = vtx_out_tmp[1]
    basis_out[0][2] = vtx_out_tmp[2]
    my_dot4_4and4(vtx_out_tmp, transfo , point_f2)
    basis_out[1][0] = vtx_out_tmp[0]
    basis_out[1][1] = vtx_out_tmp[1]
    basis_out[1][2] = vtx_out_tmp[2]
    my_dot4_4and4(vtx_out_tmp, transfo , point_f3)
    basis_out[2][0] = vtx_out_tmp[0]
    basis_out[2][1] = vtx_out_tmp[1]
    basis_out[2][2] = vtx_out_tmp[2]


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

if __name__ == "__main__":
    data_depend_surface_savepath = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Whole\Template-star-0.015-0.05\Template_skin.ply"
    dds_vtx , dds_nml , dds_rgb , dds_face , dds_vtx_num , dds_face_num = meshes_utils.load_ply(data_depend_surface_savepath)

    pose_path = r"Z:\Human\b20-kitamura\AvatarInTheShell_datasets\test_motion_list\test_motion_list\motion_lists\gLO_sBM_cAll_d14_mLO1_ch05\seqs\pose_00160.bin"
    sw_surface_path = "../../assets/heavy/weights_template.bin"

    kintree                 = load_kintree("../../assets/kintree.bin")
    skeleton                = np.zeros([24,4,4])
    smpl_param              = load_smpl(pose_path)                                      
    Tshapecoarsejoints      = Load_joint("../../assets/Tshapecoarsejoints.bin")
    joints                  = LoadSkeleton(smpl_param , Tshapecoarsejoints , kintree , fix_hands = False , fix_foots = False , fix_wrists = False )     #pose (Angle)
    joints_smpl             = LoadStarSkeleton(Tshapecoarsejoints,kintree,0.5)    #smpl t_pose (position)

    for j in range(24):
            skeleton[j] = np.dot(joints_smpl[j] , InverseTransfo(joints[j]))  
    
    skinWeights = load_skinweight_bin(sw_surface_path , 6890)
    posedsurface_vtx = SkinMeshLBS(dds_vtx, skinWeights, skeleton, dds_vtx_num, True)
    meshes_utils.save_ply("debug.ply" , posedsurface_vtx , None , None , dds_face , dds_vtx_num , dds_face_num)

    """
    data_depend_surface_savepath = r"D:\Data\Human\HUAWEI\Iwamoto\data\data_depend_surface.ply"
    
    dds_vtx , dds_nml , dds_rgb , dds_face , dds_vtx_num , dds_face_num = meshes_utils.load_ply(data_depend_surface_savepath)
    
    skeleton = np.zeros([24,4,4])
    joint_t_path      = r"D:\Data\Human\HUAWEI\Iwamoto\data\smplparams_centered/T_joints.bin"
    kintree_path      = r"D:\Data\Human\Template-star-0.015\kintree.bin"
    kintree = load_kintree(kintree_path)
    
    joints = LoadSkeleton(pose_path , joint_t_path , kintree , "bin")   #pose (Angle)   #pose to world
    joint_T = LoadStarSkeleton(joint_t_path,kintree,0.5)  #data depend t_pose (position)

    for j in range(24):
        skeleton[j] = np.dot(joint_T[j] , InverseTransfo(joints[j]))    #joint_T(=data depend t_pose) -> joints(=posed)    Canonical→world→Pose
    
    #posed_vtx , posed_basis_f1= SkinMeshLBS_gpu(dds_vtx, t_basis, skinWeights, skeleton, dds_vtx_num, True)
    """


