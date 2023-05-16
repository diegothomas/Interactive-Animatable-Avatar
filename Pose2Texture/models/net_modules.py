from turtle import forward
import cv2
import torch
from torch import nn , Tensor
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
import os
from natsort.natsort import natsorted
import glob
import struct
import pandas as pd
import sys

###LOSS#################################################################
class SSIMLoss(Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        """
        Computes the structural similarity (SSIM) index map between two images.

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:

        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:

        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:

        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d

"""
class HistLoss(Module):
    def __init__(self, bin = 100) -> None:
        \"""Computes the Histgram loss between two images\"""

        super().__init__()
        self.bin = bin
    
    def forward(self,  x: Tensor, y: Tensor):
        batch_size = x.shape[0]
        assert batch_size == y.shape[0] , "batch_size are different"
        running_loss = 0.0
        for i in range(batch_size):
            x_hist_r = torch.histc(x[i,0,:,:] , self.bin)
            y_hist_r = torch.histc(y[i,0,:,:] , self.bin)
            loss_r = torch.sum((x_hist_r - y_hist_r)**2)
            loss_r = loss_r / self.bin
            running_loss += loss_r

            x_hist_g = torch.histc(x[i,1,:,:] , self.bin)
            y_hist_g = torch.histc(y[i,1,:,:] , self.bin)
            loss_g = torch.sum((x_hist_g - y_hist_g)**2)
            loss_g = loss_g / self.bin
            running_loss += loss_g

            x_hist_b = torch.histc(x[i,2,:,:] , self.bin)
            y_hist_b = torch.histc(y[i,2,:,:] , self.bin)
            loss_b = torch.sum((x_hist_b - y_hist_b)**2)
            loss_b = loss_b / self.bin
            running_loss += loss_b

        return running_loss / batch_size
"""

class SoftHistLoss(nn.Module):
    def __init__(self, device , bins = 10, min = 0, max = 1, sigma = 3*50 , dis = "l1" ):
        super(SoftHistLoss, self).__init__()
        self.device = device
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = (float(min) + self.delta * (torch.arange(bins).float() + 0.5)).to(device)
        if dis == "l1":
            self.dis = nn.L1Loss()
        else:
            self.dis = nn.MSELoss()

    def forward(self, x,y):
        batch_size = x.shape[0]
        assert batch_size == y.shape[0] , "batch_size are different"
        running_loss = 0.0
        for i in range(batch_size):
            for j in range(3):
                #histgram about a
                a = x[i,j,:,:].flatten()
                a = torch.unsqueeze(a, 0) - torch.unsqueeze(self.centers, 1)
                a = torch.sigmoid(self.sigma * (a + self.delta/2)) - torch.sigmoid(self.sigma * (a - self.delta/2))
                a = a.sum(dim=1)

                #histgram about b
                b = y[i,j,:,:].flatten()
                b = torch.unsqueeze(b, 0) - torch.unsqueeze(self.centers, 1)
                b = torch.sigmoid(self.sigma * (b + self.delta/2)) - torch.sigmoid(self.sigma * (b - self.delta/2))
                b = b.sum(dim=1)

                running_loss += self.dis(a,b)
        loss = running_loss / batch_size
        return loss * 1e-04

########################################################################

def load_paths(folder_path):
    f = open(folder_path,"r")
    folder = f.read()
    folder_list = folder.split("\n")
    if folder_list[-1] == "":
        folder_list = folder_list[:-1]

    return folder_list

def load_smpl_bin(smpl_path , fix_hands = False , fix_foots = False , fix_wrists = False):
    f = open(smpl_path,"rb")
    skel_bin = f.read()
    smpl_list = []
    for i in range(24*3):
        smpl_list.append(struct.unpack("f",skel_bin[4*i:4*i+4])[0])

    smpl = np.array(smpl_list)
    smpl = smpl.astype(np.float32)

    if fix_hands:
        for j in range(3):
            smpl[22*3 + j] = 0.0
            smpl[23*3 + j] = 0.0
    if fix_foots:
        for j in range(3):
            smpl[10*3 + j] = 0.0
            smpl[11*3 + j] = 0.0
    if fix_wrists:
        for j in range(3):
            smpl[20*3 + j] = 0.0
            smpl[21*3 + j] = 0.0
    return smpl


def load_smpl_pkl(smpl_path , fix_hands = False , fix_foots = False , fix_wrists = False):
    smpl_pkl = pd.read_pickle(smpl_path)

    smpl_pose = np.array(smpl_pkl['pose'])
    smpl_pose = smpl_pose.astype(np.float32)

    smpl_beta = np.array(smpl_pkl['betas'])
    smpl_beta = smpl_beta.astype(np.float32)

    smpl_trans = np.array(smpl_pkl['trans'])
    smpl_trans = smpl_trans.astype(np.float32)

    if fix_hands:
        for j in range(3):
            smpl_pose[22*3 + j] = 0.0
            smpl_pose[23*3 + j] = 0.0
    if fix_foots:
        for j in range(3):
            smpl_pose[10*3 + j] = 0.0
            smpl_pose[11*3 + j] = 0.0
    if fix_wrists:
        for j in range(3):
            smpl_pose[20*3 + j] = 0.0
            smpl_pose[21*3 + j] = 0.0
    return smpl_pose , smpl_beta , smpl_trans

def load_smpl_npz(smpl_path , fix_hands = False , fix_foots = False , fix_wrists = False):   #vtx_num = 35010 is hardcoding for our task
    smpl_dict = np.load(smpl_path)
    smpl_pose = smpl_dict['pose']
    smpl_trans = smpl_dict['transl']

    if fix_hands:
        for j in range(3):
            smpl_pose[22*3 + j] = 0.0
            smpl_pose[23*3 + j] = 0.0
    if fix_foots:
        for j in range(3):
            smpl_pose[10*3 + j] = 0.0
            smpl_pose[11*3 + j] = 0.0
    if fix_wrists:
        for j in range(3):
            smpl_pose[20*3 + j] = 0.0
            smpl_pose[21*3 + j] = 0.0
    return smpl_pose , smpl_trans

def load_smpl(smpl_folder_path,testfolder_list=None , fix_hands = False , fix_foots = False , fix_wrists = False):
    if testfolder_list == None:
        testfolder_list=[]
    smpl_paths_pose = natsorted(glob.glob(os.path.join(smpl_folder_path , "pose_[0-9][0-9][0-9][0-9].bin")))
    smpl_paths_skel = natsorted(glob.glob(os.path.join(smpl_folder_path , "skeleton_*.bin")))
    smpl_paths_npz  = natsorted(glob.glob(os.path.join(smpl_folder_path , "*.npz")))
    smpl_paths_pkl  = natsorted(glob.glob(os.path.join(smpl_folder_path , "pose_[0-9][0-9][0-9][0-9]*.pkl")))
    pose_cnt = 0
    skel_cnt = 0
    npz_cnt  = 0
    pkl_cnt  = 0
    if len(smpl_paths_pose) > 0:
        pose_cnt = 1
    if len(smpl_paths_skel) > 0:
        skel_cnt = 1
    if len(smpl_paths_npz) > 0:
        npz_cnt = 1
    if len(smpl_paths_pkl) > 0:
        pkl_cnt = 1
    if (pose_cnt + skel_cnt + npz_cnt + pkl_cnt) != 1:
        print("smpl folder include different format")
        print("about" , smpl_folder_path)
        print("pose_cnt:",pose_cnt)
        print("skel_cnt:",skel_cnt)
        print("npz_cnt:",npz_cnt)
        print("pkl_cnt:",pkl_cnt)
        #print("select use dataset:  0 : pose_***.bin  , 1 : skel_***.bin , 2 : ***.npz")
        #use_fmt = input()
        if pose_cnt != 0:
            print("use pose_cnt")
            smpl_paths = smpl_paths_pose
        elif skel_cnt != 0:
            print("use skel_cnt")
            smpl_paths = smpl_paths_skel
        elif npz_cnt != 0:
            print("use npz_cnt")
            smpl_paths = smpl_paths_npz
        elif pkl_cnt != 0:
            print("use pkl_cnt")
            smpl_paths = smpl_paths_pkl
        raise AssertionError("Incorrect smpl path")
    else:
        smpl_paths = natsorted(smpl_paths_pose + smpl_paths_skel + smpl_paths_npz + smpl_paths_pkl)

    pose_list = []
    beta_list = []
    trans_list = []
    beta     = None
    trans    = None
    for i,smpl_path in enumerate(smpl_paths):
        if testfolder_list == []:
            if npz_cnt == 1 :
                smpl_npz , _ = load_smpl_npz(smpl_path  , fix_hands , fix_foots,  fix_wrists)
                smpl_npz = smpl_npz.astype("float32")
                pose_list.append(smpl_npz)
            elif pkl_cnt == 1 :
                pose , beta , trans =load_smpl_pkl(smpl_path  , fix_hands , fix_foots,  fix_wrists)
                pose_list.append(pose)
            else:
                pose_list.append(load_smpl_bin(smpl_path , fix_hands , fix_foots,  fix_wrists))
        elif testfolder_list != []:
            if i in testfolder_list:
                if npz_cnt == 1:
                    smpl_npz , _ = load_smpl_npz(smpl_path  , fix_hands , fix_foots,  fix_wrists)
                    smpl_npz = smpl_npz.astype("float32")
                    pose_list.append(smpl_npz)
                elif pkl_cnt == 1 :
                    pose , beta , trans =load_smpl_pkl(smpl_path  , fix_hands , fix_foots,  fix_wrists)
                    pose_list.append(pose)
                    
                else:
                    pose_list.append(load_smpl_bin(smpl_path , fix_hands , fix_foots,  fix_wrists) )
        beta_list.append(beta)
        trans_list.append(trans)
        
    poses = np.array(pose_list)
    betas = np.array(beta_list)
    transs = np.array(trans_list)
    return poses , smpl_paths , betas , transs

def load_smpl_set(smpl_folder_path , base_id_list , fix_hands = False , fix_foots = False , fix_wrists = False):
    multi_smpls_list = []
    beta_list = []
    trans_list = []

    for base_id in base_id_list:
        smpls_path = os.path.join(smpl_folder_path , "poses_" + base_id)
        smpl_list = []
        smpl_bin_paths = natsorted(glob.glob(os.path.join(smpls_path,"pose_*.bin")))
        smpl_pkl_paths = natsorted(glob.glob(os.path.join(smpls_path,"pose_*.pkl")))
        if len(smpl_bin_paths) > 0 and len(smpl_pkl_paths) == 0:
            for i in range(3):
                smpl = load_smpl_bin(smpl_bin_paths[i] , fix_hands , fix_foots,  fix_wrists)
                smpl_list.append(smpl)
            beta = None
            trans = None
        elif len(smpl_bin_paths) == 0 and len(smpl_pkl_paths) > 0:
            for i in range(3):
                smpl , beta , trans = load_smpl_pkl(smpl_pkl_paths[i] , fix_hands , fix_foots,  fix_wrists)
                smpl_list.append(smpl)
        multi_smpl = np.hstack(smpl_list)
        multi_smpls_list.append(multi_smpl)
        beta_list.append(beta)
        trans_list.append(trans)
    multi_smpls_list = np.array(multi_smpls_list)
    beta_list         = np.array(beta_list)
    trans_list        = np.array(trans_list)
    #return multi_smpls_list 
    return multi_smpls_list , beta_list , trans_list 
        
def load_testID(test_folder_path):
    f = open(test_folder_path,"r")
    test_folder = f.read()
    test_folder_list = test_folder.split("\n")
    return list(map(int,test_folder_list))

def load_texture(texture_folder_path,testfolder_list=None, data = "disp" , padding1024_flg = False ):
    if testfolder_list == None:
        testfolder_list=[]
    if data == "disp":
        texture_paths = natsorted(glob.glob(texture_folder_path + "/displacement_texture_[0-9][0-9][0-9][0-9].npy"))
        texture_mask_headname = texture_folder_path + "/displacement_texture_mask_"
    if data == "color":
        texture_paths = natsorted(glob.glob(texture_folder_path + "/color_texture_[0-9][0-9][0-9][0-9]*.npy"))
    if data == "sw" :
        texture_paths = (glob.glob(texture_folder_path + "/sw_texture_[0-9][0-9][0-9][0-9]*.npy"))

    texture_list      = []
    texture_mask_list = []
    base_id_list      = []
    for i,texture_path in enumerate(texture_paths):
        base_id = os.path.basename(texture_path).split(".")[0].split("_")[-1]
        base_id_list.append(base_id)
        texture_mask_path = texture_mask_headname + base_id + ".npy"
        if testfolder_list == []:
            txr = np.load(texture_path)
            txr_mask = np.load(texture_mask_path)
            if padding1024_flg:
                txr      = cv2.copyMakeBorder(txr , 128, 128, 128, 128, cv2.BORDER_CONSTANT, value = (0.5,0.5,0.5))    #padding texture to resolution 1024
                txr_mask_xyz = np.concatenate([txr_mask[:,:,np.newaxis],txr_mask[:,:,np.newaxis],txr_mask[:,:,np.newaxis]],axis = 2).astype(np.int8)
                txr_mask = cv2.copyMakeBorder(txr_mask_xyz , 128, 128, 128, 128, cv2.BORDER_CONSTANT, value = (0 ,0 ,0))[:,:,0].astype(bool)    #padding texture to resolution 1024
            
            texture_list.append(txr)
            texture_mask_list.append(txr_mask)
        elif testfolder_list != []:
            if i in testfolder_list:
                txr = np.load(texture_path)
                if padding1024_flg:
                    txr = cv2.copyMakeBorder(txr , 128, 128, 128, 128, cv2.BORDER_CONSTANT, value = (0.5,0.5,0.5))      #padding texture to resolution 1024
                texture_list.append(txr)

    textures = np.array(texture_list)
    texture_masks = np.array(texture_mask_list)
    assert(textures.shape != 0)
    textures = np.transpose(textures,(0,3,1,2))    
    textures = textures.astype(np.float32)           
    return textures , base_id_list , texture_masks


def testIDfromTexture2smpl(testfolder_list):
    smplID_list = []
    multi_smplID_list = []
    for i in testfolder_list:
        if i + 2 < 0:
            print("please set testID 0 ~")
            sys.exit() 
        smplID_list.append(i-2)
        smplID_list.append(i-1)
        smplID_list.append(i)
        multi_smplID=[i-2,i-1,i]
        multi_smplID_list.append(multi_smplID)
    smplID_list = set(smplID_list)
    smplID_list = natsorted(smplID_list)
    return smplID_list ,multi_smplID_list

def make_smpl_set(smpls, mode = "stairs" ,smpl_paths = None ):
    multi_smpls_list = []
    debug_smpl_pathList = []
    fileIDs = []
    
    if mode == "stairs":
        loop_num = len(smpls)-2
    elif mode == "duplicate":
        loop_num = len(smpls)

    for t in range(loop_num):
        if mode == "stairs":
            multi_smpl_list=[smpls[t],smpls[t+1],smpls[t+2]]
            fileID = os.path.basename(smpl_paths[t+2]).split(".")[0].split("_")[-1]
        elif mode == "duplicate":
            multi_smpl_list=[smpls[t],smpls[t],smpls[t]]
            fileID = os.path.basename(smpl_paths[t]).split(".")[0].split("_")[-1]

        multi_smpl = np.hstack(multi_smpl_list)
        multi_smpls_list.append(multi_smpl)
        fileIDs.append(int(fileID))

    multi_smpls_list = np.array(multi_smpls_list)
    return multi_smpls_list , fileIDs

def enable_multiple_testID(testIDs):
    enable_IDlist = []
    for t in range(len(testIDs)-2):
        if testIDs[t] + 1 == testIDs[t+1] and testIDs[t+1] + 1 == testIDs[t+2]:
            enable_IDlist.append(testIDs[t+2])
    return enable_IDlist

def crop_face(texture,batch=False):
    if batch == True:
        shape_gap = 2
    else:
        shape_gap = 1

    height = texture.shape[0+shape_gap]
    width = texture.shape[1+shape_gap]

    width = 410
    offset_x = 10
    offset_y = 614
    if offset_x + width > texture.shape[1+shape_gap] or offset_x < 0:
        print("bbox is out of image")
        sys.exit()
    if offset_y + width > texture.shape[0+shape_gap] or offset_y < 0:
        print("bbox is out of image")
        sys.exit()
    if batch == True:
        cropped_face = texture[:,:,offset_y:offset_y+width,offset_x:offset_x+width]
    else:
        cropped_face = texture[:,offset_y:offset_y+width,offset_x:offset_x+width]

    """
    cropped_face = np.transpose(cropped_face,(1,2,0))
    cv2.imshow("test",cropped_face)
    cv2.waitKey(0)
    """
    return cropped_face

def crop_Rhand(texture,batch):
    if batch == True:
        shape_gap = 2
    else:
        shape_gap = 1

    height = texture.shape[0+shape_gap]
    width = texture.shape[1+shape_gap]

    
    width = 264
    offset_x = 760
    offset_y = 760
    if offset_x + width > texture.shape[1+shape_gap] or offset_x < 0:
        print("bbox is out of image")
        sys.exit()
    if offset_y + width > texture.shape[0+shape_gap] or offset_y < 0:
        print("bbox is out of image")
        sys.exit()
    if batch == True:
        cropped_Rhand = texture[:,:,offset_y:offset_y+width,offset_x:offset_x+width]
    else:
        cropped_Rhand = texture[:,offset_y:offset_y+width,offset_x:offset_x+width]
    
    #debug 
    """
    cropped_Rhand = np.transpose(cropped_Rhand,(1,2,0))
    cv2.imshow("test",cropped_Rhand)
    cv2.waitKey(0)
    """
    return cropped_Rhand

def crop_Lhand(texture,batch):
    if batch == True:
        shape_gap = 2
    else:
        shape_gap = 1

    height = texture.shape[0+shape_gap]
    width = texture.shape[1+shape_gap]

    width = 264
    offset_x = 760
    offset_y = 496
    if offset_x + width > texture.shape[1+shape_gap] or offset_x < 0:
        print("bbox is out of image")
        sys.exit()
    if offset_y + width > texture.shape[0+shape_gap] or offset_y < 0:
        print("bbox is out of image")
        sys.exit()
    if batch == True:
        cropped_Lhand = texture[:,:,offset_y:offset_y+width,offset_x:offset_x+width]
    else:
        cropped_Lhand = texture[:,offset_y:offset_y+width,offset_x:offset_x+width]
    
    #debug 
    """
    cropped_Lhand = np.transpose(cropped_Lhand,(1,2,0))
    cv2.imshow("test",cropped_Lhand)
    cv2.waitKey(0)
    """

    return cropped_Lhand

def load_ktfile(kt_path):
    """load kintree file"""
    f = open(kt_path,"rb")
    kt_bin = f.read()

    kt = []
    for i in range(24):
        kt_tmp = []
        for j in range(2):
            kt_tmp.append(struct.unpack("i",kt_bin[4*(2*i+j):4*(2*i+j)+4])[0])
        kt.append(kt_tmp)
    return kt  

def repair_edge_topology(edges , joints_group):
    upd = {}
    for i , joint in enumerate(joints_group):
        upd[joint] = i

    repair_edges = []
    
    for i , edge in enumerate(edges):
        repair_edges.append((upd[edge[0]] , upd[edge[1]], edge[2]))

    return repair_edges

def split_edges(edges):
    """split edges to 6 parts"""

    #print("edges : " , edges)
    joints_group1 = [9,12,13,14,15,16,17]
    joints_group2 = [14,17,19,21,23]
    joints_group3 = [13,16,18,20,22]
    joints_group4 = [0,1,2,3,6,9,13,14,16,17]
    joints_group5 = [0,2,5,8,11]
    joints_group6 = [0,1,4,7,10]
    joints_group_list = [joints_group1,joints_group2,joints_group3,joints_group4,joints_group5,joints_group6]

    edges_group_list = []
    for joints_group in joints_group_list:
        edges_group_list.append([i-1 for i in joints_group[1:]])
    
    edges_list = []
    for edges_group in edges_group_list:
        edges_list.append([edges[i] for i in edges_group])
    for i , edges in enumerate(edges_list):
        edges_list[i] = repair_edge_topology(edges_list[i] , joints_group_list[i])
    return edges_list , joints_group_list

def GetParentFromKintree(kt):
    "get kintree data and extract only parent"
    pa = []
    for x in kt:
        pa.append(x[0])
    pa = tuple(np.array(pa))
    return pa

def LoadStarSkeleton(skeleton_path,kt):
    """load Tpose"""
    f = open(skeleton_path,"rb")
    t_bin = f.read()

    t_pose = []
    for i in range(24):
        t_tmp = []
        for j in range(3):
            t_tmp.append(struct.unpack("f",t_bin[4*(3*i+j):4*(3*i+j)+4])[0])
        t_pose.append(t_tmp)
    
    t_joint_length = []
    for i in range(24):
        t_joint_length_tmp = []
        t_joint_length_tmp.append(t_pose[kt[i][1]][0]-t_pose[kt[i][0]][0])
        t_joint_length_tmp.append(t_pose[kt[i][1]][1]-t_pose[kt[i][0]][1])
        t_joint_length_tmp.append(t_pose[kt[i][1]][2]-t_pose[kt[i][0]][2])
        t_joint_length.append(t_joint_length_tmp)
    return t_pose , t_joint_length
