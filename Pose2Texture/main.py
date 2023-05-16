import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from torchinfo import summary
import tempfile

import os
import copy
import numpy as np
import tqdm
import sys
import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace
from external import pytorch_ssim

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER,MLFLOW_SOURCE_NAME

from models import dataloader , \
                   pose2texnetWithPoseAware , net_modules , ml_flow_writer
              
from external.poseAware.models.skeleton import build_edge_topology

from Reconstruct_for_2path import AITS_Reconstructor
from lib.smplpytorch_for_nakedBody import smplpytorch_processor
from lib import skinning_utils

def if_not_exists_makedir(path ,comment = None):
    if not os.path.exists(path):
        os.makedirs(path)
        if comment is not None:
            print(comment)

def save_model(model, optimizer, lr, val_cost, epoch, save_path):
    state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': lr,
            'val_cost': val_cost,
            'optimizer' : optimizer.state_dict(),
            }
    torch.save(state, save_path)

def mask_texture(texture_masks,textures,device):
    texture_masks = texture_masks.to(device)
    height = texture_masks[0].shape[0]
    width  = texture_masks[0].shape[1]
    batch_num = textures.shape[0]
    masked_texture = []
    zeros = torch.tensor(np.zeros((height,width) , dtype = "float32")).to(device)

    for i in range(batch_num):
        texture = textures[i]
        texture_mask = texture_masks[i]
        texture = torch.where(texture_mask , texture , zeros)
        masked_texture.append(texture)
    stacked_masked_texture = torch.stack(masked_texture) 
    nonzero_count = torch.count_nonzero(stacked_masked_texture)
    return  stacked_masked_texture , nonzero_count

def dispToRGB(texture,save_path):
    #texture = texture*255   #
    #texture = (texture - np.min(texture))/(np.max(texture) - np.min(texture))*255  
    #texture = texture.to(torch.uint8)
    texture = texture.to('cpu').detach().numpy()
    #texture = (texture - np.min(texture))/(np.max(texture) - np.min(texture))*255  
    texture = texture * 255  
    texture = np.where(texture > 255 , 255 , texture)
    texture = np.where(texture < 0 , 0 , texture)
    texture = texture.astype(np.uint8).transpose(1, 2, 0)
    texture = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path,texture)

def save_disp_as_rgb(texture,save_path):
    texture = texture * 255  
    texture = texture.astype(np.uint8)
    
    cv2.imwrite(save_path,texture)

def OmegaConf_list_to_Arg_namespace(OmegaConf_list):
    new_dict = {}
    for OmegaConf_dict in OmegaConf_list:
        new_dict = dict(**new_dict , **OmegaConf.to_container(OmegaConf_dict))      #omegaconf to dict  #from python3.5
        #new_dict = new_dict | OmegaConf.to_container(OmegaConf_dict)                #omegaconf to dict  #from python3.9(not tested)
    Arg_namespace = Namespace(**new_dict)
    return Arg_namespace

class loss_counter:
    def __init__(self , *loss_names):
        self.data_num_count = 0
        
        loss_dict = {}
        for loss_name in loss_names:
            loss_dict[loss_name] = 0.0
        
        self.loss_dict = loss_dict
    
    def print_var(self):
        print("data_num_count : ", self.data_num_count)
        print("loss_dict      : ", self.loss_dict)
    
    def add_loss(self , batch_size , *losses):
        for i , loss_name in enumerate(self.loss_dict):
            self.loss_dict[loss_name] += losses[i] * batch_size
        self.data_num_count += batch_size
    def output_total_loss(self):
        self.total_loss_dict = copy.deepcopy(self.loss_dict)
        for i , loss_name in enumerate(self.total_loss_dict):
            self.total_loss_dict[loss_name] /= self.data_num_count 
        print(self.total_loss_dict)
        return  self.data_num_count , self.total_loss_dict

class AITS_Trainer():
    def __init__(self , cfg , test_type_in = None):
        self.cfg                        = cfg
        self.asset_path                 = "../assets"
        self.kt_path                    = os.path.join(self.asset_path , "kintree.bin")
        self.multi_datasets_flg         = False

        self.device                     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode                       = cfg.train.mode      
        self.duplicate_smpl_flg         = cfg.train.duplicate_smpl_flg
        self.naked_texture_type         = cfg.train.naked_texture_type
        self.naked_topology             = cfg.train.naked_topology
        self.Canonical_Mesh_folder      = cfg.train.Canonical_Mesh_folder

        if self.mode == "train":
            self.type                   = cfg.train.type
            self.rundir                 = os.path.abspath(cfg.train.rundir_header) + "_" + self.type
            self.img_dir                = tempfile.TemporaryDirectory()
            self.img_dir_path           = self.img_dir.name
            self.cp_dir                 = os.path.join(self.rundir,"cp")
            
            self.train_datasetroot      = cfg.train.train_datasetroot
            self.train_datasetroots_txt_path = cfg.train.train_datasetroots_txt_path
            if self.train_datasetroots_txt_path != None:
                self.multi_datasets_flg     = True

            self.save_only_best         = cfg.train.save_only_best
            self.save_skip              = cfg.train.save_skip
            self.train_continue_flg     = cfg.train.train_continue_flg
            self.cp_continue            = cfg.train.cp_continue

            self.overfit_flg            = cfg.train.overfit_flg
            self.val_ratio              = cfg.train.val_ratio
            self.artifact_save_every_epoch = cfg.train.artifact_save_every_epoch
            self.cp_save_every_epoch    = cfg.train.cp_save_every_epoch

            self.lr_finish_cnt          = 0
            self.best_train_cost          = float('Inf')
            self.best_val_cost          = float('Inf')

        elif self.mode == "test":
            self.type                        = test_type_in
            self.rundir                      = os.path.abspath(cfg.train.rundir_header) + "_" + self.type
            self.smpl_folder_path_test       = cfg.train.smpl_folder_path_test
            self.save_mode                  = cfg.train.save_mode

            if self.type == "naked":
                if cfg.train.cp_naked_inference   != "latest":
                    self.cp_inference           = cfg.train.cp_naked_inference
                else:
                    self.cp_inference           = os.path.join(self.rundir , "cp" , "best.pth.tar")
            if self.type == "clothes":
                if cfg.train.cp_clothes_inference != "latest":
                    self.cp_inference           = cfg.train.cp_clothes_inference
                else:
                    self.cp_inference           = os.path.join(self.rundir , "cp" , "best.pth.tar")
            if self.type == "color":
                if cfg.train.cp_color_inference   != "latest":
                    self.cp_inference           = cfg.train.cp_color_inference
                else:
                    self.cp_inference           = os.path.join(self.rundir , "cp" , "best.pth.tar")
            if self.type == "clothes_and_color":
                if cfg.train.cp_clothes_and_color_inference   != "latest":
                    self.cp_inference           = cfg.train.cp_clothes_and_color_inference
                else:
                    self.cp_inference           = os.path.join(self.rundir , "cp" , "best.pth.tar")
            
            self.save_predict_path      = cfg.train.save_predict_path
            self.IDstart                = cfg.train.IDstart
            self.IDend                  = cfg.train.IDend
            self.IDstep                 = cfg.train.IDstep


        #model
        self.num_epoch              = cfg.model.num_epoch
        self.batch_size             = cfg.model.Mybatch_size

        self.texture_disp_loss      = cfg.model.texture_disp_loss
        self.texture_color_loss     = cfg.model.texture_color_loss
        self.init_lr                = cfg.model.init_lr
        self.loss_weight_disp       = cfg.model.loss_weight_disp
        self.loss_weight_color      = cfg.model.loss_weight_color
        self.loss_weight_ssim       = cfg.model.loss_weight_ssim
        self.loss_weight_hist       = cfg.model.loss_weight_hist
        self.loss_weight_disp_mask  = cfg.model.loss_weight_disp_mask

        self.multi_topology_flg     = cfg.model.multi_topology_flg
        self.separate_flg           = cfg.model.separate_flg
        self.use_lr_scheduler       = cfg.model.use_lr_scheduler
        self.lr_scheduler_patience  = cfg.model.lr_scheduler_patience
        self.prediction_texture_mask = cfg.model.prediction_texture_mask

        self.texture_disp_dir_name  = "texture_disp"
        self.texture_color_dir_name = "texture_color"
        
        self.texture_resolution = 1024
        self.padding1024_flg    = True
        if self.type == "naked" :
            self.texture_naked_dir_name = "texture_naked_" + self.naked_texture_type + "_" + self.naked_topology
            if self.naked_topology  == "SMPL":
                self.texture_resolution = 512
                self.padding1024_flg    = False

        kintree = net_modules.load_ktfile(self.kt_path)

        joint_topology = net_modules.GetParentFromKintree(kintree)
        edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
    
        args = OmegaConf_list_to_Arg_namespace(cfg.model.aware)

        self.train_model = pose2texnetWithPoseAware.pose2texnet(self.device,self.type , args ,edges ,\
                                                                self.separate_flg, self.texture_resolution , self.multi_topology_flg , self.prediction_texture_mask).to(self.device)

        #summary(self.train_model,[(1,24*3,3)])

        params_to_update = []
        for name, param in self.train_model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        self.optimizer = torch.optim.Adam(params_to_update, self.init_lr)

        if self.mode == "train" and self.use_lr_scheduler:
            if self.overfit_flg:
                self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.lr_scheduler_patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=20, min_lr=1e-06, eps=1e-08)
            else:
                self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.lr_scheduler_patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=3, min_lr=1e-06, eps=1e-08)

        if self.type == "clothes" or self.type == "clothes_and_color" or self.type == "naked":      
            if self.texture_disp_loss == "l1":
                self.criterion_disp = nn.L1Loss(reduction='sum').cuda()   
            elif self.texture_disp_loss == "mse":
                self.criterion_disp = nn.MSELoss(reduction='sum').cuda()    
            #self.criterion_disp_ssim = net_modules.SSIMLoss().cuda()    
            self.criterion_disp_ssim = pytorch_ssim.SSIM().cuda()   
            self.criterion_disp_mask = nn.BCELoss().cuda()
    
        if self.type == "color" or self.type == "clothes_and_color":      
            if self.texture_color_loss == "l1":
                self.criterion_color = nn.L1Loss(reduction='sum').cuda()   
            elif self.texture_color_loss == "mse":
                self.criterion_color = nn.MSELoss(reduction='sum').cuda()  
            self.criterion_color_hist = net_modules.SoftHistLoss(self.device)

    def make_rundir(self):
        if_not_exists_makedir(self.rundir  , comment = 'rundir : ' + self.rundir)
        
        #if_not_exists_makedir(self.img_dir_path)

        if_not_exists_makedir(self.cp_dir  , comment = 'latest cp:' + self.cp_dir)

    def data_loader(self):
        dataset_list = []
        if self.multi_datasets_flg and self.mode == "train":
            train_datasetroots                   = net_modules.load_paths(self.train_datasetroots_txt_path)
            smpl_folderpath_train_list           = []
            for train_datasetroot in train_datasetroots:
                smpl_params_path                 = os.path.join(train_datasetroot , "smplparams")
                smpl_seqs_path                   = os.path.join(train_datasetroot , "seqs")
                if os.path.exists(smpl_params_path):
                    smpl_folderpath_train = smpl_params_path
                elif os.path.exists(smpl_seqs_path):
                    smpl_folderpath_train = smpl_seqs_path      
                else:
                    print("smpl_params_path : " , smpl_params_path , " or " , smpl_seqs_path )
                    raise AssertionError("above smpl_params_path is incorrect ")
             
                if self.type == "clothes":
                    texture_disp_folderpath_train = os.path.join(train_datasetroot  , self.texture_disp_dir_name)
                    dataset = dataloader.Dataloader(self.type , smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= texture_disp_folderpath_train,texture_color_folder_path = None , load_smpl_set_flg = True  , padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)
                elif self.type == "color" : 
                    texture_color_folderpath_train = os.path.join(train_datasetroot  , self.texture_color_dir_name)
                    dataset = dataloader.Dataloader(self.type , smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= None,texture_color_folder_path = texture_color_folderpath_train, load_smpl_set_flg = True , padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)
                elif self.type == "naked" : 
                    texture_disp_folderpath_train = os.path.join(train_datasetroot  , self.texture_naked_dir_name)
                    dataset = dataloader.Dataloader(self.type , smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= texture_disp_folderpath_train,texture_color_folder_path = None , load_smpl_set_flg = False , duplicate_smpl = self.duplicate_smpl_flg  , padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)
                elif self.type == "clothes_and_color" : 
                    texture_disp_folderpath_train = os.path.join(train_datasetroot  , self.texture_disp_dir_name)
                    texture_color_folderpath_train = os.path.join(train_datasetroot  , self.texture_color_dir_name)
                    dataset = dataloader.Dataloader(self.type , smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= texture_disp_folderpath_train,texture_color_folder_path = texture_color_folderpath_train, load_smpl_set_flg = True , padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)
                dataset_list.append(dataset)
            dataset = torch.utils.data.ConcatDataset(dataset_list)  
        else:
            if self.mode == "test" :
                if self.type == "naked" :
                    dataset = dataloader.Dataloader( "test_naked" ,  smpl_folder_path = self.smpl_folder_path_test,texture_disp_folder_path= None ,texture_color_folder_path = None , load_smpl_set_flg = False,duplicate_smpl = self.duplicate_smpl_flg ,  padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)
                    return dataset
                else :
                    dataset = dataloader.Dataloader( "test_fix" ,  smpl_folder_path = self.smpl_folder_path_test,texture_disp_folder_path= None ,texture_color_folder_path = None , load_smpl_set_flg = False,duplicate_smpl = self.duplicate_smpl_flg ,  padding1024_flg = self.padding1024_flg , use_skeleton_aware = True )
                    return dataset
                

            smpl_folderpath_train                = os.path.join(self.train_datasetroot , "smplparams")
            if self.type == "clothes":
                texture_disp_folderpath_train  = os.path.join(self.train_datasetroot  , self.texture_disp_dir_name)
                dataset = dataloader.Dataloader(self.type, smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= texture_disp_folderpath_train,texture_color_folder_path = None , load_smpl_set_flg = True ,padding1024_flg = self.padding1024_flg , use_skeleton_aware = True )
                print(len(dataset))
            elif self.type == "color" : 
                raise AssertionError("not implemented or not tested") #FIXME
                texture_color_folderpath_train = os.path.join(self.train_datasetroot  , self.texture_color_dir_name)
                dataset = dataloader.Dataloader(self.type, smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= None,texture_color_folder_path = texture_color_folderpath_train, load_smpl_set_flg = True , padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)
            elif self.type == "naked" : 
                raise AssertionError("not implemented or not tested") #FIXME
                texture_disp_folderpath_train  = os.path.join(self.train_datasetroot  , self.texture_naked_dir_name)
                dataset = dataloader.Dataloader(self.type, smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= texture_disp_folderpath_train,texture_color_folder_path = None , load_smpl_set_flg = True , duplicate_smpl = self.duplicate_smpl_flg , padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)
            elif self.type == "clothes_and_color" : 
                raise AssertionError("not implemented or not tested") #FIXME
                texture_disp_folderpath_train  = os.path.join(self.train_datasetroot  , self.texture_disp_dir_name)
                texture_color_folderpath_train = os.path.join(self.train_datasetroot  , self.texture_color_dir_name)
                dataset = dataloader.Dataloader(self.device , self.type, smpl_folder_path = smpl_folderpath_train,texture_disp_folder_path= texture_disp_folderpath_train,texture_color_folder_path = texture_color_folderpath_train, load_smpl_set_flg = True , padding1024_flg = self.padding1024_flg , use_skeleton_aware = True)

        if self.overfit_flg == True:
            val_data_size = 0
        else:
            val_data_size = int(len(dataset) * self.val_ratio) 
        train_data_size = len(dataset) - val_data_size  
        #trainデータセットとvalデータセットにランダムで分ける

        dataset_train , dataset_val =  torch.utils.data.random_split(
            dataset,
            [train_data_size, val_data_size],
            generator=torch.Generator().manual_seed(0)  # 乱数シードの固定
        )
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True )
        dataloader_val   = torch.utils.data.DataLoader(dataset_val  , batch_size=self.batch_size, shuffle=False)
        print("train_data_size : " , train_data_size , " , val_data_size : " , val_data_size )
        return dataset , dataloader_train , dataloader_val , train_data_size , val_data_size

    def init_ml_flow(self):
        #mlflow.set_tracking_uri("file://" + ".." + "/mlruns") #/mlrunsディレクトリの場所変えたかったらこれでできるよ
        EXPERIMENT_NAME = self.type
        self.writer = ml_flow_writer.MlflowWriter(EXPERIMENT_NAME)
        
        if self.multi_datasets_flg:
            dataset_name = self.train_datasetroots_txt_path.split("/")[-3]
            dataset_type = self.train_datasetroots_txt_path.split("/")[-1]
        else:
            dataset_name = self.train_datasetroot.split("/")[-3]
            dataset_type = self.train_datasetroot.split("/")[-1]
                
        run_name = self.mode + "_" + self.type + "_" + dataset_name + "_" + dataset_type
        tags = {'train':0,
                MLFLOW_RUN_NAME:run_name,
                MLFLOW_USER:"None",
                MLFLOW_SOURCE_NAME:"None",
            }

        self.writer.create_new_run(tags) 
        self.writer.log_params_from_omegaconf_dict(self.cfg)

    def train_from_continued(self):
        print("train is from continue")
        cp_continue=torch.load(self.cp_continue,map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
        print('trained for', cp_continue['epoch'], 'epochs', 'val cost',cp_continue['val_cost'])
        self.train_model.load_state_dict(cp_continue['state_dict'])
        #optimizergen_whole.load_state_dict(cp_continue['optimizer'])  #if want to load learning rate
        # fix exept last layer
        for param in self.train_model.parameters():
            param.requires_grad = False

        last_layer = list(self.train_model.children())[-1]
        print(f'except last layer: {last_layer}')
        for param in last_layer.parameters():
            param.requires_grad = True

    def train_step(self , train_mode, smpl_input , texture_disp_label, texture_mask , texture_color_label):
        loss_total      = torch.Tensor([0.0]).to(self.device)
        loss_disp       = torch.Tensor([0.0]).to(self.device)
        loss_disp_ssim  = torch.Tensor([0.0]).to(self.device)
        loss_disp_mask  = torch.Tensor([0.0]).to(self.device)
        loss_color      = torch.Tensor([0.0]).to(self.device)
        loss_color_hist = torch.Tensor([0.0]).to(self.device)
        if train_mode == "train":
            self.train_model.zero_grad()   

        if self.type == "clothes" or self.type == "naked":   
            if self.prediction_texture_mask:
                texture_disp_output , texture_disp_mask_output = self.train_model(smpl_input.to(self.device) , None)
            else:
                texture_disp_output  = self.train_model(smpl_input.to(self.device) , None)
        if self.type == "color" :   
            raise AssertionError("not implemented or not tested") #FIXME
            texture_color_output = self.train_model(smpl_input.to(self.device) , None)
        elif self.type == "clothe and color" :   
            raise AssertionError("not implemented or not tested") #FIXME
            texture_disp_output , texture_color_output = self.train_model(smpl_input.to(self.device) , None)

        if self.type == "clothes" or self.type == "naked" or self.type == "clothe and color":   
            masked_texture_disp_output , _ = mask_texture(texture_mask,texture_disp_output                ,self.device)
            masked_texture_disp_label  , count_nonzero = mask_texture(texture_mask,texture_disp_label.to(self.device) ,self.device)

            loss_disp      =  self.criterion_disp(masked_texture_disp_output,masked_texture_disp_label)
            loss_disp      =  loss_disp / count_nonzero
            loss_total     += loss_disp * self.loss_weight_disp
            if self.loss_weight_ssim != 0.0:
                loss_disp_ssim =  - self.criterion_disp_ssim(masked_texture_disp_output,masked_texture_disp_label)
                loss_total     += loss_disp_ssim * self.loss_weight_ssim 
            if self.prediction_texture_mask:
                loss_disp_mask = self.criterion_disp_mask(texture_disp_mask_output , texture_mask.unsqueeze(1).to(torch.float32).to(self.device))
                loss_total     += loss_disp_mask * self.loss_weight_disp_mask 

        if self.type == "color" or self.type == "clothe and color":
            raise AssertionError("not implemented or not tested") #FIXME
            masked_texture_color_output , _             = mask_texture(texture_mask,texture_color_output                 ,self.device)
            masked_texture_color_label  , count_nonzero = mask_texture(texture_mask,texture_color_label.to(self.device)  ,self.device)
            
            loss_color      =  self.criterion_color(masked_texture_color_output,masked_texture_color_label)
            loss_color      =  loss_color / count_nonzero
            loss_total      += loss_color * self.loss_weight_color
            if self.loss_weight_hist != 0.0:
                loss_color_hist =  self.criterion_color_hist(masked_texture_disp_output,masked_texture_disp_label)
                loss_total      += loss_color_hist * self.loss_weight_hist
    
        if train_mode == "train":
            loss_total.backward()
            self.optimizer.step()

        return (loss_total.item() ,     \
                loss_disp.item() ,      \
                loss_disp_ssim.item() , \
                loss_disp_mask.item() , \
                loss_color.item() ,      \
                loss_color_hist.item())
        
    def train_epoch(self , train_mode , epoch , dataset , dataloader , data_size):
        loss_counter_instance = loss_counter(train_mode + "_total" ,                           \
                                             train_mode + "_disp_"  + self.texture_disp_loss , \
                                             train_mode + "_disp_ssim" ,                       \
                                             train_mode + "_disp_mask" ,                       \
                                             train_mode + "_color_" + self.texture_color_loss ,\
                                             train_mode + "_color_hist")

        if train_mode == "train":
            self.train_model.train()
            torch.set_grad_enabled(True)
            
            if self.train_continue_flg and epoch == 20:
                # unfreeze all layers
                for param in self.train_model.parameters():
                    param.requires_grad = True

        elif train_mode == "val":
            self.train_model.eval()
            torch.set_grad_enabled(False)

        for itr , datas in tqdm.tqdm(enumerate(dataloader)):
            if self.type == "clothes" or self.type == "naked":
                _ , smpl_input , texture_disp_label , texture_mask  = datas
                (loss_total , loss_disp , loss_disp_ssim , loss_disp_mask , loss_color , loss_color_hist) = self.train_step(train_mode, smpl_input, texture_disp_label, texture_mask , None )
            elif self.type == "color":
                raise AssertionError("not implemented or not tested") #FIXME
                _ , smpl_input , texture_color_label , texture_mask = datas
                (loss_total , loss_disp , loss_disp_ssim , loss_disp_mask, loss_color , loss_color_hist) = self.train_step(train_mode, smpl_input, None              , texture_mask , texture_color_label )
            elif self.type == "clothes_and_color":
                raise AssertionError("not implemented or not tested") #FIXME
                _ , smpl_input , texture_disp_label , texture_color_label , texture_mask = datas
                (loss_total , loss_disp , loss_disp_ssim , loss_disp_mask  , loss_color , loss_color_hist) = self.train_step(train_mode, smpl_input, texture_disp_label, texture_mask , texture_color_label )

            tmp_batch_size = smpl_input.shape[0]
            loss_counter_instance.add_loss(tmp_batch_size ,        \
                                           loss_total ,           \
                                           loss_disp  ,           \
                                           loss_disp_ssim ,       \
                                           loss_disp_mask ,       \
                                           loss_color ,           \
                                           loss_color_hist)      

        data_num_count , loss_dict = loss_counter_instance.output_total_loss()
        assert(data_num_count == data_size)
        
        ### update learning rate and save cp
        lr = self.optimizer.param_groups[0]["lr"]

        if train_mode == "val":
            if self.use_lr_scheduler:
                self.lr_scheduler.step(loss_dict["val_total"])
            print('Validation loss: {:.4e}'.format(loss_dict["val_total"]))
            print("lr:" , lr)
            if lr == 1e-06:
                if self.lr_finish_cnt == 50:
                    if epoch < 200:
                        save_path = os.path.join(self.cp_dir , "best.pth.tar")
                        save_model(self.train_model, self.optimizer, lr, self.best_val_cost, epoch, save_path)
                        self.writer.log_artifact(save_path)
                        print('saved model state', save_path)
                    print("program finish because lr == 1e-06")
                    sys.exit()
                else:
                    self.lr_finish_cnt += 1
                    print("count to finish : " , self.lr_finish_cnt)

            # save model if val_loss reduce than before
            if loss_dict["val_total"] < self.best_val_cost:
                print('validation cost is lower than best before, saving model_gen...')
                self.best_val_cost = loss_dict["val_total"] 

                if self.save_skip == True and epoch < 200:  
                    print("!!!!!!!!!!save skipped!!!!!!!!!!")
                else:
                    save_path = os.path.join(self.cp_dir , "best.pth.tar")
                    save_model(self.train_model, self.optimizer, lr, self.best_val_cost, epoch, save_path)
                    self.writer.log_artifact(save_path)
                    print('saved model state', save_path)
        
        ### write log ###
        for key in loss_dict:
            self.writer.log_metric_step(key , loss_dict[key] , step=epoch)
        self.writer.log_metric_step("learning_rate" , lr, step=epoch)

        ### save checkpoint ###
        if (epoch+1) % self.cp_save_every_epoch == 0: 
            if self.overfit_flg == True:
                saved_loss_dict = loss_dict["train_total"]
                
                if self.save_only_best == True :
                    save_path = os.path.join(self.cp_dir , "best.pth.tar")
                    print('saved model state', save_path)
                    save_model(self.train_model, self.optimizer, lr, saved_loss_dict, epoch, save_path)
                    self.writer.log_artifact(save_path)
                else:
                    save_path = os.path.join(self.cp_dir, "checkpoint-%d.pth.tar" %(epoch+1) )
                    print('saved model state', save_path)
                    save_model(self.train_model, self.optimizer, lr, saved_loss_dict, epoch, save_path)
                    self.writer.log_artifact(save_path)

            elif self.overfit_flg == False and train_mode == "val" and self.save_only_best == False:
                save_path = os.path.join(self.cp_dir, "checkpoint-%d.pth.tar" %(epoch+1) )
                print('saved model state', save_path)
                saved_loss_dict = loss_dict["val_total"]  
                save_model(self.train_model, self.optimizer, lr, saved_loss_dict, epoch, save_path)
                self.writer.log_artifact(save_path)

        ### save artifact ###
        if (epoch+1) % self.artifact_save_every_epoch == 0 :
            if self.overfit_flg == False and train_mode == "val" or self.overfit_flg == True and train_mode == "train" :
                img_dir_epoch = os.path.join(self.img_dir_path , str(epoch))
                if_not_exists_makedir(img_dir_epoch)
                dataset_len = len(dataset) 
                for i in range(dataset_len):
                    if i%(int(dataset_len/5)) == 0:
                        test_smpl = torch.from_numpy(dataset[i][1])
                        test_smpl = torch.unsqueeze(test_smpl,0)
                        test_smpl = test_smpl.to(self.device)

                        if self.type == "clothes" or self.type == "naked":   
                            if self.prediction_texture_mask:
                                texture_disp_output , texture_disp_mask_output  = self.train_model(test_smpl , None)
                            else:
                                texture_disp_output  = self.train_model(test_smpl , None)
                        if self.type == "color" :   
                            raise AssertionError("not implemented or not tested") #FIXME
                            texture_color_output = self.train_model(test_smpl , None)
                        elif self.type == "clothe and color" :   
                            raise AssertionError("not implemented or not tested") #FIXME
                            texture_disp_output , texture_color_output = self.train_model(test_smpl , None)
                        
                        if self.type == "clothes" or self.type == "naked" or self.type == "clothes_and_color":
                            test_disp = torch.from_numpy(dataset[i][2])
                            test_disp = torch.unsqueeze(test_disp,0)
                            test_disp_maskGT = torch.from_numpy(dataset[i][3])
                            test_disp_maskGT = torch.cat([test_disp_maskGT.unsqueeze(0),test_disp_maskGT.unsqueeze(0),test_disp_maskGT.unsqueeze(0)],axis = 0)
                            test_disp_maskGT = torch.unsqueeze(test_disp_maskGT,0)
                            dispToRGB(texture_disp_output[0]  , os.path.join(img_dir_epoch,"disp"   + str(i).zfill(4) + ".png" ))
                            dispToRGB(test_disp[0]            , os.path.join(img_dir_epoch,"dispGT" + str(i).zfill(4) + ".png" ))
                            dispToRGB(test_disp_maskGT[0]     , os.path.join(img_dir_epoch,"disp_maskGT" + str(i).zfill(4) + ".png" ))
                            if self.prediction_texture_mask:
                                test_disp_mask = torch.cat([texture_disp_mask_output,texture_disp_mask_output,texture_disp_mask_output],axis = 1)
                                dispToRGB(test_disp_mask[0]  , os.path.join(img_dir_epoch,"disp_mask" + str(i).zfill(4) + ".png" ))
                            
                        elif self.type == "color" or self.type == "clothes_and_color":
                            raise AssertionError("not implemented or not tested") #FIXME
                            test_color = torch.from_numpy(dataset[i][3])
                            test_color = torch.unsqueeze(test_color,0)
                            dispToRGB(texture_color_output[0] ,os.path.join(img_dir_epoch, "color.png" ))
                            dispToRGB(test_color[0]           ,os.path.join(img_dir_epoch, "colorGT.png" ))
                        self.writer.log_artifacts(img_dir_epoch)

    def test_preprocess(self):
        # Load weights
        print('Load ' + self.type + '_checkpoint',self.cp_inference)
        cp=torch.load(self.cp_inference,map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
        print('trained for', cp['epoch'], 'epochs', 'val cost',cp['val_cost'])
        
        self.train_model.load_state_dict(cp['state_dict'])
        self.optimizer.load_state_dict(cp['optimizer'])

        self.train_model.eval()
        torch.set_grad_enabled(False)

        if self.save_predict_path == None:
            self.save_dir  = os.path.join(self.rundir, 'predict_texture')
        else:
            self.save_dir  = os.path.join(self.save_predict_path, 'predict_texture')
        
        if_not_exists_makedir(self.save_dir)
        
    def test_aft_process(self,texture):
        texture_tensor = texture[0].permute((1,2,0))
        if self.padding1024_flg:
            texture_tensor = texture_tensor[128:1024 -128 , 128:1024 -128]
        return texture_tensor

    def test_step(self, fileID , smpl_input):
        smpl_input = torch.from_numpy(smpl_input).to(self.device)
        test_smpl  = torch.unsqueeze(smpl_input,0)

        if self.type == "clothes" or self.type == "naked" or self.type == "color":   
            if self.prediction_texture_mask:
                texture_disp_output , texture_disp_mask_output  = self.train_model(test_smpl , None)
                texture_disp_mask_output = self.test_aft_process(texture_disp_mask_output)
                texture_disp_mask_output = torch.where(texture_disp_mask_output > 0.5 , 1 , 0)
            else:
                texture_disp_output  = self.train_model(test_smpl, None)
                
            texture_disp_output      = self.test_aft_process(texture_disp_output)

            if self.save_mode == "debug":
                disp_output_texture_path        = os.path.join(self.save_dir,"texture_" + str(fileID).zfill(4))
                disp_output_texture_debug_path  = os.path.join(self.save_dir,"_debug_texture_" + str(fileID).zfill(4) + ".png")
        
                texture_disp_output_debug = texture_disp_output.to('cpu').detach().numpy().copy()
                np.save(disp_output_texture_path,texture_disp_output_debug)
                save_disp_as_rgb(texture_disp_output_debug,disp_output_texture_debug_path) 
                disp_mask_output_texture_path        = os.path.join(self.save_dir,"texture_mask_" + str(fileID).zfill(4))
                disp_mask_output_texture_debug_path  = os.path.join(self.save_dir,"_debug_texture_mask_" + str(fileID).zfill(4) + ".png")
                if self.prediction_texture_mask:
                    texture_disp_mask_output_debug = texture_disp_mask_output.to('cpu').detach().numpy().copy()
                    np.save(disp_mask_output_texture_path,texture_disp_mask_output_debug)
                    save_disp_as_rgb(texture_disp_mask_output_debug,disp_mask_output_texture_debug_path) 

            
            if self.prediction_texture_mask:
                return texture_disp_output , texture_disp_mask_output
            else:
                return texture_disp_output

        elif self.type == "clothes_and_color" :   
            raise AssertionError("not implemented or not tested") #FIXME
            texture_disp_output , texture_color_output = self.train_model(test_smpl.to(self.device) , None)
            texture_disp_output  = self.test_aft_process(texture_disp_output)
            texture_color_output = self.test_aft_process(texture_color_output)    
            
            if self.save_mode == "debug":
                output_disp_path         = os.path.join(self.save_dir,"texture_disp_" + str(fileID).zfill(4))
                output_disp_debug_path   = os.path.join(self.save_dir,"_debug_texture_disp_" + str(fileID).zfill(4) + ".png")
                output_color_path        = os.path.join(self.save_dir,"texture_color_" + str(fileID).zfill(4))
                output_color_debug_path  = os.path.join(self.save_dir,"_debug_texture_color_" + str(fileID).zfill(4) + ".png")
                
                texture_disp_output_debug       = texture_disp_output.to('cpu').detach().numpy().copy()
                texture_color_output_debug      = texture_color_output.to('cpu').detach().numpy().copy()
                np.save(output_disp_path,texture_disp_output)
                np.save(output_color_path,texture_color_output)
                save_disp_as_rgb(texture_disp_output_debug ,output_disp_debug_path)
                save_disp_as_rgb(texture_color_output_debug,output_color_debug_path)
            return texture_disp_output , texture_color_output
          
    #def __del__(self):
    #    self.writer.set_terminated()      

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    if cfg.train.mode == "train":
        Trainer = AITS_Trainer(cfg)
        Trainer.make_rundir()
        dataset , dataloader_train , dataloader_val , train_data_size , val_data_size = Trainer.data_loader()
        Trainer.init_ml_flow()
    
        if cfg.train.train_continue_flg:
            Trainer.train_from_continued()

        for epoch in range(cfg.model.num_epoch):
            print('-----------------------------------------------------')
            print('Epoch {}/{}'.format(epoch+1, cfg.model.num_epoch))
            print('-----------------------------------------------------')

            ### train ###
            Trainer.train_epoch("train" , epoch , dataset, dataloader_train , train_data_size)
            
            if val_data_size == 0:
                continue
            
            ###Val###
            Trainer.train_epoch("val"   , epoch , dataset, dataloader_val   , val_data_size)
    
    elif cfg.train.mode == "test":      
        print("Prediction mode")

        if cfg.train.test_type == "naked":                    #naked only
            raise AssertionError("not implemented or not tested") #FIXME
            Trainer_naked   = AITS_Trainer(cfg , "naked")
            Trainer_naked.test_preprocess()
            dataset = Trainer_naked.data_loader()
        elif cfg.train.test_type == "clothes":                #naked + clothes
            if cfg.train.cp_naked_inference != "smplpytorch":
                Trainer_naked   = AITS_Trainer(cfg , "naked")
                Trainer_naked.test_preprocess()
            Trainer_clothes = AITS_Trainer(cfg , "clothes")
            Trainer_clothes.test_preprocess()
            dataset = Trainer_clothes.data_loader()
        elif cfg.train.test_type == "color":                  #naked + clothes + color
            raise AssertionError("not implemented or not tested") #FIXME
            if cfg.train.cp_naked_inference != "smplpytorch":
                Trainer_naked   = AITS_Trainer(cfg , "naked")
                Trainer_naked.test_preprocess()
            Trainer_clothes = AITS_Trainer(cfg , "clothes")
            Trainer_clothes.test_preprocess()
            Trainer_color = AITS_Trainer(cfg , "color")
            Trainer_color.test_preprocess()
            dataset = Trainer_clothes.data_loader()
        elif cfg.train.test_type == "clothes_and_color":      #naked + clothes + color
            raise AssertionError("not implemented or not tested") #FIXME
            if cfg.train.cp_naked_inference != "smplpytorch":
                Trainer_naked   = AITS_Trainer(cfg , "naked")
                Trainer_naked.test_preprocess()
            Trainer_clothes_and_color = AITS_Trainer(cfg , "clothes_and_color")
            Trainer_clothes_and_color.test_preprocess()

        Reconstructor = AITS_Reconstructor(cfg) 
        Reconstructor.asset_loader()            
        Reconstructor.reconst_preprocess()   
        if cfg.train.cp_naked_inference == "smplpytorch":
            smplpytorch_processor_instance = smplpytorch_processor(Reconstructor.smpl_model_dir , gender = Reconstructor.smpl_gender)
            
            if Reconstructor.use_beta == True:
                beta = np.load(Reconstructor.beta_path)
                if np.all(beta == 0.0):
                    beta = None
            else:
                smplpytorch_processor_instance.rescale_smpl_model(Reconstructor.Tshapecoarsejoints , Reconstructor.T_joints , Reconstructor.kintree)
                beta = None
            smplpytorch_processor_instance.init_SMPL_layer()

        IDs = []
        if cfg.train.IDstart != -999 and cfg.train.IDend != -999 and cfg.train.IDstep != -999:
            IDs = [i for i in range(cfg.train.IDstart , cfg.train.IDend, cfg.train.IDstep)]
        
        for datas in tqdm.tqdm(dataset):           #for datas in tqdm.tqdm(dataset):
            fileID , smpl_input , _ , trans = datas

            if IDs == [] or fileID in IDs:
                smpl_for_reconst = smpl_input[48:,:]
                if cfg.train.test_type == "naked":      
                    raise AssertionError("not implemented or not tested") #FIXME
                    predicted_texture_naked   = Trainer_naked.test_step(fileID , smpl_input)
                    Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = None, texture_naked = predicted_texture_naked , texture_clothes = None , texture_color = None )   
                elif cfg.train.test_type == "clothes":
                    #smpl_for_reconst = skinning_utils.fix_smpl_parts(smpl_for_reconst , fix_hands = True , fix_foots = True , fix_wrists = False)
                    predicted_texture_clothes_mask = None
                    if Trainer_clothes.prediction_texture_mask:
                        predicted_texture_clothes , predicted_texture_clothes_mask = Trainer_clothes.test_step(fileID , smpl_input)  #2ループ目以降 : 0.0027sくらい(十分速い)
                    else:
                        predicted_texture_clothes = Trainer_clothes.test_step(fileID , smpl_input)  #2ループ目以降 : 0.0027sくらい(十分速い)
                    if cfg.train.cp_naked_inference == "smplpytorch": 
                        smpl_vtxs           = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(smpl_for_reconst , beta)
                        if trans is not None:
                            smpl_vtxs = smpl_vtxs + torch.from_numpy(trans.astype(np.float32)).cuda()
                        Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = smpl_vtxs, texture_naked = None , texture_clothes = predicted_texture_clothes , texture_color = None , texture_mask = predicted_texture_clothes_mask)    #0.013sくらい(interpolation無し)                    
                    else:                       
                        predicted_texture_naked   = Trainer_naked.test_step(fileID , smpl_input)
                        Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = None, texture_naked = predicted_texture_naked , texture_clothes = predicted_texture_clothes , texture_color = None , texture_mask = predicted_texture_clothes_mask)   
                elif cfg.train.test_type == "color":    #TODO:implement here
                    raise AssertionError("not implemented or not tested") #FIXME
                    #smpl_input = torch.from_numpy(smpl_input).to(Trainer_clothes.device)
                    predicted_texture_clothes = Trainer_clothes.test_step(fileID , smpl_input)
                    predicted_texture_color   = Trainer_color.test_step(fileID , smpl_input)

                    predicted_texture_naked   = Trainer_naked.test_step(fileID , smpl_input)

                    if cfg.train.cp_naked_inference == "smplpytorch":
                        #smpl_vtxs = Reconstructor.Predict_naked_with_smplpytorch(smpl_for_reconst)
                        smpl_vtxs           = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(smpl_for_reconst , Trainer.beta)
                        Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = smpl_vtxs, texture_naked = None , texture_clothes = predicted_texture_clothes , texture_color = predicted_texture_color )  
                    else:
                        predicted_texture_naked   = Trainer_naked.test_step(fileID , smpl_input)
                        Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = None, texture_naked = predicted_texture_naked , texture_clothes = predicted_texture_clothes , texture_color = predicted_texture_color )   
                elif cfg.train.test_type == "clothes and color":
                    raise AssertionError("not implemented or not tested") #FIXME
                    smpl_input = torch.from_numpy(smpl_input).to(Trainer_clothes_and_color.device)
                    smpl_for_reconst = smpl_input[48:,:]
                    predicted_texture_clothes , predicted_texture_color = Trainer_clothes_and_color.test_step(fileID , smpl_input)

                    if cfg.train.cp_naked_inference == "smplpytorch":
                        #smpl_vtxs = Reconstructor.Predict_naked_with_smplpytorch(smpl_for_reconst)
                        smpl_vtxs           = smplpytorch_processor_instance.predict_smpl_by_smplpytorch(smpl_for_reconst , Trainer.beta)
                        Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = smpl_vtxs, texture_naked = None , texture_clothes = predicted_texture_clothes , texture_color = predicted_texture_color )  
                    else:
                        predicted_texture_naked   = Trainer_naked.test_step(fileID , smpl_for_reconst)
                        Reconstructor.post_process(fileID , smpl_for_reconst, smpl_vtxs = None, texture_naked = predicted_texture_naked , texture_clothes = predicted_texture_clothes , texture_color = predicted_texture_color )   

                    Reconstructor.post_process(fileID , smpl_for_reconst , predicted_texture_naked , predicted_texture_clothes , predicted_texture_color)   


if __name__=="__main__":
    main()
    
