from operator import is_not
import numpy as np
import sys
import torch
from torchvision import transforms 

from models import net_modules
import os
import cv2

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, type , smpl_folder_path, texture_disp_folder_path =None , texture_color_folder_path =None ,\
                 load_smpl_set_flg = False  ,duplicate_smpl=False , padding1024_flg = False ,\
                 mask_value_disp = 0.5 , mask_value_color = 0.0 , use_skeleton_aware = False):      
        self.type = type
        self.textures_masks = None
        self.use_skeleton_aware = use_skeleton_aware
        if self.type == "clothes" or self.type == "naked" or self.type == "clothes and color" or self.type == "debug":
            print(texture_disp_folder_path)
            self.texture_disp , self.fileIDs , self.textures_masks  = net_modules.load_texture(texture_disp_folder_path ,None , data = "disp" , padding1024_flg = padding1024_flg)
            
            """
            texture_mask_list = []
            for tdp in range(self.texture_disp.shape[0]):
                texture_mask_np_bool = np.all(self.texture_disp[tdp] != mask_value_disp,axis = 0)   
                texture_mask_list.append(texture_mask_np_bool)

            self.textures_masks = np.array(texture_mask_list)
            """

        if self.type == "color" or self.type == "clothes and color"  :
            self.texture_color , self.fileIDs , self.textures_masks = net_modules.load_texture(texture_color_folder_path,None, data = "color" , padding1024_flg = padding1024_flg)  

            """
            if self.textures_masks is None :
                for tdp in range(self.texture_color.shape[0]):
                    texture_mask_np_bool = np.all(self.texture_color[tdp] != mask_value_color,axis = 0)
                    texture_mask_list.append(texture_mask_np_bool)

                self.textures_masks = np.array(texture_mask_list)
            """
                
        if load_smpl_set_flg == True:
            if self.type == "clothes" or self.type == "clothes and color" or self.type == "color" or self.type == "test_fix" or self.type == "debug":
                #self.smpl_sets = net_modules.load_smpl_set(smpl_folder_path , self.fileIDs , fix_hands = True , fix_foots = True , fix_wrists = False)
                self.smpl_sets , self.betas , self.transs  = net_modules.load_smpl_set(smpl_folder_path , self.fileIDs , fix_hands = False , fix_foots = False , fix_wrists = False)
            elif self.type == "naked" or self.type == "test_naked":
                self.smpl_sets , self.betas , self.transs = net_modules.load_smpl_set(smpl_folder_path , self.fileIDs , fix_hands = False , fix_foots = False , fix_wrists = False)
        else:
            if self.type == "clothes" or self.type == "clothes and color" or self.type == "color" or self.type == "test_fix" or self.type == "debug":
                #smpls , smpl_paths = net_modules.load_smpl(smpl_folder_path,None , fix_hands = True , fix_foots = True , fix_wrists = False)
                smpls , smpl_paths , self.betas , self.transs = net_modules.load_smpl(smpl_folder_path,None , fix_hands = False , fix_foots = False , fix_wrists = False)
            elif self.type == "naked" or self.type == "test_naked":
                smpls , smpl_paths , self.betas , self.transs = net_modules.load_smpl(smpl_folder_path,None , fix_hands = False , fix_foots = False , fix_wrists = False)
            if duplicate_smpl == True:
                self.smpl_sets  , self.fileIDs = net_modules.make_smpl_set(smpls , mode = "duplicate" , smpl_paths = smpl_paths)
            else:
                self.smpl_sets  , self.fileIDs = net_modules.make_smpl_set(smpls , mode = "stairs"    , smpl_paths = smpl_paths)

        if self.type == "clothes" or self.type == "naked" or self.type == "clothes and color" or self.type == "debug" :
            if self.smpl_sets.shape[0] != self.texture_disp.shape[0]:
                raise AssertionError("datanum is different between smpl and texture_disp\nsmpl : "  , self.smpl_sets.shape[0] , "texture_disp : " , self.texture_disp.shape[0] )

        if self.type == "color" or self.type == "clothes and color"  :
            if self.smpl_sets.shape[0] != self.texture_color.shape[0]:
                raise AssertionError("datanum is different between smpl and texture_color\nsmpl : " , self.smpl_sets.shape[0] , "texture_color : " , self.texture_color.shape[0] )

        self.datanum = self.smpl_sets.shape[0]

        if self.use_skeleton_aware:
            self.smpl_sets = self.smpl_sets.reshape(-1,24*3,3)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_smpl_set = self.smpl_sets[idx]
        fileID       = self.fileIDs[idx]      
        beta        = self.betas[idx]
        trans       = self.transs[idx]
        
        if self.type == "test_fix" or self.type == "test_naked" :    
            return (fileID , out_smpl_set , beta , trans)
        elif self.type == "clothes" or self.type == "naked":
            out_texture_disp = self.texture_disp[idx]
            textures_masks   = self.textures_masks[idx]
            return (fileID , out_smpl_set , out_texture_disp , textures_masks) #, beta , trans)
        elif self.type == "color":        
            out_texture_color = self.texture_color[idx]
            textures_masks   = self.textures_masks[idx]
            return (fileID , out_smpl_set , out_texture_color , textures_masks) #, beta , trans)
        elif self.type == "clothes and color":       
            #texture_disp_mu = self.texture_disp_mu
            textures_masks = self.textures_masks[idx]
            out_texture_disp = self.texture_disp[idx]
            out_texture_color = self.texture_color[idx]
            return (fileID , out_smpl_set , out_texture_disp , out_texture_color , textures_masks) #, beta , trans)
        elif self.type == "debug":     
            out_texture_disp = self.texture_disp[idx]
            textures_masks   = self.textures_masks[idx]
            return (fileID , out_smpl_set , out_texture_disp , textures_masks , beta , trans)  
        else:
            raise AssertionError("self.type is strange" )

if __name__ == "__main__":
    
    #transform = transforms.Compose([transforms.ToTensor()])
    smpl_folder_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\smplparams_centered"
    texture_disp_folderpath_train = r"D:\Data\Human\HUAWEI\Iwamoto\data\texture_globalbasis"
    #dataset = Dataloader(smpl_folder_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\smplparams_centered",texture_disp_folder_path= r"D:\Data\Human\HUAWEI\Iwamoto\data\texture_globalbasis", texture_color_folder_path = None,  multi_smpl_flg=True , testTextureID_folder_path =None, use_data_cnt=3)
    dataset = Dataloader(smpl_folder_path = smpl_folder_path,texture_disp_folder_path= texture_disp_folderpath_train,texture_color_folder_path = None, skining_weight_folder_path = None,multi_smpl_flg = True,testTextureID_folder_path = None ,use_data_cnt = 1,use_sw_preprocess=False ,IDs=None , duplicate_smpl=False)
    print("dataset[0] : " , dataset[0])
    print("dataset len : " , len(dataset))
    datasets = torch.utils.data.ConcatDataset([dataset,dataset])
    #print("datasets len : " , len(datasets))
    

    #dataset = Dataloader(smpl_folder_path = r"D:\Data\Human\ARTICULATED\I_jumping\smplparams_centered",texture_folder_path= r"D:\Data\Human\ARTICULATED\I_jumping\displacement_texture_base0" )
    #dataset = Dataloader(smpl_folder_path = r"D:\Data\Human\ARTICULATED\I_jumping\smplparams_centered",texture_folder_path= r"D:\Data\Human\ARTICULATED\I_jumping\displacement_texture_base0" ,multi_smpl_flg=True)
    #dataset = Disc_Dataloader(texture_disp_folder_path= r"D:\Data\Human\HUAWEI\Iwamoto\data\texture_globalbasis" )
    #dataset = Partial_Dataloader(texture_disp_folder_path= r"D:\Data\Human\HUAWEI\Iwamoto\data\displacement_texture_base0_0.15", texture_color_folder_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\color_texture_base0_0.15" , part = "Lhand" ,use_data_cnt = 3)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=10, shuffle=False)
    for datas in dataloader:
        print(datas[0])
    #tmp = dataloader.__iter__()
    #label = tmp.next()

    #print(dataset.datanum)
    #print(len(dataset))
    #print(dataset[0][0][10])
    #print(dataset[0][0][10])







    """
    train_size = int( len(dataset) * 0.8 ) # 教師データのサイズ 全体の80%とする
    val_size = len(dataset) - train_size  # テスト用データのサイズ
    #trainデータセットとvalデータセットにランダムで分ける
    dataset_train , dataset_val =  torch.utils.data.random_split(
        dataset,
        [train_size, val_size ],
        #[train_size, val_size , test_size], #if need 3 data typeds
        generator=torch.Generator().manual_seed(0)  # 乱数シードの固定
    )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=10, shuffle=True )
    print("all_data:",dataset.datanum)
    print(len(dataloader_train))
    for datas in dataloader_train:
        #print(len(datas))
        print(datas[0].shape)
        print(datas[1].shape)
        break

    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=10, shuffle=False)
    print(len(dataloader_val))
    for datas in dataloader_val:
        #print(len(datas))
        print(datas[0].shape)
        print(datas[1].shape)
        break"""

    #dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True )
    """for datas in dataloader_test:
        print(datas.shape)
        break
    """
    """
    for i in range(20):
        datas = next(iter(dataloader_test))
        print(idx)
        print(datas.shape)
    print()
    print()
    """