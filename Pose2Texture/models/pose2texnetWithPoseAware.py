from matplotlib.pyplot import axis
import torch
import torch.nn as nn
from torchsummary import summary
from torchinfo import summary
import sys
import numpy as np
from models import net_modules
import os

if __name__ == "__main__":
    import Generater
else:
    from models import Generater

from external.poseAware.models.enc_and_dec import Encoder , Decoder , StaticEncoder
from external.poseAware import option_parser
from external.poseAware.models.skeleton import build_edge_topology

class pose2texnet(nn.Module):
    def __init__(self, device , type  , args , topology , separate_flg, texture_resolution ,  multi_topology_flg , prediction_texture_mask):
        super(pose2texnet,self).__init__()

        self.device = device
        self.type = type
        self.args = args
        self.separate_flg = separate_flg
        self.multi_topology_flg = multi_topology_flg
        self.prediction_texture_mask = prediction_texture_mask

        """
        self.fc_seq = nn.Sequential(
        #fc1
        nn.Linear(24*3, 128) ,
        nn.ReLU(inplace=False),
        #fc2
        nn.Linear(128, 256),
        nn.ReLU(inplace=False),
        #fc3
        nn.Linear(256, 512),
        nn.ReLU(inplace=False),
        #fc4 
        nn.Linear(512, 1024),
        nn.ReLU(inplace=False),
        #fc5
        nn.Linear(1024, 2048),
        )
        
        self.fc_seq_multi = nn.Sequential(
        #fc1
        nn.Linear(24*3*3, 256),
        nn.ReLU(inplace=False),
        #fc2
        nn.Linear(256, 512),
        nn.ReLU(inplace=False),
        #fc3 
        nn.Linear(512, 1024),
        nn.ReLU(inplace=False),
        #fc4
        nn.Linear(1024, 2048),
        )
        """

        self.fc_poseAware = self.fc_layer(feature_in = 84, feature_out = 2048)

        self.fc_poseAware_cond = self.fc_layer(feature_in = 84, feature_out = 1024)

        self.dropout = nn.Dropout(0.3)   
        #layer = 2
        """
        self.fc_poseAware_multi0 = nn.Sequential(
            #fc1
            nn.Linear(48, 512),
            nn.ReLU(inplace=False),
            #fc2
            nn.Linear(512, 2048),
            )
        """

        self.fc_poseAware_concateneted = self.fc_layer_concateneted(2048*6,2048) 

        #layer = 1
        self.fc_poseAware_multi0 = self.fc_layer_multi1(48,2048)
        self.fc_poseAware_multi1 = self.fc_layer_multi1(24,2048)
        self.fc_poseAware_multi2 = self.fc_layer_multi1(24,2048)
        self.fc_poseAware_multi3 = self.fc_layer_multi1(72,2048)
        self.fc_poseAware_multi4 = self.fc_layer_multi1(24,2048)
        self.fc_poseAware_multi5 = self.fc_layer_multi1(24,2048)
        #layer = 2
        """
        self.fc_poseAware_multi3 = nn.Sequential(
            nn.Linear(72, 512),
            nn.ReLU(inplace=False),
            #fc2
            nn.Linear(512, 2048),
            )
        """
        self.fc_poseAware_multi_cond0 = self.fc_layer_multi1(48,1024)
        self.fc_poseAware_multi_cond1 = self.fc_layer_multi1(24,1024)
        self.fc_poseAware_multi_cond2 = self.fc_layer_multi1(24,1024)
        self.fc_poseAware_multi_cond3 = self.fc_layer_multi1(72,1024)
        self.fc_poseAware_multi_cond4 = self.fc_layer_multi1(24,1024)
        self.fc_poseAware_multi_cond5 = self.fc_layer_multi1(24,1024)

        self.poseAware_multi_conv = nn.Sequential(
            #conv1_1
            nn.Conv2d(18, 18,
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(18),
            nn.ReLU(inplace=True),

            #conv2_1
            nn.Conv2d(18, 6,
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),

            #conv3_1
            nn.Conv2d(6, 3,
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),

            #conv3_1
            nn.Conv2d(3, 3,
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),

            #conv4_1
            nn.Conv2d(3, 3,
                    kernel_size=3, stride=1, padding=1, bias=False),
            #nn.ReLU(inplace=True)
            nn.Sigmoid(),
        )
        
        #Encoder layer : in :(256,256,3)
        self.enc1 = nn.Sequential(            
        nn.Conv2d(3, 64, 4, stride=2 ,padding=1)).to(device)    #(128,128,64)  

        self.enc2 = self.enc_mid(64, 128).to(device)     #(64,64,128)
        self.enc3 = self.enc_mid(128, 256).to(device)    #(32,32,256)
        self.enc4 = self.enc_mid(256, 512).to(device)    #(16,16,512)
        self.enc5 = self.enc_mid(512, 512).to(device)    #(8,8,512)
        self.enc6 = self.enc_mid(512, 512).to(device)    #(4,4,512)
        self.enc7 = self.enc_mid(512, 1024).to(device)    #(2,2,1024)

        self.enc8 = nn.Sequential( 
        nn.LeakyReLU(inplace=False),
        nn.Conv2d(1024, 1024, 4, stride=2 ,padding=1)).to(device)    #(1,1,1024))     

        self.enc_all = nn.Sequential(self.enc1,self.enc2,self.enc3,self.enc4,self.enc5,self.enc6,self.enc7,self.enc8)

        self.args.kernel_size = 3
        #print("args.skeleton_info:",self.args.skeleton_info)
        self.args.skeleton_info = ""      #should comment out if use offsets
        self.args.rotation = "euler_angle"
        #args.extra_conv = 0
        #self.static_encoder = StaticEncoder(self.args, topology)
        self.args.num_layers = 2
        if self.multi_topology_flg:
            self.SkeletonAwareEncoder_list = []
            self.SkeletonAwareGenerater_list = []
            topology , joints_group_list = net_modules.split_edges(topology)
            self.joints_group_list = joints_group_list
            
            #self.args.num_layers = 2
            self.args.num_layers = 1
            self.SkeletonAwareEncoder_list.append(Encoder(self.args, topology[0]).to(device)) #.to(device)    
            self.args.num_layers = 1
            self.SkeletonAwareEncoder_list.append(Encoder(self.args, topology[1]).to(device)) #.to(device)
            self.args.num_layers = 1
            self.SkeletonAwareEncoder_list.append(Encoder(self.args, topology[2]).to(device)) #.to(device)
            #self.args.num_layers = 2
            self.args.num_layers = 1
            self.SkeletonAwareEncoder_list.append(Encoder(self.args, topology[3]).to(device)) #.to(device)
            self.args.num_layers = 1
            self.SkeletonAwareEncoder_list.append(Encoder(self.args, topology[4]).to(device)) #.to(device)
            self.args.num_layers = 1
            self.SkeletonAwareEncoder_list.append(Encoder(self.args, topology[5]).to(device)) #.to(device)    
            for i in range(6):        
                self.SkeletonAwareGenerater_list.append(Generater.Generater(2048,128,output_dim = 3 ,texture_resolution=texture_resolution).to(device)) #.to(device)
        else:
            self.SkeletonAwareEncoder = Encoder(self.args, topology) #.to(device)
        
        if self.separate_flg == False:
            if self.type == "clothes" or self.type == "naked":          
                self.Generater = Generater.Generater(2048,128,output_dim = 3+0+0 ,texture_resolution=texture_resolution) #.to(device)
            elif self.type == "color":          
                self.Generater = Generater.Generater(2048,128,output_dim = 0+3+0 ,texture_resolution=texture_resolution) #.to(device)
            elif self.type == "clothes_and_color":          
                self.Generater = Generater.Generater(2048,128,output_dim = 3+3+0 ,texture_resolution=texture_resolution) #.to(device)
        else:
            if self.type == "clothes" or self.type == "naked":       
                self.Generater_disp = Generater.Generater(2048,128,output_dim = 3 ,texture_resolution=texture_resolution , prediction_texture_mask = self.prediction_texture_mask) #.to(device)
            elif self.type == "color":          
                self.Generater_color = Generater.Generater(2048,128,output_dim = 3 ,texture_resolution=texture_resolution , prediction_texture_mask = self.prediction_texture_mask) #.to(device)
            elif self.type == "clothes_and_color":          
                self.Generater_disp  = Generater.Generater(2048,128,output_dim = 3 ,texture_resolution=texture_resolution , prediction_texture_mask = self.prediction_texture_mask) #.to(device)
                self.Generater_color = Generater.Generater(2048,128,output_dim = 3 ,texture_resolution=texture_resolution , prediction_texture_mask = self.prediction_texture_mask) #.to(device)
            

    def enc_mid(self , feature_in , feature_out):
            layers = []
            layers.append(nn.LeakyReLU(inplace=False))
            layers.append(nn.Conv2d(feature_in, feature_out, 4, stride=2 ,padding=1))  
            layers.append(nn.BatchNorm2d(feature_out))  
            return nn.Sequential(*layers)

    def fc_layer(self , feature_in , feature_out):
        out_fc = nn.Sequential(
        #fc1
        #fc1
        nn.Linear(feature_in, 512),
        nn.ReLU(inplace=False),
        #fc2
        nn.Linear(512, feature_out),
        )
        return out_fc
    
    def fc_layer_multi1(self , feature_in , feature_out):
        out_fc = nn.Sequential(
                nn.Flatten(),
                #fc1
                nn.Linear(feature_in, 512),
                nn.ReLU(inplace=False),
                #fc2
                nn.Linear(512, feature_out),
                )
        return out_fc

    def fc_layer_concateneted(self , feature_in , feature_out):
        out_fc = nn.Sequential(
                nn.Flatten(),
                #fc1
                nn.Linear(feature_in, 2048*3),
                nn.ReLU(inplace=False),
                #fc2
                nn.Linear(2048*3, feature_out),
                )
        return out_fc

    """
    def fc_layer_multi1(self , feature_in , feature_out):
        out_fc = nn.Sequential(
                nn.Flatten(),
                #fc1
                nn.Linear(feature_in, 256),
                nn.ReLU(inplace=False),
                #fc1
                nn.Linear(256, 512),
                nn.ReLU(inplace=False),
                #fc1
                nn.Linear(512, 1024),
                nn.ReLU(inplace=False),
                #fc2
                nn.Linear(1024, feature_out),
                )
        return out_fc
    """

    def forward(self,smpl , condition_mu_texture = None):
        
        #kt_path = r"D:\Data\Human\Template-star-0.015\kintree.bin"
        #kintree = net_modules.load_ktfile(kt_path)

        #t_pose_path = r"D:\Data\Human\HUAWEI\Iwamoto\data\smplparams_centered_new\T_joints.bin"
        #t_pose ,t_joint_length = net_modules.LoadStarSkeleton(t_pose_path,kintree)

        #t_pose = np.array([t_pose])
        #print(t_pose.shape)
        #t_pose = torch.from_numpy(t_pose.astype(np.float32)).to(device)

        """
        offset_idx = 0
        self.offset_repr = []
        self.offset_repr.append(self.static_encoder(t_pose.to(device)))
        self.offset_repr = [self.static_encoder(t_pose)]
        offsets = [self.offset_repr[0][p][offset_idx] for p in range(self.args.num_layers + 1)]     #####
        """
        offsets = None       #comment out if use offset
        
        if self.multi_topology_flg:
            x_list1 = []
            for i in range(6):
                joints_group = self.joints_group_list[i]
                joints_group_for_smpl = []
                for j in joints_group:
                    joints_group_for_smpl.append(j)
                    joints_group_for_smpl.append(j+1)
                    joints_group_for_smpl.append(j+2)
                x_list1.append(self.SkeletonAwareEncoder_list[i](smpl[:,joints_group_for_smpl],offsets))  #[1,84,1]
            
            x_list2 = []
            #x_list1[0] = torch.squeeze(x_list1[0],-1)
            if condition_mu_texture != None:
                cMu = self.enc_all(condition_mu_texture)
                cMu = torch.squeeze(cMu , -1)
                cMu = torch.squeeze(cMu , -1)
                x_tmp0 = self.fc_poseAware_multi_cond0(x_list1[0])
                x_list2.append(torch.cat([cMu , x_tmp0] , dim = 1))               

                x_tmp1 = self.fc_poseAware_multi_cond1(x_list1[1])
                x_list2.append(torch.cat([cMu , x_tmp1] , dim = 1)) 

                x_tmp2 = self.fc_poseAware_multi_cond2(x_list1[2])
                x_list2.append(torch.cat([cMu , x_tmp2] , dim = 1))

                x_tmp3 = self.fc_poseAware_multi_cond3(x_list1[3])
                x_list2.append(torch.cat([cMu , x_tmp3] , dim = 1))  

                x_tmp4 = self.fc_poseAware_multi_cond4(x_list1[4])
                x_list2.append(torch.cat([cMu , x_tmp4] , dim = 1))   

                x_tmp5 = self.fc_poseAware_multi_cond5(x_list1[5])
                x_list2.append(torch.cat([cMu , x_tmp5] , dim = 1))
            else:
                x_list2.append(self.fc_poseAware_multi0(x_list1[0]))
                x_list2.append(self.fc_poseAware_multi1(x_list1[1]))
                x_list2.append(self.fc_poseAware_multi2(x_list1[2]))

                x_list2.append(self.fc_poseAware_multi3(x_list1[3]))
                x_list2.append(self.fc_poseAware_multi4(x_list1[4]))
                x_list2.append(self.fc_poseAware_multi5(x_list1[5]))
            
            #concatenate on 1d feature
            x = torch.cat(x_list2,axis = 1)
            x = self.fc_poseAware_concateneted(x)
            x = torch.unsqueeze(x,-1)
            x = torch.unsqueeze(x,-1)
            output_disp = self.Generater_disp(x) 
            return output_disp  
        else:
            x = self.SkeletonAwareEncoder(smpl,offsets)  #[1,84,1]

            x = torch.squeeze(x,-1)

            if condition_mu_texture != None:
                x = self.fc_poseAware_cond(x)
                cMu = self.enc_all(condition_mu_texture)
                cMu = torch.squeeze(cMu , -1)
                cMu = torch.squeeze(cMu , -1)
                x = torch.cat([cMu , x] , dim = 1)
            else:
                x = self.fc_poseAware(x)
            
            x = torch.unsqueeze(x,-1)
            x = torch.unsqueeze(x,-1)

            if self.separate_flg == False:
                output = self.Generater(x)    
                
                if self.type == "clothes" or self.type == "naked":          
                    return output
                elif self.type == "color" :          
                    return output
                elif self.type == "clothes_and_color":
                    out_disp = output[:,0:3,:,:] 
                    out_color = output[:,3:3+3,:,:] 
                    return out_disp ,out_color
                else:
                    raise AssertionError("not defined pose2texnet output")
            else:
                if self.type == "clothes" or self.type == "naked":    
                    if self.prediction_texture_mask:
                        output_disp , output_disp_mask = self.Generater_disp(x) 
                        return output_disp , output_disp_mask
                    else:
                        output_disp = self.Generater_disp(x) 
                        return output_disp
                elif self.type == "color":   
                    output_color = self.Generater_color(x) 
                    return output_color
                elif self.type == "clothes_and_color":   
                    output_disp = self.Generater_disp(x) 
                    output_color = self.Generater_color(x) 
                    return output_disp ,output_color
                else:
                    raise AssertionError("not defined pose2texnet output")

class discriminater(nn.Module):
    def __init__(self,device,part):
        super(discriminater,self).__init__()
        self.device = device

        self.conv_discriminater = nn.Sequential(
        #cnn1
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),   # in:(3,224*224)、out:(64,112*112)
        nn.BatchNorm2d(64),
        nn.ReLU()
        )

        n_features = 64
        self.dcgan_discriminater = nn.Sequential(
        nn.Conv2d(3, n_features,
                      kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # conv2
        nn.Conv2d(n_features, n_features * 2,
                    kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(n_features * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # conv3
        nn.Conv2d(n_features * 2, n_features * 4,
                    kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(n_features * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # conv4
        nn.Conv2d(n_features * 4, n_features * 8,
                    kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(n_features * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # conv5
        nn.Conv2d(n_features * 8, 1,
                    kernel_size=4, stride=2, padding=0, bias=False),
        nn.Sigmoid(),
        nn.Flatten(),
        # fmt: on
        )

    def forward(self,body_part):
        body_part = self.dcgan_discriminater(body_part)
        return body_part

if __name__ == "__main__":
    args = option_parser.get_args()
    kt_path = r"D:\Data\Human\Template-star-0.015\kintree.bin"
    kintree = net_modules.load_ktfile(kt_path)
    print("kintree:",kintree)
    joint_topology = net_modules.GetParentFromKintree(kintree)
    print("joint_topology")
    print(type(joint_topology))
    print(joint_topology)

    edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
    print("edges")
    print(len(edges))
    print(edges)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_gen = pose2texnet(device,True,1,args,edges , "True")
    summary(model_gen)
    print("aaa")
    summary(model_gen,[(1,72,3)])

    #x = torch.rand(1,72,3).to(device)
    #y = model_gen(x)
    #print(y.shape)

    #model_disc = discriminater(device,"face")
    #summary(model_disc,[(1,3,410,410)])
    #summary(model_disc,[(1,3,264,264)])
    #print(x.shape)

    """
    x = torch.rand(1,24*3*3)

    crierion = nn.BCELoss()
    txr = model_gen(x)
    face = model_disc(txr)
    print(face.shape)
    y_real = torch.full_like(face, 1)
    loss_real = crierion(face, y_real)
    print(loss_real)