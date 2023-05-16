import torch.nn as nn
import torch
from torchsummary import summary
from torchinfo import summary

class Generater(nn.Module):
    def __init__(self,input_dim,n_features , output_dim = 3 ,texture_resolution=256 , prediction_texture_mask = False):
        super(Generater,self).__init__()
        self.n_features = n_features
        self.output_dim = output_dim
        self.texture_resolution = texture_resolution
        self.prediction_texture_mask = prediction_texture_mask

        self.main256 = nn.Sequential(                                                      #1,1,2048
        # conv 1
        nn.ConvTranspose2d(input_dim,n_features * 8, 
                            kernel_size= 4 , stride = 1 , padding = 0, bias = False),  #4,4,1024    #bug?
        nn.BatchNorm2d(n_features * 8),
        nn.ReLU(inplace=True),
        
        # conv2
        nn.ConvTranspose2d(n_features * 8, n_features * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),    #8,8,512
        nn.BatchNorm2d(n_features * 4),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv3
        nn.ConvTranspose2d(n_features * 4, n_features * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),    #16,16,256
        nn.BatchNorm2d(n_features * 2),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv4
        nn.ConvTranspose2d(n_features * 2, n_features,
                            kernel_size=4, stride=2, padding=1, bias=False),    #32,32,128
        nn.BatchNorm2d(n_features),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv5
        nn.ConvTranspose2d(n_features , int(n_features/2),
                            kernel_size=4, stride=2, padding=1, bias=False),    #64,64,64
        nn.BatchNorm2d(int(n_features/2)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/8),               #128,128,16
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/8) , output_dim,                      #256,256,3
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.BatchNorm2d(int(n_features/8)),
        #nn.ReLU(inplace=True)
        nn.Sigmoid()
        )

        """
        # conv8
        nn.ConvTranspose2d(int(n_features/8) , int(n_features/16),              #512,512,8
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/16)),
        nn.ReLU(inplace=True)
        
        # conv9 (conv5)
        nn.ConvTranspose2d(int(n_features/16), output_dim,                      #1024,1024,out_dim(=3 or 6)
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.Tanh()
        nn.ReLU(inplace=True)
        """

        """
        self.main531 = nn.Sequential(                                                      #1,1,2048
        # conv 1
        nn.ConvTranspose2d(input_dim,n_features * 8, 
                            kernel_size= 4 , stride = 1 , padding = 0, bias = False),  #4,4,1024    #bug?
        nn.BatchNorm2d(n_features * 8),
        nn.ReLU(inplace=True),
        
        # conv2
        nn.ConvTranspose2d(n_features * 8, n_features * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),    #8,8,512
        nn.BatchNorm2d(n_features * 4),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv3
        nn.ConvTranspose2d(n_features * 4, n_features * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),    #16,16,256
        nn.BatchNorm2d(n_features * 2),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv4
        nn.ConvTranspose2d(n_features * 2, n_features,
                            kernel_size=4, stride=2, padding=1, output_padding = 1 , bias=False),    #33,33,128
        nn.BatchNorm2d(n_features),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv5
        nn.ConvTranspose2d(n_features , int(n_features/2),
                            kernel_size=4, stride=2, padding=1, bias=False),    #66,66,64
        nn.BatchNorm2d(int(n_features/2)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/8),               #132,132,16
                            kernel_size=4, stride=2, padding=1 , bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/8) , int(n_features/16),               #265,265,8
                            kernel_size=4, stride=2, padding=1, output_padding= 1 , bias=False),
        nn.BatchNorm2d(int(n_features/16)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/16) , output_dim,                      #531,531,3
                            kernel_size=4, stride=2, padding=1, output_padding = 1 , bias=False),
        nn.Sigmoid()
        )
        """

        self.main531 = nn.Sequential(                                                      #1,1,2048
        # conv 1
        nn.ConvTranspose2d(input_dim,n_features * 8, 
                            kernel_size= 4 , stride = 1 , padding = 0, bias = False),  #4,4,1024    #bug?
        nn.BatchNorm2d(n_features * 8),
        nn.ReLU(inplace=True),
        
        # conv2
        nn.ConvTranspose2d(n_features * 8, n_features * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),    #8,8,512
        nn.BatchNorm2d(n_features * 4),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv3
        nn.ConvTranspose2d(n_features * 4, n_features * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),    #16,16,256
        nn.BatchNorm2d(n_features * 2),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv4
        nn.ConvTranspose2d(n_features * 2, n_features,
                            kernel_size=5, stride=2, padding=1, bias=False),    #33,33,128
        nn.BatchNorm2d(n_features),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv5
        nn.ConvTranspose2d(n_features , int(n_features/2),
                            kernel_size=4, stride=2, padding=1, bias=False),    #66,66,64
        nn.BatchNorm2d(int(n_features/2)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/8),               #132,132,16
                            kernel_size=4, stride=2, padding=1 , bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/8) , int(n_features/16),               #265,265,8
                            kernel_size=5, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/16)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/16) , output_dim,                      #531,531,3
                            kernel_size=5, stride=2, padding=1, bias=False),
        nn.Sigmoid()
        )


        self.main768 = nn.Sequential(                                                      #1,1,2048
        # conv 1
        nn.ConvTranspose2d(input_dim,n_features * 8, 
                            kernel_size= 4 , stride = 1 , padding = 0, bias = False),  #4,4,1024
        nn.BatchNorm2d(n_features * 8),
        nn.ReLU(inplace=True),
        
        # conv2
        nn.ConvTranspose2d(n_features * 8, n_features * 4,
                            kernel_size=4, stride=3, padding=1, output_padding = 1 , bias=False),    #12,12,512
        nn.BatchNorm2d(n_features * 4),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv3
        nn.ConvTranspose2d(n_features * 4, n_features * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),    #24,24,256
        nn.BatchNorm2d(n_features * 2),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv4
        nn.ConvTranspose2d(n_features * 2, n_features,
                            kernel_size=4, stride=2, padding=1, bias=False),    #48,48,128
        nn.BatchNorm2d(n_features),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv5
        nn.ConvTranspose2d(n_features , int(n_features/2),
                            kernel_size=4, stride=2, padding=1, bias=False),    #96,96,64
        nn.BatchNorm2d(int(n_features/2)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/8),               #192,192,16
                            kernel_size=4, stride=2, padding=1 , bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/8) , int(n_features/16),               #384,384,8
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/16)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        #conv 8
        nn.ConvTranspose2d(int(n_features/16), output_dim, 
                            kernel_size=4, stride=2, padding=1, bias=False), #768,768,3
        nn.Sigmoid()
        )
        
        """
        self.main1024 = nn.Sequential(                                                      #1,1,2048
        # conv 1
        nn.ConvTranspose2d(input_dim,n_features * 8, 
                            kernel_size= 4 , stride = 1 , padding = 0, bias = False),  #4,4,1024    #bug?
        nn.BatchNorm2d(n_features * 8),
        nn.ReLU(inplace=True),
        
        # conv2
        nn.ConvTranspose2d(n_features * 8, n_features * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),    #8,8,512
        nn.BatchNorm2d(n_features * 4),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv3
        nn.ConvTranspose2d(n_features * 4, n_features * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),    #16,16,256
        nn.BatchNorm2d(n_features * 2),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),
        # conv4
        nn.ConvTranspose2d(n_features * 2, n_features,
                            kernel_size=4, stride=2, padding=1, bias=False),    #32,32,128
        nn.BatchNorm2d(n_features),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv5
        nn.ConvTranspose2d(n_features , int(n_features/2),
                            kernel_size=4, stride=2, padding=1, bias=False),    #64,64,64
        nn.BatchNorm2d(int(n_features/2)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/4),               #128,128,32
                            kernel_size=4, stride=2, padding=1 , bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/4) , int(n_features/8),               #256,256,16
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(int(n_features/8) , int(n_features/16),               #512,512,8
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/16)),
        #nn.Dropout2d(0.5),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/16) , output_dim,                      #1024,1024,3
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.Sigmoid()
        )
        """
        
        #if output_dim < (n_features/16):
        self.main_pre = nn.Sequential(                                                      #1,1,2048
        # conv 1
        nn.ConvTranspose2d(input_dim,n_features * 8, 
                            kernel_size= 4 , stride = 1 , padding = 0, bias = False),  #4,4,1024    #bug?
        nn.BatchNorm2d(n_features * 8),
        nn.ReLU(inplace=True),
        # conv2
        nn.ConvTranspose2d(n_features * 8, n_features * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),    #8,8,512
        nn.BatchNorm2d(n_features * 4),
        nn.ReLU(inplace=True),
        # conv3
        nn.ConvTranspose2d(n_features * 4, n_features * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),    #16,16,256
        nn.BatchNorm2d(n_features * 2),
        nn.ReLU(inplace=True),
        # conv4
        nn.ConvTranspose2d(n_features * 2, n_features,
                            kernel_size=4, stride=2, padding=1, bias=False),    #32,32,128
        nn.BatchNorm2d(n_features),
        nn.ReLU(inplace=True),

        # conv5
        nn.ConvTranspose2d(n_features , int(n_features/2),
                            kernel_size=4, stride=2, padding=1, bias=False),    #64,64,64
        nn.BatchNorm2d(int(n_features/2)),
        nn.ReLU(inplace=True)
        )


        self.main1024 = nn.Sequential(
        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/4),               #128,128,32
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/4) , int(n_features/8),               #256,256,16
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        nn.ReLU(inplace=True),

        # conv8
        nn.ConvTranspose2d(int(n_features/8) , int(n_features/16),              #512,512,8
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/16)),
        nn.ReLU(inplace=True),
                
        # conv9 (conv5)
        nn.ConvTranspose2d(int(n_features/16), output_dim,                      #1024,1024,out_dim(=3 or 6)
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.Tanh()
        nn.ReLU(inplace=True)
        )


        self.main1024_pre = nn.Sequential(
        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/4),               #128,128,32
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/4) , int(n_features/8),               #256,256,16
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        nn.ReLU(inplace=True),

        # conv8
        nn.ConvTranspose2d(int(n_features/8) , int(n_features/16),              #512,512,8
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/16)),
        nn.ReLU(inplace=True)
        )

        self.main1024_aft = nn.Sequential(
        # conv9 (conv5)
        nn.ConvTranspose2d(int(n_features/16), output_dim,                      #1024,1024,out_dim(=3 or 6)
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.Tanh()
        nn.ReLU(inplace=True)
        )

        self.main1024_aft_mask = nn.Sequential(
        # conv9 (conv5)
        nn.ConvTranspose2d(int(n_features/16), 1,                      #1024,1024,out_dim(=3 or 6)
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.Sigmoid()
        )

        self.main512 = nn.Sequential(
        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/4),               #128,128,32
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/4) , int(n_features/8),               #256,256,16
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/8)),
        nn.ReLU(inplace=True),

        # conv8
        nn.ConvTranspose2d(int(n_features/8) , output_dim,              #512,512,8
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.BatchNorm2d(int(n_features/16)),
        nn.ReLU(inplace=True)
        )
        
        self.main3= nn.Sequential(
        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/4),               #128,128,32
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/4) , int(n_features/4),               #256,256,32 *
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        nn.ReLU(inplace=True),

        # conv8
        nn.ConvTranspose2d(int(n_features/4) , int(n_features/4),              #512,512,32  *
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        nn.ReLU(inplace=True),
        
        
        # conv9 (conv5)
        nn.ConvTranspose2d(int(n_features/4), output_dim,                      #1024,1024,out_dim(24,3+24,,3+3+24)
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.Tanh()
        nn.ReLU(inplace=True)
        )

        
        self.main_sw = nn.Sequential(                                                      #1,1,2048
        # conv 1
        nn.ConvTranspose2d(input_dim,n_features * 8, 
                            kernel_size= 4 , stride = 1 , padding = 0, bias = False),  #4,4,1024
        nn.BatchNorm2d(n_features * 8),
        nn.ReLU(inplace=True),
        # conv2
        nn.ConvTranspose2d(n_features * 8, n_features * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),    #8,8,512
        nn.BatchNorm2d(n_features * 4),
        nn.ReLU(inplace=True),
        # conv3
        nn.ConvTranspose2d(n_features * 4, n_features * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),    #16,16,256
        nn.BatchNorm2d(n_features * 2),
        nn.ReLU(inplace=True),
        # conv4
        nn.ConvTranspose2d(n_features * 2, n_features,
                            kernel_size=4, stride=2, padding=1, bias=False),    #32,32,128      
        nn.BatchNorm2d(n_features),
        nn.ReLU(inplace=True),

        # conv5
        nn.ConvTranspose2d(n_features , n_features,
                            kernel_size=4, stride=2, padding=1, bias=False),    #64,64,128  *
        nn.BatchNorm2d(int(n_features)),
        nn.ReLU(inplace=True),

        # conv6
        nn.ConvTranspose2d(n_features , int(n_features/2),                 #128,128,64  
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/2)),
        nn.ReLU(inplace=True),

        # conv7
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/2),               #256,256,64 *
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/2)),
        nn.ReLU(inplace=True),

        # conv8
        nn.ConvTranspose2d(int(n_features/2) , int(n_features/4),              #512,512,32      
                            kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(int(n_features/4)),
        nn.ReLU(inplace=True),
        
        
        # conv9 (conv5)
        nn.ConvTranspose2d(int(n_features/4), output_dim,                      #1024,1024,out_dim(24,3+24,,3+3+24)
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.Tanh()
        nn.ReLU(inplace=True)
        )

                
        """
        self.sw_main = nn.Sequential(
        # conv6
        nn.ConvTranspose2d(int(n_features/2) , int(sw_output_dim),               #128,128,24
                            kernel_size=4, stride=2, padding=1, bias=False),
        #nn.Tanh()
        nn.ReLU(inplace=True)
        )
        """
        """
        else:
            self.main = nn.Sequential(
            # conv 1
            nn.ConvTranspose2d(input_dim,n_features * 8, 
                                kernel_size= 4 , stride = 1 , padding = 0, bias = False),
            nn.BatchNorm2d(n_features * 8),
            nn.ReLU(inplace=True),
            # conv2
            nn.ConvTranspose2d(n_features * 8, n_features * 6,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 6),
            nn.ReLU(inplace=True),
            # conv3
            nn.ConvTranspose2d(n_features * 6, n_features * 4,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU(inplace=True),
            # conv4
            nn.ConvTranspose2d(n_features * 4, n_features*3,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features*3),
            nn.ReLU(inplace=True),

            # conv5
            nn.ConvTranspose2d(n_features*3 , n_features*2,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features*2),
            nn.ReLU(inplace=True),

            # conv6
            nn.ConvTranspose2d(n_features*2 , n_features,               #256,256,128
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),

            # conv7
            nn.ConvTranspose2d(n_features , int(n_features/2),               #256,256,64
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(n_features/2)),
            nn.ReLU(inplace=True),

            # conv8
            nn.ConvTranspose2d(int(n_features/2) , int(n_features/4),              #512,512,32
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(n_features/4)),
            nn.ReLU(inplace=True),
            
            # conv9 (conv5)
            nn.ConvTranspose2d(int(n_features/4), output_dim,                      #1024,1024,out_dim
                                kernel_size=4, stride=2, padding=1, bias=False),
            #nn.Tanh()
            nn.ReLU(inplace=True)
            )
        """

    def forward(self,x):
        if self.output_dim == 24 or self.output_dim == 3+24 or self.output_dim == 3+3+24:
            
            #tmp = self.main(x)
            #out = self.main3(tmp)
            
            out = self.main_sw(x)
            return out
        elif self.output_dim == 3 or self.output_dim == 3+3 or self.output_dim == 1 :            
            if self.texture_resolution == 256:
                out = self.main256(x)
            elif self.texture_resolution == 768:
                out = self.main768(x)
            elif self.texture_resolution == 531:
                out = self.main531(x)
            elif self.texture_resolution == 512:
                tmp = self.main_pre(x)
                out = self.main512(tmp)
            elif self.texture_resolution == 1024:   
                tmp = self.main_pre(x)
                if self.prediction_texture_mask:
                    tmp2 = self.main1024_pre(tmp)

                    out  = self.main1024_aft(tmp2)
                    out_mask = self.main1024_aft_mask(tmp2)
                    return out , out_mask
                else:
                    out = self.main1024(tmp)
                    return out


if __name__ == "__main__":
    #100次元のベクトルをinputに
    model = Generater(2048,128 , output_dim=30)
    
    summary(model,[(1,2048,1,1)])

    #1*1のchanel 100のランダムノイズを生成
    x = torch.rand(1,2048,1,1)
    out = model(x)
    print(out.shape)

    #print(out[0])

