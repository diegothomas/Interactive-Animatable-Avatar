import sys
sys.path.append(r"D:\Project\Human\Pose2Texture\Network\lib\models")
sys.path.append(r"D:\Project\Human\Pose2Texture\Network\lib\models\poseAware")

from datasets import create_dataset, get_character_names
import option_parser
from external.poseAware.models.skeleton import build_edge_topology
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(r"D:\Project\Human\Pose2Texture\Network")

def test_create():
    args = option_parser.get_args()
    characters = get_character_names(args)
    print("characters")
    print(characters)

    #characters00 = [[characters[0][0]]]
    characters00 = characters

    print(characters00)
    dataset = create_dataset(args, characters00)

    print("len(dataset):",len(dataset))                            #total window num of datasets1(['Aj', 'BigVegas', 'Kaya', 'SportyGranny'])
    print("len(dataset[0]):",len(dataset[0]))                        #2042 -> dataset[x] = a one window
    print("len(dataset[0][0]:",len(dataset[0][0]))                      #
    print("len(dataset[0][1]:",len(dataset[0][1]))
    print("dataset[0][0][0].shape:",dataset[0][0][0].shape)        #[91,64] -> dataset[0][0][x] = [22*4+3 + 64]  (4 = quarternion , 3 = rootJoint(x,t,z)? , 22 = joint_num , 64 = window_size)
    print("dataset[:][0][1]:",dataset[:][0][1])                    #offset_idx(one human correspond one offset_idx) (for example,"AJ" has offset_idx "0" and "BigVegas" hs offset_idx "1" ...)
    print("len(dataset[:][0][1]):",len(dataset[:][0][1]))

    print("len(dataset[0][1][0]:",len(dataset[0][1][0]))
    print("len(dataset[0][1][0][0]:",len(dataset[0][1][0][0]))
    print("dataset[0][1][0][0]:",dataset[0][1][0][0])

    
    print("len(dataset.offsets):",len(dataset.offsets))             #2 -> mean 2datasets
    print("dataset.offsets[0].shape:",dataset.offsets[0].shape)     #[4,23,3]   4 = dataset(human) num , 23 = joints num , 3 = x,y,z? #input of static encoder
    print("dataset.offsets[1].shape:",dataset.offsets[1].shape)     #[20,28,3] 20 = dataset(human) num , 28 = joints num , 3 = x,y,z? #input of static encoder
    print("type(dataset.offsets[0]):",type(dataset.offsets[0]))     #[4,23,3]   4 = dataset(human) num , 23 = joints num , 3 = x,y,z? #input of static encoder

    print("dataset.offsets[0]:",dataset.offsets[0][0])              #23joints and [x,y,z] -> maybe initial pose
    print("dataset.joint_topologies:",dataset.joint_topologies)
    print("len(dataset.joint_topologies[0]):",len(dataset.joint_topologies[0]))
    print("len(dataset.joint_topologies[1]):",len(dataset.joint_topologies[1]))


    #3d output
    # Figureを追加
    fig = plt.figure(figsize = (8, 8))

    # 3DAxesを追加
    ax = fig.add_subplot(111, projection='3d')

    # Axesのタイトルを設定
    ax.set_title("", size = 20)

    # 軸ラベルを設定
    ax.set_xlabel("x", size = 30, color = "r")
    ax.set_ylabel("y", size = 30, color = "r")
    ax.set_zlabel("z", size = 30, color = "r")

    # 軸目盛を設定
    #ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
    #ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0])
    offsets_x = dataset.offsets[0][0][:,0].to('cpu').detach().numpy().copy()
    offsets_y = dataset.offsets[0][0][:,1].to('cpu').detach().numpy().copy()
    offsets_z = dataset.offsets[0][0][:,2].to('cpu').detach().numpy().copy()
    
    print(len(offsets_x))
    print(offsets_x)
    print(len(offsets_y))
    print(offsets_y)
    print(len(offsets_z))
    print(offsets_z)

    # 曲線を描画
    ax.scatter(offsets_x, offsets_y, offsets_z, s = 40, c = "blue")
    #ax.scatter(np.array([0,1,2,3]), np.array([0,1,2,3]), np.array([0,1,2,3]), s = 40, c = "blue")


    plt.show()

    """
    motion, offset_idx = dataset[0][0]
    print(len(motion))
    print(offset_idx)
    """
    

    """
    #print(dataset[0])
    print(len(dataset.joint_topologies))
    print(len(dataset.joint_topologies[0]))

    
    print("build_edge_topology")
    print(type(dataset.joint_topologies[0]))
    print(dataset.joint_topologies[0])
    print(type(dataset.joint_topologies[0][0]))
    print(dataset.joint_topologies[0][0])
    edges = build_edge_topology(dataset.joint_topologies[0], torch.zeros((len(dataset.joint_topologies[0]), 3)))

    print("edges")
    print(len(edges))
    print(edges)
    print(type(dataset.joint_topologies[0][0]))
    """

    """
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for step, motions in enumerate(data_loader):
        motion, offset_idx = motions[0]                                              #####
        print("len(offset_idx):",len(offset_idx))
        print(offset_idx)
    """
    
    #return edges


if __name__ == "__main__":
    test_create()