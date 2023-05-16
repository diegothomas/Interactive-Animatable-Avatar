import glob
import os
import argparse
import sys
import numpy as np

def load_ply(path):
    """
    return -> vtx , nml , rgb , face , vtx_num , face_num
    """
    print(path)
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
    return vtx , nml , rgb , face , vtx_num , face_num

def save_ply(path, vtx , nml = None , rgb = None, face = None , vtx_num = 0 , face_num = 0):
    if nml != None and len(vtx) != len(nml):
        print("vtx num and nml num is different")
        print("please check you don't use nml from obj")
        sys.exit()
        
    if rgb != None and len(vtx) != len(rgb):
        print("vtx_num and rgb num is different")
        print("len(vtx) : " , len(vtx))
        print("len(rgb) : " , len(rgb))
        sys.exit()
    
    f = open(path,"w")
    
    header1 = [
        "ply",
        "format ascii 1.0",
        "comment ply file created by Kitamrua",
        "element vertex " + str(vtx_num),
        "property float x",
        "property float y",
        "property float z"
    ]
    
    header_nml = []
    if nml != None:
        header_nml = [
            "property float nx",
            "property float ny",
            "property float nz"
        ]

    header_rgb = []
    if rgb != None:
        header_rgb =[
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ]

    header_face = []
    if face != None:
        header_face = [
            "element face " + str(face_num),
            "property list uchar int vertex_index"
        ]

    header2 = [
        "end_header"
    ]

    vtx_lines = []
    if nml != None and rgb != None:
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2]) + " " + str(nml[i][0]) + " "  + str(nml[i][1]) + " "  + str(nml[i][2]) + " " + str(rgb[i][0]) + " "  + str(rgb[i][1]) + " "  + str(rgb[i][2]))

    elif nml != None :
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2]) + " " + str(nml[i][0]) + " "  + str(nml[i][1]) + " "  + str(nml[i][2]))

    elif rgb != None:
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2]) + " " + str(rgb[i][0]) + " "  + str(rgb[i][1]) + " "  + str(rgb[i][2]))
    else :
        for i in range(vtx_num):
            vtx_lines.append(str(vtx[i][0]) + " "  + str(vtx[i][1]) + " "  + str(vtx[i][2])) 

    face_lines = []
    if face != None:
        for i in range(face_num):
            face_lines.append(str(3) + " " + str(face[i][0]) + " "  + str(face[i][1]) + " "  + str(face[i][2]))

    eof = ['']
    lines = header1 + header_nml + header_rgb + header_face + header2 + vtx_lines + face_lines + eof
    f.write("\n".join(lines))

parser = argparse.ArgumentParser(description='Fit SMPL body to vertices.')
parser.add_argument('--path', type=str, default="D:/Human_data/4D_data/data/",
                    help='Path to the 3D skeleton that corresponds to the 3D scan.')

parser.add_argument('--smpl_flg', type=str, default="False",
                    help='True or False')

parser.add_argument('--smplparams_name', type=str, default="smplparams",
                    help='True or False')
                    
args = parser.parse_args()
path = args.path

if args.smpl_flg == "True":
    smpl_flg = True
elif args.smpl_flg == "False":
    smpl_flg = False
else:
    print("select from True of False")

smplparams_name = args.smplparams_name

if smpl_flg:
    print(os.path.join(path , "Fittedsmpl_pose","PosedMesh_[0-9][0-9][0-9][0-9].ply"))
    fittedmesh_paths = glob.glob(os.path.join(path , "Fittedsmpl_pose","PosedMesh_[0-9][0-9][0-9][0-9].ply"))
    save_dir_path    = os.path.join(path , "Fittedsmpl_pose")
else:
    fittedmesh_paths = glob.glob(os.path.join(path , "FittedMesh_pose","PosedMesh_[0-9][0-9][0-9][0-9].ply"))
    save_dir_path    = os.path.join(path , "FittedMesh_pose")


trans_dir_path   = os.path.join(path , smplparams_name)

for fittedmesh_path in fittedmesh_paths:
    print(id)
    id = str(os.path.basename(fittedmesh_path).split(".")[0].split("_")[-1]).zfill(4)

    print("load fittedmesh : " , fittedmesh_path)
    vtx , nml , rgb , face , vtx_num , face_num = load_ply(fittedmesh_path)
    vtx = np.array(vtx)

    tran_path = os.path.join(trans_dir_path , "trans_" + id + ".bin")
    trans = np.fromfile(tran_path , np.float32)

    valid_id = np.nonzero(np.any((vtx != np.zeros(3)) , axis = 1))
    vtx[valid_id] = vtx[valid_id] - trans

    save_path = os.path.join(save_dir_path , "IniPosiPosedMesh_" + id +".ply")
    print("save_path : " , save_path)
    save_ply(save_path , vtx , None , None , face , vtx_num , face_num)