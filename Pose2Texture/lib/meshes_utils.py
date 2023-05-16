import sys
import struct
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

def save_skeleton_ply(path, vtx , nml = None , rgb = None, edge = None,  vtx_num = 0 , edge_num = 0):
    if nml != None and len(vtx) != len(nml):
        print("vtx num and nml num is different")
        print("please check you don't use nml from obj")
        sys.exit()
        
    if rgb != None and len(vtx) != len(rgb):
        print("vtx_num and rgb num is different")
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

    header_edge = []
    if edge != None:
        header_edge = [
            "element edge " + str(edge_num),
            "property int vertex1",
            "property int vertex2"
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

    edge_lines = []
    if edge != None:
        for i in range(edge_num):
            if (edge[2 * i] >= 0 and edge[2 * i + 1] >= 0):
                edge_lines.append(str(edge[2 * i]) + " "  + str(edge[2 * i + 1]))
            else:
                edge_lines.append("0" + " " + "0") 

    eof = ['']
    lines = header1 + header_nml + header_rgb + header_edge + header2 + vtx_lines + edge_lines + eof
    f.write("\n".join(lines))

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
    i = 0
    while(1):
        if lines[i] == '':
            #print("end of file")
            #print(i)
            break

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
            faceID2vtxIDset   .append([int(line[1].split("/")[0])-1,int(line[2].split("/")[0])-1,int(line[3].split("/")[0])-1]) 
            faceID2uvIDset    .append([int(line[1].split("/")[1])-1,int(line[2].split("/")[1])-1,int(line[3].split("/")[1])-1]) 
            faceID2normalIDset.append([int(line[1].split("/")[2])-1,int(line[2].split("/")[2])-1,int(line[3].split("/")[2])-1]) 
        i+=1

    return vtxs , nmls , uvs , faceID2vtxIDset , faceID2uvIDset , faceID2normalIDset , vtx_num , face_num


def get_vtxID2uvIDs(faceID2vtxIDset , faceID2uvIDset ):
    '''
    make vtxID2uvIDs list. 
    '''
    vtxID2uvIDs = {}
    face_num = len(faceID2uvIDset)
    ### make dict : vtxID → uvID
    for faceID in range(face_num):
        vtxIDs = faceID2vtxIDset[faceID]
        uvIDs  = faceID2uvIDset[faceID]
        for j in range(3):
            vtxID = vtxIDs[j]
            uvID  = uvIDs[j]
            if (vtxID in vtxID2uvIDs) == False:
                vtxID2uvIDs[vtxID] = []
            vtxID2uvIDs[vtxID].append(uvID)

    ### delete deplicate
    for vtxID in vtxID2uvIDs:
        vtxID2uvIDs[vtxID] = list(set(vtxID2uvIDs[vtxID]))
    return vtxID2uvIDs

def get_vtxID2uvCoordinate(uvs , faceID2vtxIDset , faceID2uvIDset , vtx_num , uvs_np_format = np.float32 , regular_array_size = None):
    '''
    make vtxID2uvCoordinate list . 
    '''
    vtxID2uvIDs = {}
    vtxID2uvCoordinate = []
    
    face_num = len(faceID2uvIDset)
    ### make dict : vtxID → uvID
    for faceID in range(face_num):
        vtxIDs = faceID2vtxIDset[faceID]
        uvIDs  = faceID2uvIDset[faceID]
        for j in range(3):
            vtxID = vtxIDs[j]
            uvID  = uvIDs[j]
            if (vtxID in vtxID2uvIDs) == False:
                vtxID2uvIDs[vtxID] = []
            vtxID2uvIDs[vtxID].append(uvID)

    ### delete deplicate
    for vtxID in vtxID2uvIDs:
        vtxID2uvIDs[vtxID] = list(set(vtxID2uvIDs[vtxID]))

    ### make list : vtxID → uvID
    if regular_array_size == None:
        for i in range(vtx_num):
            vtxID2uvCoordinate_tmp = []
            for j , uvID in enumerate(vtxID2uvIDs[i]):
                uvs_tmp = np.array(uvs[uvID] , dtype = uvs_np_format)
                vtxID2uvCoordinate_tmp.append(uvs_tmp)
            vtxID2uvCoordinate.append(vtxID2uvCoordinate_tmp)
        return vtxID2uvCoordinate
    else:
        for i in range(vtx_num):
            vtxID2uvCoordinate_tmp = []
            for j , uvID in enumerate(vtxID2uvIDs[i]):
                if j == regular_array_size:
                    break
                uvs_tmp = np.array(uvs[uvID] , dtype = uvs_np_format)
                vtxID2uvCoordinate_tmp.append(uvs_tmp)
            for k in range(j+1,regular_array_size):
                vtxID2uvCoordinate_tmp.append(np.zeros(2, dtype = uvs_np_format))
            vtxID2uvCoordinate.append(vtxID2uvCoordinate_tmp)
        return np.array(vtxID2uvCoordinate)

def save_obj(path ,vtxs , nmls = None , uvs = None , faceID2vtxIDset=None , faceID2uvIDset=None , faceID2normalIDset=None , vtx_num=None , face_num=None):  
    f = open(path,"w")
    header1 = [ 
    "# Blender v2.93.0 OBJ File: 'unskin2.blend'",
    "# www.blender.org",
    "mtllib uvskin.mtl",
    "o skin"
    ]

    vtx_lines = []
    for i in range(vtx_num):
        vtx_lines.append("v " + str(vtxs[i][0]) + " "  + str(vtxs[i][1]) + " "  + str(vtxs[i][2]))

    if uvs != None:
        txr_lines = []
        for t in uvs:
            txr_lines.append("vt " + str(t[0]) + " "  + str(t[1]))
    else:
        txr_lines = []

    if nmls != None:
        nml_lines = []
        for n in nmls:
            nml_lines.append("vn " + str(n[0]) + " "  + str(n[1])+ " "  + str(n[2]))
    else:
        nml_lines = []


    header2 = [
    "usemtl None",
    "s off"
    ]

    if faceID2vtxIDset != None:
        face_lines = []
        for i,fc in enumerate(faceID2vtxIDset):
            if faceID2uvIDset != None and faceID2normalIDset != None: 
                face_lines.append("f " + str(fc[0]+1) + "/" + str(faceID2uvIDset[i][0]+1) + "/" + str(faceID2normalIDset[i][0]+1) + " " + str(fc[1]+1) + "/" + str(faceID2uvIDset[i][1]+1) + "/" + str(faceID2normalIDset[i][1]+1) + " " + str(fc[2]+1) + "/" + str(faceID2uvIDset[i][2]+1) + "/" + str(faceID2normalIDset[i][2]+1))
            elif faceID2uvIDset != None :
                face_lines.append("f " + str(fc[0]+1) + "/" + str(faceID2uvIDset[i][0]+1) + "/" + " " + str(fc[1]+1) + "/" + str(faceID2uvIDset[i][1]+1) + "/" + " " + str(fc[2]+1) + "/" + str(faceID2uvIDset[i][2]+1) + "/")
            elif faceID2normalIDset != None: 
                face_lines.append("f " + str(fc[0]+1) + "/" + "/" + str(faceID2normalIDset[i][0]+1) + " " + str(fc[1]+1) + "/" + "/" + str(faceID2normalIDset[i][1]+1) + " " + str(fc[2]+1) + "/" + "/" + str(faceID2normalIDset[i][2]+1))
            else:
                face_lines.append("f " + str(fc[0]+1) + "/" + "/"  + " " + str(fc[1]+1) + "/" + "/" + " " + str(fc[2]+1) + "/" + "/" )
    else:
        face_lines = []

    eof = ['']
    lines = header1 + vtx_lines + txr_lines + nml_lines + header2 + face_lines + eof
    f.write("\n".join(lines))

if __name__ == "__main__":
    vtxs , nmls , uvs , faceID2vtxIDset , faceID2uvIDset , faceID2normalIDset , vtx_num , face_num = load_obj("/mnt/z/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.015-0.05/Basic_data/uv_coordinate0422/Shellb_expandUV.obj")
    vtxID2uvIDs = get_vtxID2uvIDs(faceID2vtxIDset , faceID2uvIDset)
    print(vtxID2uvIDs)