import bpy
import json
import re
import argparse    

#from bpy import context
import glob

import sys
import os

from pathlib import Path
def parentpath(path='.', f=0):
    return Path(path).resolve().parents[f]
import os

#sys.path.append(".")
#import render
#from render.blender_render import render


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

def render(mesh_paths,save_folder_path,rgb_flg,  camera_location , lamp_location , data = "whole"):
    camera = bpy.data.objects["Camera"]
    #centerd camera

    if data == "whole":
        scene = bpy.data.scenes['Scene']
        render_setting = scene.render
        render_setting.resolution_x = 1920
        render_setting.resolution_y = 1920

        camera.location =  camera_location    
        camera.rotation_euler = (0, 0.0, 0)

        lamp = bpy.data.objects["Light"]
        lamp.location   = lamp_location   

    elif data == "face":
        camera.location = (0, 0.45, 2)
        camera.rotation_euler = (0, 0.0, 0)

        lamp = bpy.data.objects["Light"]
        #lamp.location = (0.0, -1, 5)   #old
        lamp.location = (0.0, 0.0, 9.0)   #old

    targetob = bpy.data.objects.get("Cube")
    if targetob != None:
        bpy.data.objects.remove(targetob)
        #bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
        
    if os.path.exists(save_folder_path) == False:
        os.mkdir(save_folder_path)

    frame_num = 0
    old_vtx_num = -1
    for k ,mesh_path in enumerate(mesh_paths):
        print("input_path:",mesh_path)
        #開始のフレームを設定する
        bpy.context.scene.frame_set(frame_num)
        
        data_type = os.path.basename(mesh_path).split(".")[-1]
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)

        if data_type == "ply":
            bpy.ops.import_mesh.ply(filepath=mesh_path)
        elif data_type == "obj":
            bpy.ops.wm.obj_import(filepath=mesh_path)
        bpy.context.object.rotation_euler[0] = 0

        """
        if data_type == "ply":
            vtx , _ , rgb , face , vtx_num , face_num = load_ply(mesh_path)
        elif data_type == "obj":
            vtx , _ , _ , face , _ , _ , vtx_num , face_num = load_obj(mesh_path)

        if k==0 or old_vtx_num != vtx_num:
            if old_vtx_num != vtx_num:
                targetob = bpy.data.objects.get("Cube")
                if targetob != None:
                    bpy.data.objects.remove(targetob)
                    #bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
            old_vtx_num = vtx_num
            msh = bpy.data.meshes.new("cubemesh") #create mesh data
            msh.from_pydata(vtx, [], face) # 頂点座標と各面の頂点の情報でメッシュを作成
            obj = bpy.data.objects.new("Cube", msh) # メッシュデータでオブジェクトを作成

            if rgb_flg == True:
                # make vertex colour data
                msh.vertex_colors.new(name='col')

                print("len:",len(msh.vertex_colors['col'].data))
                print("face_len:",face_num)
                for idx, vc in enumerate(msh.vertex_colors['col'].data):
                    vtx_cnt = idx%3
                    #vc.color =  [1.0,0.0,0.0,1.0]    
                    vc.color =  [rgb[face[int(idx/3)][vtx_cnt]][0]/255,rgb[face[int(idx/3)][vtx_cnt]][1]/255,rgb[face[int(idx/3)][vtx_cnt]][2]/255,1.0]

                # make its first material slot
                mat = bpy.data.materials.new(name = "Skin")
                mat.use_nodes = True #Make so it has a node tree

                #Get the shader
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                base_color = bsdf.inputs['Base Color'] #Or principled.inputs[0]

                #Add the vertex color node
                vtxclr = mat.node_tree.nodes.new('ShaderNodeVertexColor')
                #Assign its layer
                vtxclr.layer_name = "col"
                #Link the vertex color to the shader
                mat.node_tree.links.new( vtxclr.outputs[0], base_color )
                obj.data.materials.append(mat)

            bpy.context.scene.collection.objects.link(obj) # シーンにオブジェクトを配置
        else:
            for i in range(vtx_num):                         #update mesh when same topology
                obj.data.vertices[i].co.x = vtx[i][0]
                obj.data.vertices[i].co.y = vtx[i][1]
                obj.data.vertices[i].co.z = vtx[i][2]
            if rgb_flg == True:
                for idx, vc in enumerate(obj.data.vertex_colors['col'].data):
                    vtx_cnt = idx%3
                    #vc.color =  [1.0,0.0,0.0,1.0]    
                    vc.color =  [rgb[face[int(idx/3)][vtx_cnt]][0]/255,rgb[face[int(idx/3)][vtx_cnt]][1]/255,rgb[face[int(idx/3)][vtx_cnt]][2]/255,1.0]
        """

        #msh.vertices[0].keyframe_insert('co',index = 2,frame = frame_num)
        #キーフレームの挿入を行う
        """
        obj.keyframe_insert(data_path = "location",index = -1)
        frame_num += 5
        """
        
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)   #sleep
        #time.sleep(5)

        #rendering with image
        bpy.ops.render.render()
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        save_path = os.path.join(save_folder_path , name + '.png')
        bpy.data.images['Render Result'].save_render(filepath = save_path)
        print("save_path:",save_path)

def ReadJSONC(filepath:str):
    with open(filepath, 'r', encoding='utf-8') as f:      
        text = f.read()                                   
    re_text = re.sub(r'/\*[\s\S]*?\*/|//.*', '', text)    
    json_obj = json.loads(re_text)                        
    return json_obj       

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def main():
    parser = argparse.ArgumentParser(description='')   

    parser.add_argument('--mesh_folder_path',  help='')    
    parser.add_argument('--save_root_path'  ,  help='')
    parser.add_argument('--rgb_flg'         ,  help='')
    parser.add_argument('--camera_location' , nargs="*" ,type=float)
    parser.add_argument('--lamp_location'   , nargs="*" ,type=float)

    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])    

    #mesh_paths = sorted(glob.glob(args.mesh_folder_path) , key=numericalSort)
    mesh_paths = glob.glob(args.mesh_folder_path)
    if args.rgb_flg == "True":
        rgb_flg = True
    else:
        rgb_flg = False
    
    if os.path.isdir(args.save_root_path) == False:
        os.mkdir(args.save_root_path)

    render(mesh_paths,args.save_root_path,rgb_flg,args.camera_location,args.lamp_location)
    bpy.ops.wm.quit_blender()
if __name__ == "__main__":
    main()