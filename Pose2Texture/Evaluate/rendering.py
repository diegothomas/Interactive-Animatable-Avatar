import json
import re
import os
import sys
import subprocess

from render.img2video import img2video

def ReadJSONC(filepath:str):
    with open(filepath, 'r', encoding='utf-8') as f:      
        text = f.read()                                   
    re_text = re.sub(r'/\*[\s\S]*?\*/|//.*', '', text)    
    json_obj = json.loads(re_text)                        
    return json_obj       

if __name__ == "__main__":
    cfg = ReadJSONC("render/render_config.jsonc")
    data_setting     = cfg["data_setting"]
    data_name_list   = cfg["data_name"]
    env_setting      = cfg["environment"]

    for data_name in data_name_list:
        path             = data_setting[data_name]["path"]
        format           = data_setting[data_name]["format"]
        rgb_flg          = data_setting[data_name]["rgb_flg"]
        environment      = data_setting[data_name]["environment"]
        fps              = data_setting[data_name]["fps"]

        camera_location  = env_setting[environment]["camera_location"]
        lamp_location    = env_setting[environment]["lamp_location"]

        mesh_folder_path = os.path.join(path,format)
        save_root_path   = os.path.join(path,"rendering")
        
        camera_location  = list(map(str , camera_location))
        lamp_location    = list(map(str , lamp_location))

        #subprocess.run(["C:/Program Files/Blender Foundation/Blender 2.93/blender.exe" , "--python" ,  "render/rendering_blender.py" , \
        subprocess.run(["D:/Program Files/Blender Foundation/Blender 3.3/blender.exe" , "--python" ,  "render/rendering_blender.py" , \
                        "--" , \
                        "--mesh_folder_path" , mesh_folder_path , \
                        "--save_root_path"   , save_root_path   , \
                        "--rgb_flg"          , rgb_flg          , \
                        "--camera_location"  , camera_location[0] , camera_location[1] , camera_location[2], \
                        "--lamp_location"    , lamp_location[0]   , lamp_location[1]   ,lamp_location[2]  , \
                        ] , shell=True )
        
        img2video(save_root_path , fps , save_name=data_name)