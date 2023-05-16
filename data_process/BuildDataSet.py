import os
import subprocess
import glob

TEMPLATE_PATH = 'D:/Data/Human/Template/'
OPENPOSE_PATH = 'D:/Project/Human/openpose'
FFMPEG_PATH = 'D:/Project\Human/ffmpeg-4.3.2-2021-02-27-essentials_build/bin/'
#'C:\\Users\\thomas\\Documents\\Projects\\Human-AI\\ffmpeg-4.4-essentials_build\\bin\\'

# <=== read the list of data
data_list = []
if not os.path.isfile('data_list.txt'):
    print("no data to process")
    exit()
with open('data_list.txt', 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            print(line)
            line_s = line.split(' ')
            if line_s[0] == '#':
                continue
            data_list.append(line_s[0].split('\n')[0])

print(data_list)

CURR_PATH = os.getcwd()

for DATA_PATH in data_list:
    print("Preparing: " + DATA_PATH)
    words = DATA_PATH.split('/')
    data_name = words[len(words)-2]
    print(data_name)
    
    # <=== Prepare folder
    os.chdir(DATA_PATH)
    subprocess.call(['mkdir', 'cropped_img'], shell=True)
    subprocess.call(['mkdir', 'nml_images'], shell=True)
    subprocess.call(['mkdir', 'meshes_centered'], shell=True)
    subprocess.call(['mkdir', 'TSDF'], shell=True)
    subprocess.call(['mkdir', 'smplparams'], shell=True)
    subprocess.call(['mkdir', 'videos'], shell=True)
    subprocess.call(['mkdir', 'video_blender'], shell=True)
    subprocess.call(['mkdir', 'openpose'], shell=True)
    subprocess.call(['mkdir', 'openpose_blender'], shell=True)
    
    # Get name and format of images
    names_img = sorted(glob.glob(DATA_PATH + "images/Camera*.jpg"))
    nb_img = int(len(names_img)/8)
    if nb_img > 0:
        prefix_name = "Camera"
        suffix_name = ".jpg"
    else:
        names_img = sorted(glob.glob(DATA_PATH + "images/Image*.png"))
        #print(DATA_PATH + "images/Image*.png")
        nb_img = int(len(names_img)/8)
        if nb_img == 0:
            print("UNKNOWN data format --> image file")
            continue
        prefix_name = "Image"
        suffix_name = ".png"
        
    print(prefix_name, suffix_name, nb_img)
    
    # <==== 1. Create videos from the images
    # Need to download ffmpeg binaries: https://www.gyan.dev/ffmpeg/builds/
    '''for i in range(1,9):
        command = [FFMPEG_PATH+'ffmpeg', '-i', DATA_PATH + 'images/' + prefix_name+str(i)+'_%04d' + suffix_name, DATA_PATH + 'videos/' + prefix_name+str(i)+'.avi']
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command, shell=True)

    # <==== 2. Run openpose demo on each video
    # Need to download openpose demo
    # Change the current working directory
    os.chdir(OPENPOSE_PATH)
    for i in range(1,9):
        command = ['bin\\OpenPoseDemo.exe', '--video', DATA_PATH + 'videos/' + prefix_name + str(i) +'.avi', '--write_json', DATA_PATH + 'openpose']
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)
                    
    
    # Change the current working directory
    os.chdir(CURR_PATH)
    # <==== 3. Crop images
    command = ['python', 'data_process/cropimg.py', '--imgdir', DATA_PATH + 'images', '--skeletondir', DATA_PATH + 'openpose', '--silhouettedir', DATA_PATH + 'silhouettes', '--savedir', DATA_PATH + 'cropped_img', '--prefix', prefix_name]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command, shell=True)'''
                        

    # Change the current working directory
    os.chdir(CURR_PATH)
    os.chdir('network')
    command = ['python', 'ImageToNML.py', '--imgdir', DATA_PATH + 'cropped_img', '--savedir', DATA_PATH + 'nml_images', '--pref', prefix_name]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command, shell=True)
        
    '''os.chdir(CURR_PATH)
    
    # <=== fit smpl and center mesh
    # <==== 1. Create videos from the images
    # Need to download ffmpeg binaries: https://www.gyan.dev/ffmpeg/builds/
        
    for i in range(1,13):
        command = [FFMPEG_PATH+'ffmpeg', '-i', DATA_PATH + 'images_blender\\Images'+str(i)+'_%04d.png', DATA_PATH + 'video_blender\\Images'+str(i)+'.avi']
        print('Videos')
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command, shell=True)
        
    # <==== 2. Run openpose demo on each video
    # Need to download openpose demo
    # Change the current working directory \\mesh_'+str(i).zfill(4)
    os.chdir(OPENPOSE_PATH)
    for i in range(1,13):
        currpath = DATA_PATH + 'openpose_blender'
        command = ['bin\\OpenPoseDemo.exe', '--video', DATA_PATH + 'video_blender\\Images'+str(i)+'.avi', '--write_json', currpath]
        print('OpenPose')
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)
            
    # Change the current working directory
    os.chdir(CURR_PATH)
    # <==== 3. Compute 3D skeleton
    command = ['python', 'FitSMPL_python/process_openpose.py', '--path_in', DATA_PATH, '--nb_views', '6', '--nb_scans', str(nb_img)]
    print('Get 3D skeleton')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    
    # <==== 4. Center the meshes
    command = ['build_gpu/Release/deepanim', '-mode', 'ObjToPlyCentered', '-inputpath', DATA_PATH, '-size', str(nb_img)]
    print('Center meshes')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
        
    # <==== 4. Fit SMPL model
    os.chdir('FitSMPL_python')
    command = ['python', 'FitSMPL_3DSkeleton_Scan.py', '--path', DATA_PATH, '--size', str(nb_img), '--verbose', 'True']
    print('FitSMPL')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    
    # <==== 4. Generate TSDF
    os.chdir(CURR_PATH)
    command = ['build_gpu/Release/deepanim', '-mode', 'CreateLevelSet', '-inputpath', DATA_PATH, '-outputpath', DATA_PATH + 'TSDF', '-templatepath', TEMPLATE_PATH, '-size', str(nb_img)]
    print('TSDF')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)'''
 
''' os.chdir(DATA_PATH)
    subprocess.call(['rm', '-r', 'videos'])
    subprocess.call(['rm', '-r', 'openpose'])
    subprocess.call(['rm', '-r', 'openpose_blender'])'''

