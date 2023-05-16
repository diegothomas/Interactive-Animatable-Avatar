import os
import subprocess
import glob

TEMPLATE_PATH = 'D:/Data/Human/Template/'
OPENPOSE_PATH = 'D:/Project/Human/openpose'
FFMPEG_PATH = 'D:/Project\Human/ffmpeg-4.3.2-2021-02-27-essentials_build/bin/'
#'C:\\Users\\thomas\\Documents\\Projects\\Human-AI\\ffmpeg-4.4-essentials_build\\bin\\'

# <=== read the list of data
data_list = []
if not os.path.isfile('data_list_cape.txt'):
    print("no data to process")
    exit()
with open('data_list_cape.txt', 'r') as fin:
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
    subprocess.call(['mkdir', 'video_blender'], shell=True)
    subprocess.call(['mkdir', 'openpose_blender'], shell=True)
    subprocess.call(['mkdir', 'tmp'], shell=True)
    
    # Get name and format of images
    names_img = sorted(glob.glob(DATA_PATH + "Images_blender/Camera*.jpg"))
    nb_img = int(len(names_img)/12)
    if nb_img > 0:
        prefix_name = "Camera"
        suffix_name = ".jpg"
    else:
        names_img = sorted(glob.glob(DATA_PATH + "Images_blender/Image*.png"))
        #print(DATA_PATH + "images/Image*.png")
        nb_img = int(len(names_img)/12)
        if nb_img == 0:
            print("UNKNOWN data format --> image file")
            continue
        prefix_name = "Image"
        suffix_name = ".png"
        
    print(prefix_name, suffix_name, nb_img)
    
    # <==== 1. Create videos from the images
    # Need to download ffmpeg binaries: https://www.gyan.dev/ffmpeg/builds/
    for i in range(1,13):
        command = [FFMPEG_PATH+'ffmpeg', '-i', DATA_PATH + 'Images_blender/' + prefix_name+str(i)+'_%04d' + suffix_name, DATA_PATH + 'video_blender/' + prefix_name+str(i)+'.avi']
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command, shell=True)

    # <==== 2. Run openpose demo on each video
    # Need to download openpose demo
    # Change the current working directory
    os.chdir(OPENPOSE_PATH)
    for i in range(1,13):
        command = ['bin\\OpenPoseDemo.exe', '--video', DATA_PATH + 'video_blender/' + prefix_name + str(i) +'.avi', '--write_json', DATA_PATH + 'openpose_blender']
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)
                    
    
    # Change the current working directory
    os.chdir(CURR_PATH)
    # <==== 3. Crop images
    command = ['python', 'data_process/cropimg_cape.py', '--imgdir', DATA_PATH + 'Images_blender', '--skeletondir', DATA_PATH + 'openpose_blender', '--silhouettedir', DATA_PATH + 'Images_blender', '--savedir', DATA_PATH + 'cropped_img', '--prefix', prefix_name]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command, shell=True)
                            
    # <=== fit smpl and center mesh
                
    # Change the current working directory
    os.chdir(CURR_PATH)
    # <==== 3. Compute 3D skeleton
    command = ['python', 'FitSMPL_python/process_openpose.py', '--path_in', DATA_PATH, '--nb_views', '12', '--nb_scans', str(nb_img)]
    print('Get 3D skeleton')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    
    # <==== 4. Center the meshes
    os.chdir(CURR_PATH)
    command = ['build_gpu/Release/deepanim', '-mode', 'ObjToPlyCentered', '-inputpath', DATA_PATH, '-size', str(nb_img)]
    print('Center meshes')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
        
    # <==== 4. Fit SMPL model
    os.chdir(CURR_PATH)
    os.chdir('FitSMPL_python')
    command = ['python', 'FitSMPL_3DSkeleton_Cape.py', '--path', DATA_PATH, '--size', str(nb_img), '--verbose', 'True']
    print('FitSMPL')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    
    # <==== 4. Generate TSDF
    '''os.chdir(CURR_PATH)
    command = ['build_gpu/Release/deepanim', '-mode', 'CreateLevelSet', '-inputpath', DATA_PATH, '-outputpath', DATA_PATH + 'TSDF', '-templatepath', TEMPLATE_PATH, '-size', str(nb_img)]
    print('TSDF')
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)'''
 
''' os.chdir(DATA_PATH)
    subprocess.call(['rm', '-r', 'videos'])
    subprocess.call(['rm', '-r', 'openpose'])
    subprocess.call(['rm', '-r', 'openpose_blender'])'''

