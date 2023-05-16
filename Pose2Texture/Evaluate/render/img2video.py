import sys
import cv2
import glob
from natsort import natsorted
import os

if __name__ == "__main__":
    from gooey import GooeyParser

def img2video(folder_path , fps , save_name = None):
    #folder_path = r"D:\Project\Human\Avatar-In-The-Shell\Result\result\20211115_SCANimate_data_collection\Processing_lists\8_Muscle_range_of_motion\test\release_muscle_range_of_motion_test_Taunt\test_Taunt\seqs\rendering"
    #img_folder_path = natsorted(glob.glob(os.path.join(folder_path ,  r"*.png")))
    img_folder_path = glob.glob(os.path.join(folder_path ,  r"*.png"))

    if save_name != None:
        save_path = os.path.join(folder_path , save_name + ".mp4")
    else:
        save_path = os.path.join(folder_path , r"rendering.mp4")
    print("img_folder_path : " , img_folder_path)
    print("save_path       : " , save_path)

    # encoder(for mp4)
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 's')
    #fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '5')

    # output file name, encoder, fps, size(fit to image size)
    image_size = cv2.imread(img_folder_path[0]).shape[:2]
    video = cv2.VideoWriter(save_path,fourcc, fps, image_size)

    if not video.isOpened():
        print("can't be opened")
        sys.exit()

    for i,img_path in enumerate(img_folder_path):
        print(img_path)
        img = cv2.imread(img_path)
        #frame_name = i + 2
        #frame_name = int(os.path.basename(img_path).split(".")[0].split("_")[-1])
        #frame_name = int(os.path.basename(img_path).split(".")[0].split("_")[-1])
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        #if (frame_id<70):
        #    continue
        #if (frame_name%10!=5):
        #   continue
        print(frame_name , ":" ,frame_name)

        #write frame_id
        cv2.putText(img,
                #text= "dataset : " + dataname + "  " + str(fps) + "fps  id :" + str(frame_id),
                text= frame_name,
                #text= "dataset : " + "avgtexture_alpha3_rgb" + "  " + str(fps) + "fps  id :" + str(frame_id),
                org=(30,30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=2,
                #thickness=1,
                lineType=cv2.LINE_4)


        # can't read image, escape
        if img is None:
            print("can't read")
            break
        # add
        video.write(img)

    video.release()
    print('written')

if __name__ == "__main__":
    parser = GooeyParser(description='Regress texture from pose')

    parser.add_argument(
            '--folder_path',
            default= None,
            type=str,
            help='')

    parser.add_argument(
            '--fps',
            default= 18,
            type=float,
            help='')
    
    parser.add_argument(
            '--save_name',
            default= None,
            type=str,
            help='')

    args = parser.parse_args()

    folder_path = args.folder_path
    fps = args.fps
    save_name = args.save_name
    img2video(folder_path , fps , save_name)