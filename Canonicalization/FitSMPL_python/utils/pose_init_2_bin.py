import numpy as np
import argparse 
import os

def pose_txt_2_bin(pose_path):
    path_without_ext = pose_path.split(".")[0]
    with open(pose_path) as f:
        lines = f.readlines()

    words = lines[0].split(", ")
    data = np.array(words, dtype = np.float32)

    output_name = path_without_ext + ".bin"
    print("saved to : " , output_name)
    data.tofile(output_name)

def trans_txt_2_npy(trans_path):
    path_without_ext = trans_path.split(".")[0]

    with open(trans_path) as f:
        lines = f.readlines()
    words = lines[0].split(" ")
    data = np.array(words, dtype = np.float32)

    """
    tmp = [float(i) for i in words]
    output_name = path_without_ext + ".npy"
    print("saved to : " , output_name)
    np.save(output_name , tmp)
    """

    output_name = path_without_ext + ".bin"
    print("saved to : " , output_name)
    data.tofile(output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit SMPL body to voxels predictions.')
    parser.add_argument('--pose_path', type=str, default=r"None",
                        help='Path to init SMPL init poseparam.')
    parser.add_argument('--trans_path', type=str, default=r"None",
                        help='Path to init SMPL init trans.')
    args = parser.parse_args()
    pose_path = args.pose_path
    trans_path = args.trans_path

    if pose_path != "None":
        pose_txt_2_bin(pose_path)
    if trans_path != "None":
        trans_txt_2_npy(trans_path)