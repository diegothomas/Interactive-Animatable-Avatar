from meshlab_utils import resampling_mesh
import tqdm
import os
from natsort import natsorted
import glob
import argparse

abs = 0.006

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str, default="path to dataname" , help='')
args = parser.parse_args()
path = args.path

dataset_path = os.path.join(path , "data/src_meshes/*")
save_dir     = os.path.join(path , "data/resampled_meshes")

print(dataset_path)
paths = natsorted(glob.glob(dataset_path))

for path in tqdm.tqdm(paths):
    basename = str(int(os.path.splitext(os.path.basename(path))[0])).zfill(4)
    save_path = os.path.join(save_dir , basename + ".obj")
    resampling_mesh(path , save_path , abs , is_percent = False)