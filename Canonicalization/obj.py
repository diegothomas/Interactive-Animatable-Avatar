import sys, os
import skimage.color
import matplotlib.cm as cm

import numpy as np
# import file_io as fio


def load_obj(filename):
    if not os.path.isfile(filename):
        return None
    with open(filename, 'r') as fin:
        vertices = []
        faces = []
        Lines = fin.readlines()
        for line in Lines:
            line_s = line.split(' ')
            if line_s[0] == 'v':
                vertices += [[float(line_s[1]), float(line_s[2]), float(line_s[3])]]
            if line_s[0] == 'f':
                faces += [[int(line_s[1])-1, int(line_s[2])-1, int(line_s[3])-1]]
    return np.array(vertices), np.array(faces)
    
def load_obj_V(filename):
    if not os.path.isfile(filename):
        return None
    with open(filename, 'r') as fin:
        vertices = []
        Lines = fin.readlines()
        for line in Lines:
            line_s = line.split(' ')
            if line_s[0] == 'v':
                vertices += [[float(line_s[1]), float(line_s[2]), float(line_s[3])]]
    return np.array(vertices)



if __name__ == "__main__":
    pass
    # if len(sys.argv) < 3:
    #     print 'usage : python {0} [(input)obj file] [(output)ply file]'
    #     exit(0)

    # X, _, f = fio.load_obj_file(sys.argv[1])
    # save_ply(sys.argv[2], X, f=f)
