import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

def plot(args , save_path , bins1 , counts1 , bins2 , counts2 , inverse = False):
            fig, ax = plt.subplots()

            c1,c2 = "blue","red"     # 各プロットの色
            l1,l2 = args.name_tag1, args.name_tag2  # 各ラベル

            ax.set_xlabel('s2m (m)')  # x軸ラベル
            ax.set_ylabel('number of points')  # y軸ラベル

            #ax.set_title(r'')  #グラフタイトル

            ax.grid()

            ax.plot(bins1, counts1, color=c1, label=l1)
            ax.plot(bins2, counts2, color=c2, label=l2)

            ax.legend(loc=1)    # 凡例]

            if inverse:
                ax.invert_xaxis()
            
            plt.savefig(save_path , dpi=300)
            plt.close()
def main():
    parser = argparse.ArgumentParser(description='chamfer distance')
    parser.add_argument(
        '--dir_path1',
        type=str,
        help='')
    
    parser.add_argument(
        '--name_tag1',
        type=str,
        help='')

    parser.add_argument(
        '--dir_path2',
        type=str,
        help='')

    parser.add_argument(
        '--name_tag2',
        type=str,
        help='')

    args = parser.parse_args()

    hist_types = ["gt2pred_hist" , "pred2gt_hist"]
    for hist_type in hist_types:
        npz_paths1   = glob.glob(os.path.join(args.dir_path1 , hist_type, "*.npz"))

        for npz_path1 in npz_paths1:
            id = os.path.basename(npz_path1).split(".")[0]
            npz_path2 = os.path.join(args.dir_path2 , hist_type, id + ".npz")
            print(npz_path1)
            print(npz_path2)
            
            if os.path.isfile(npz_path2) == False:
                print("Skip !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("npz_path1 != npz_path2")
                print("npz_path1 : " , npz_path1)
                print("npz_path2 : " , npz_path2)
                continue

            hist_npz1 = np.load(npz_path1)
            hist_npz2 = np.load(npz_path2)

            counts1  = hist_npz1["counts"]
            bins1    = hist_npz1["bins"]
            counts2  = hist_npz2["counts"]
            bins2   = hist_npz2["bins"]

            #累積分布関数化
            new_counts1 = []
            new_counts2 = []
            sum_points1 = 0
            sum_points2 = 0
            for c1 ,c2 in zip(counts1 , counts2):
                sum_points1 += c1
                new_counts1.append(sum_points1)
                sum_points2 += c2
                new_counts2.append(sum_points2)

            if np.any(bins1 != bins2):
                print("bins1 : " . bins1)
                print("bins2 : " . bins2)
                raise AssertionError("bins1 != bins2")
            #plt.figure() 
            #plt.plot(new_bins, counts)
            #plt.show()
            
            save_name = os.path.basename(npz_path1).split(".")[0].split("_")[-1]
            save_path = os.path.join(args.dir_path1 , hist_type ,"hist_" + save_name + ".png")

            plot(args , save_path , bins1 , new_counts1 , bins2 , new_counts2)

            #逆！累積分布関数化
            counts1_inverse = counts1[::-1]
            counts2_inverse = counts2[::-1]
            bins1_inverse   = bins1[::-1]
            bins2_inverse   = bins2[::-1]

            new_counts1 = []
            new_counts2 = []
            sum_points1 = 0
            sum_points2 = 0

            total_points_num = counts1_inverse.sum()
            for c1 ,c2 in zip(counts1_inverse , counts2_inverse):
                sum_points1 += c1 
                new_counts1.append((sum_points1 / total_points_num)*100)
                sum_points2 += c2
                new_counts2.append((sum_points2 / total_points_num)*100)

            if np.any(bins1_inverse != bins2_inverse):
                print("bins1 : " . bins1_inverse)
                print("bins2 : " . bins2_inverse)
                raise AssertionError("bins1 != bins2")
            #plt.figure() 
            #plt.plot(new_bins, counts)
            #plt.show()
            
            save_path = os.path.join(args.dir_path1 , hist_type , "hist_inverse_" + save_name + ".png")

            plot(args , save_path , bins1_inverse , new_counts1 , bins2_inverse , new_counts2 , inverse = True)

            save_path = os.path.join(args.dir_path1 , hist_type , "hist_inverse_by0.1_" + save_name + ".png")
            plot(args , save_path , bins1_inverse[:-10] , new_counts1[:-10] , bins2_inverse[:-10] , new_counts2[:-10] , inverse = True)

            save_path = os.path.join(args.dir_path1 , hist_type , "hist_inverse_by0.2_" + save_name + ".png")
            plot(args , save_path , bins1_inverse[:-20] , new_counts1[:-20] , bins2_inverse[:-20] , new_counts2[:-20] , inverse = True)
        
if __name__ == "__main__":
    main()