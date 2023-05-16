python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/03375/blazerlong_volleyball_trial1/data" ^
                                                         --dataset_type "cape" --gender "male" --Training_type "Interpolation" ^
                                                         --useID "True" --use_beta "True" ^
                                                         --IDstart 9 --IDend 163 --IDstep 10^
                                                         --norm_a 3 --Debug "False"

python ./main.py train=Cape_Snarf_blazerlong_volleyball_itp train.mode=train model.prediction_texture_mask=False

python ./main.py train=Cape_Snarf_blazerlong_volleyball_itp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False

python Evaluate/chamfer_distance_kaolin.py   --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/03375/blazerlong_volleyball_trial1/data/resampled_meshes" ^
                                             --gt_header "" --gt_ext ".obj"^
                                             --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_snarf/1"^
                                             --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"




python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/03375/blazerlong_climb_trial1/data" ^
                                                        --dataset_type "cape" --gender "male" --Training_type "Interpolation" ^
                                                        --useID "True" --use_beta "True" ^
                                                        --IDstart 9 --IDend 125 --IDstep 10^
                                                        --norm_a 3 --Debug "False"

python ./main.py train=Cape_Snarf_blazerlong_climb_itp train.mode=train model.prediction_texture_mask=False

python ./main.py train=Cape_Snarf_blazerlong_climb_itp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False

python Evaluate/chamfer_distance_kaolin.py   --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/03375/blazerlong_climb_trial1/data/resampled_meshes" ^
                                             --gt_header "" --gt_ext ".obj"^
                                             --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_snarf/2"^
                                             --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"



@REM 
@REM python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/00096/shortlong_squats/data" ^
@REM                                                        --dataset_type "cape" --gender "male" --Training_type "Interpolation" ^
@REM                                                        --useID "True" --use_beta "True" ^
@REM                                                        --IDstart 9 --IDend 147 --IDstep 10^
@REM                                                        --norm_a 3 --Debug "False"
@REM 
@REM 
@REM python ./main.py train=Cape_Snarf_shortlong_squats_itp train.mode=train model.prediction_texture_mask=False
@REM 
@REM python ./main.py train=Cape_Snarf_shortlong_squats_itp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False
@REM 
@REM python Evaluate/chamfer_distance_kaolin.py   --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/00096/shortlong_squats/data/resampled_meshes" ^
@REM                                              --gt_header "" --gt_ext ".obj"^
@REM                                              --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_snarf/3"^
@REM                                              --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"
@REM 
@REM 
@REM 
@REM 
@REM python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/00215/poloshort_punching/data" ^
@REM                                                        --dataset_type "cape" --gender "male" --Training_type "Interpolation" ^
@REM                                                        --useID "True" --use_beta "True" ^
@REM                                                        --IDstart 9 --IDend 114 --IDstep 10^
@REM                                                        --norm_a 3 --Debug "False"
@REM 
@REM python ./main.py train=Cape_Snarf_poloshort_punching_itp train.mode=train model.prediction_texture_mask=False
@REM 
@REM python ./main.py train=Cape_Snarf_poloshort_punching_itp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False
@REM 
@REM python Evaluate/chamfer_distance_kaolin.py   --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_SNARF/AITS/00215/poloshort_punching/data/resampled_meshes" ^
@REM                                              --gt_header "" --gt_ext ".obj"^
@REM                                              --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_snarf/4"^
@REM                                              --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"
@REM 