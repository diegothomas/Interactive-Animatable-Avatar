python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_volleyball_trial1/data" ^
                                                        --dataset_type "pop" --Training_type "Extrapolation" ^
                                                        --useID "True" --use_beta "True" ^
                                                        --IDstart 0 --IDend 24 --IDstep 1 ^
                                                        --norm_a 3 --Debug "False"

python ./main.py train=Cape_pop_blazerlong_volleyball_etp train.mode=train model.prediction_texture_mask=False

python ./main.py train=Cape_pop_blazerlong_volleyball_etp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False

@REM 
@REM python Evaluate/chamfer_distance_kaolin.py  --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_volleyball_trial1/data/resampled_meshes" ^
@REM                                             --gt_header "" --gt_ext ".obj" ^
@REM                                             --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_pop_sigma4/5" ^
@REM                                              --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"
@REM 
@REM 
@REM 
@REM python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_climb_trial1/data" ^
@REM                                                         --dataset_type "pop" --Training_type "Extrapolation" ^
@REM                                                         --useID "True" --use_beta "True" ^
@REM                                                         --IDstart 0 --IDend 20 --IDstep 1 ^
@REM                                                         --norm_a 3 --Debug "False"
@REM 
@REM python ./main.py train=Cape_pop_blazerlong_climb_etp train.mode=train model.prediction_texture_mask=False
@REM 
@REM python ./main.py train=Cape_pop_blazerlong_climb_etp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False
@REM 
@REM python Evaluate/chamfer_distance_kaolin.py   --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_climb_trial1/data/resampled_meshes" ^
@REM                                                 --gt_header "" --gt_ext ".obj" ^
@REM                                                 --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_pop_sigma4/6" ^
@REM                                                 --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"
@REM 
@REM 
@REM 
@REM 
@REM python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/00096_shortlong/shortlong_squats/data" ^
@REM                                                         --dataset_type "pop" --Training_type "Extrapolation" ^
@REM                                                         --useID "True" --use_beta "True" ^
@REM                                                         --IDstart 0 --IDend 43 --IDstep 1 ^
@REM                                                         --norm_a 3 --Debug "False"
@REM 
@REM python ./main.py train=Cape_pop_shortlong_squats_etp train.mode=train model.prediction_texture_mask=False
@REM 
@REM python ./main.py train=Cape_pop_shortlong_squats_etp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False
@REM 
@REM python Evaluate/chamfer_distance_kaolin.py   --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/00096_shortlong/shortlong_squats/data/resampled_meshes" ^
@REM                                                 --gt_header "" --gt_ext ".obj" ^
@REM                                                --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_pop_sigma4/7" ^
@REM                                                 --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"
@REM 
@REM 
@REM 
@REM python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/00215_poloshort/poloshort_punching/data" ^
@REM                                                         --dataset_type "pop" --Training_type "Extrapolation" ^
@REM                                                         --useID "True" --use_beta "True" ^
@REM                                                         --IDstart 0 --IDend 33 --IDstep 1
@REM                                                         --norm_a 3 --Debug "False"
@REM 
@REM python ./main.py train=Cape_pop_poloshort_punching_etp train.mode=train model.prediction_texture_mask=False
@REM 
@REM python ./main.py train=Cape_pop_poloshort_punching_etp train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=result model.prediction_texture_mask=False
@REM 
@REM python Evaluate/chamfer_distance_kaolin.py   --gt_dir_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/00215_poloshort/poloshort_punching/data/resampled_meshes" ^
@REM                                                --gt_header "" --gt_ext ".obj" ^
@REM                                                --pred_dir_path "D:/Project/Human/AITS/reconst_0218_vs_pop_sigma4/8" ^
@REM                                                --pred_glob_name "mesh_[0-9][0-9][0-9][0-9].ply"
@REM 