python ./make_displacementTexture_for_2path_for_bat.py --root_path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_volleyball_trial1/data" ^
                                                         --dataset_type "pop" --Training_type "Interpolation" ^
                                                         --useID "True" --use_beta "True" ^
                                                         --IDstart 4 --IDend 74 --IDstep 10 ^
                                                         --norm_a 3 --Debug "True"

python ./Reconstruct_for_2path.py train=Debug_only_reconst_clothes_pop train.mode=test reconst.norm_a=3 reconst.use_beta=True train.save_mode=debug

@REM pause