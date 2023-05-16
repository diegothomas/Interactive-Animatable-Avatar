set Dataset_path=D:/data_name           @REM change to your path


@REM Don't touch here
set data_dir_name=/data/
set Data_dir_path="%Dataset_path%%data_dir_name%"
set assets_path = "assets/for_fitting/"

@REM echo %Dataset_path%
@REM echo %Data_dir_path%

Canonicalization\build_skinweights\Release\skinweights.exe --path_model %assets_path%^
                --path %Data_dir_path%^
                --size 2 --scan_name "smpl_" --smpl_flg "True" --smplparams_dir_name "smplparams_pop"  

python Canonicalization\Poisson_rec.py --path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_volleyball_trial1/data/canonical_smpls" --size 66

Canonicalization\build_affectweights\Release\affectweights.exe --path_model "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.015-0.05/"^
                     --path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_volleyball_trial1/data/"^
                     --size 2 --smpl_flg "True"

data_process\build\Release\deepanim.exe -mode "FitTetraToCanonicalFromOut"^
                -path "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Cape_POP/AITS/03375_blazerlong/blazerlong_volleyball_trial1/"^
               -templatepath0015 "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.015-0.05/"^
               -templatepath0008 "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.008/"^
               -file_num 2 --smpl_flg "True" --smplparams_dir_name "smplparams_pop"

python data_process\Revert_position.py --path "Z:\Human\b20-kitamura\AvatarInTheShell_datasets\Cape_POP\AITS\03375_blazerlong\blazerlong_volleyball_trial1/data"^
                                       --smpl_flg "True" --smplparams_name "smplparams_pop"