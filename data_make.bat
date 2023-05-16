@echo off
@REM change to your path
set Dataset_path=D:/data_name/
set /a size=3

@REM Don't change here
set Data_dir_path="%Dataset_path%data/"
set canonical_path="%Dataset_path%data/canonical_meshes"

set assets_path="./assets/for_fitting/"
set assets_0008_path="./assets/for_fitting/Template-star-0.008/"

@REM echo %Dataset_path%
@REM echo %Data_dir_path%
@REM echo %canonical_path%
@REM echo %assets_path%
@REM echo %assets_0008_path%


echo ### 1 : Refine_FitSMPL_Scan init ###########################
cd ./Canonicalization/FitSMPL_python
python Refine_FitSMPL_Scan.py --path %Dataset_path% --init_flg True --start_id 1 --resampled_flg True
SET RESULT=%ERRORLEVEL%
cd ../..
if %ERRORLEVEL% geq 1 (
    echo "stop in : 1.Refine_FitSMPL_Scan init"
    exit /b
)



echo ### 2 : Refine_FitSMPL_Scan Loop ###########################
cd ./Canonicalization/FitSMPL_python
python Refine_FitSMPL_Scan.py --path %Dataset_path% --init_flg False --start_id 1 --size %size% --resampled_flg True
SET RESULT=%ERRORLEVEL%
cd ../..
if %ERRORLEVEL% geq 1 (
    echo "stop in : 2.Refine_FitSMPL_Scan Loop"
    exit /b
)



echo ### 3 : skinweights.exe ###########################
Canonicalization\build_skinweights\Release\skinweights.exe^
                --path_model %assets_path% --path %Data_dir_path%^
                --size %size% --scan_name "" --smpl_flg "False" --smplparams_dir_name "smplparams"  
if %ERRORLEVEL% geq 1 (
    echo "stop in : 3.skinweights.exe"
    exit /b
)



echo ### !!! 4 : Poisson_rec.py ###########################
python Canonicalization\Poisson_rec.py --path %canonical_path% --size %size%
if %ERRORLEVEL% geq 1 (
    echo "stop in : 4.Poisson_rec.py"
    exit /b
)



echo ### 5 : affectweights.exe ###########################
Canonicalization\build_affectweights\Release\affectweights.exe --path_model %assets_path%^
                     --path %Data_dir_path%^
                     --size %size% --smpl_flg "False"
if %ERRORLEVEL% geq 1 (
    echo "stop in : 5.affectweights.exe"
    exit /b
)



echo ### 6 : deepanim.exe ###########################
data_process\build\Release\deepanim.exe -mode "FitTetraToCanonicalFromOut"^
                -path %Dataset_path%^
                -templatepath0015 %assets_path%^
                -templatepath0008 %assets_0008_path%^
                -file_num %size% --smpl_flg "False" --smplparams_dir_name "smplparams"
if %ERRORLEVEL% geq 1 (
    echo "stop in : 6.deepanim.exe"
    exit /b
)



echo ### 7 : Revert_position.py ###########################
python data_process\Revert_position.py --path %Data_dir_path%^
                                       --smpl_flg "False" --smplparams_name "smplparams"
if %ERRORLEVEL% geq 1 (
    echo "stop in : 7.Revert_position.py"
    exit /b
)