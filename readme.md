## Environment
- Windows / Ubuntu (WSL tested only)
- Anaconda

## Install
1. Build [Skinweights Project](./Canonicalization/README.md) in `Canonicalization/`.
2. Build [AffectWeights Project](./Canonicalization/README.md) in `Canonicalization/`.
3. Build [MatchSMPL Project](./Canonicalization/FitSMPL_python/README.md) in `Canonicalization/FitSMPL_python/`.
4. Build [deepanim Project](./data_process/README.md) in `data_process/`.
5. Set up the conda environment by running the command `conda env create -f AITS.yml` in the `env/` folder. Then , activate the environment with `conda activate AITS`.

## Data Generation (Up to creating fitting meshes)
1. Place the folder `datasets_template/data_name` in an appropriate location and rename `data_name` to your desired name.
2. Place input scans inside `data_name/data/src_meshes` folder (it is recommended to name them "0001.obj", "0002.obj", etc., but the extension can be `.obj` or `.ply`).
3. Execute `python .\Pose2Texture\utils\resample.py --path "path\to\[data_name]"`. If successful, the resampled meshes will be generated and placed in the `data\resampled_meshes\` folder.
4. Follow the instructions in [FitSMPL_python_README](./Canonicalization/FitSMPL_python/README.md) to initialize SMPL fitting.
5. Change the variable `Dataset_path` in `data_make.bat` to the path of `data_name`, and input the dataset size in `size`. Then, Run ```data_make.bat```.  
    If you get an error on the way, comment it out with ```@REM``` or something similar and re-run from the point where it stopped.

## Training, Testing, and Evaluation

To perform training, testing, and evaluation, execute a bat file in the ```Pose2Texture``` folder:
- pop_itp.bat: Interpolation testing using the dataset provided by pop
- pop_etp.bat: Extrapolation testing using the dataset provided by pop
- snarf_itp.bat: Interpolation testing using the dataset provided by snarf
- snarf_etp.bat: Extrapolation testing using the dataset provided by snarf

The bat file performs the following steps:
1. Creation of displacement texture
2. Network training
3. Testing
4. Evaluation

In addition, it is necessary to add and edit the yaml file with the conf name specified in the bat file (e.g., ```Cape_pop_blazerlong_volleyball_itp``` for ```pop_itp.bat```) in ```Pose2Texture/conf/train```.

You can check the training loss graphs etc. by running ```mlflow ui``` in ```Pose2Texture/```. ([mlflow](https://mlflow.org/))

## 4D dataset
Our own 4D dataset is available for download with fitted SMPL parameters at the following url:
https://archive.iii.kyushu-u.ac.jp/public/FjIrAHGJbYBLDlpRzQ_R5gt2Xg8nMQ0F-_0wxgka-jzD

(内部者へ)
Gitlabに付随のWikiに詳細を記載しています.公開する際には、公開すべきでない情報(パス等)が含まれていないかの確認をお願いします。
