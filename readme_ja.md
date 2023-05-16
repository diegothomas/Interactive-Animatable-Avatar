## Environment
- Windows / Ubuntu(Only tested WSL)
- Anaconda

## Install
1. ```Canonicalization/```内で、[Skinweights Project](./Canonicalization/README.md)をビルドする.
2. ```Canonicalization/```内で、[AffectWeights Project](./Canonicalization/README.md)をビルドする.
3. ```Canonicalization/FitSMPL_python/```内で、[MatchSMPL Project](./Canonicalization/FitSMPL_python/README.md)をビルドする.
4. ```data_process/```内で、[deepanim Project](./data_process/README.md)をビルドする. 
5. ```env/```内で```conda env create -f AITS.yml```コマンドでconda 環境を構築. その後、 ```conda activate AITS```で環境内に入る.

## データ生成(Fittingメッシュの作成まで)
1. ```datasets_template/data_name```フォルダーを適当な場所に置き、```data_name```を適当な好きなものに変える.
2. ```data_name/data/src_meshes```フォルダの中に入力スキャンを配置("0001.obj","0002.obj"... という風に配置することをおすすめします.(拡張子はobj/ply、どちらでも可))
3. ```python .\Pose2Texture\utils\resample.py --path "path\to\[data_name]``` を実行する. 成功すると、```data\resampled_meshes\```フォルダの中にリサンプリングされたメッシュが生成配置される. 
4. [FitSMPL_python_README](./Canonicalization/FitSMPL_python/README.md)を参考にして、SMPL fittingのinitializationを行う.
5. ```data_make.bat```内の変数```Dataset_path```を```data_name```のパスに変え、```size```にデータセットのサイズを入力し、```data_make.bat```を実行する.
    途中でエラーが出た場合は、```@REM```等でコメントアウトして、止まった箇所から再実行してください.

## トレーニング及びテスト、評価
```Pose2Texture``` フォルダ内で、batファイルを実行
- pop_itp.bat    : popの提供するデータセットでの内挿テスト
- pop_etp.bat    : popの提供するデータセットでの外挿テスト
- snarf_itp.bat　: snarfの提供するデータセットでの内挿テスト
- snarf_etp.bat  : snarfの提供するデータセットでの外挿テスト
のいずれかを参考にする

batファイル内では
1. displacement textureの作成
2. ネットワークの学習
3. テスト
4. 評価
を行う.

また、bat内に記載のconf名(```pop_itp.bat```なら、```Cape_pop_blazerlong_volleyball_itp```)のyamlファイルを```Pose2Texture/conf/train```に追加、編集する必要がある.

トレーニングのLossグラフ等は```Pose2Texture/```内で```mlflow ui```を実行して確認可能.([mlflow](https://mlflow.org/))