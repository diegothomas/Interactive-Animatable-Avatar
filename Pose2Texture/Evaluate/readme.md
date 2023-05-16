# Chamfer distance
## 環境  
wsl(Ubuntu-20.04 , cuda-11.7)    
## インストール(チュートリアルから多少変更)    
```
conda create -n pytorch3d python=3.9  
conda activate pytorch3d  
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch  
conda install -c fvcore -c iopath -c conda-forge fvcore iopath 
conda install -c bottler nvidiacub 
conda install pytorch3d==0.6.2 -c pytorch3d
```

参照 : https://github.com/facebookresearch/pytorch3d/blob/v0.6.2/INSTALL.md 


# Rendering with blender
## 環境  
- Pose2Textureと同じ (conda activate AITS)
- Blender version 3.3
## 実行方法
1. render/render_config.json内を編集
2. python rendering.py
