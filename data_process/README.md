# REQUIREMENTS
For windows
- Visual Studio
- cmake
- CUDA
- cudnn

For Mac

# Build deepanim Project
Run cmake file

```
mkdir build
cd build
cmake .. -D_CUDA_=TRUE -G"Visual Studio 16 2019"    (for Windows)
cmake .. -D_APPLE_=TRUE -G"Xcode"   (for Mac)
make    (for Mac)
```
 
<!---
and for get color UV corresponding,
```
deepanim.exe -mode "GetOriginalUV" -inputpath "path/to/dataset" -outputpath "path/to/dataset/data/FittedMesh_pose/" -templatepath "path/to/Template-star-0.015-0.05/"  -file_num <file_num>
(example) 
deepanim.exe -mode "GetOriginalUV" -inputpath "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/4D_datasets/Iwamoto/" -outputpath "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/4D_datasets/Iwamoto/data/FittedMesh_pose/" -templatepath "Z:/Human/b20-kitamura/AvatarInTheShell_datasets/Whole/Template-star-0.015-0.05/"  -file_num 209
--->


