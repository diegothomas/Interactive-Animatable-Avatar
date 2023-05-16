# Guide to prepare the canonical meshes

## Build  
### Build the SkinWeights project

1. copy "CMakeLists_s.txt" to "CMakeLists.txt"
2. Run following codes

    ```
    mkdir build_skinweights  
    cd build_skinweights  
    cmake .. -D_CUDA_=TRUE -G"Visual Studio 16 2019"    (if Windows)
    cmake .. -D_APPLE_=TRUE -G"Xcode"   (if Mac)
    make                                (if Mac)
    ```

3. open the project and build.

### Build the AffectWeights project

1. copy "CMakeLists_a.txt" to "CMakeLists.txt"
2. Run following codes

    ```
    mkdir build_affectweights
    cd build_affectweights  
    cmake .. -D_CUDA_=TRUE -G"Visual Studio 16 2019"  (if Windows)
    cmake .. -D_APPLE_=TRUE -G"Xcode"   (if Mac)
    make                                (if Mac)
    ```

3. open the project and build.

<!--
# 1. Prepare SMPL pose parameters 
Use code in "FitSMPL_python" and save all pose parameters pose_****.bin in folder like "data/smplparams"

# 2. Make canonical point cloud

```
skinweights.exe --path_model "path/to/Template-star-0.015-0.05/" --path "path/to/<your_dataset>/data/" --size <size> --scan_name ""
```

=> That should generate a list of point clouds PC_***.ply in the folder "data/canonical_meshes" and a list of "skin_weights_****.bin" files in the folder "data/centered_meshes"
There are also additional files for debugging like canonical fitted SMPL meshes and recentered meshes.

# 3. Make a 3D mesh for each cloud of point
Use the "Poisson_rec.py" script to generate tight 3D mesh for each cloud of points using Poisson reconstruction.
```
python Poisson_rec.py --path "path/to/<your_dataset>/data/canonical_meshes" --size <size_num>
```

# 4. Generate skinweights for each canonical mesh to repose
```
affectweights.exe --path_model "path/to/Template-star-0.015-0.05/" --path "path/to/<your_dataset>/data/" --size <file size>  
```
--!>

