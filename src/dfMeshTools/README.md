dfMeshTools
===========
A collection of tools for mesh generation and manipulation, developed as part of the DeepFlame project.

## List of tools
### cellSources
1. cylinderCutToCell: A tool to cut a cylindrical region from a cell in the mesh
### searchableSurfaces
2. searchableCone: A tool to define a searchable cone surface for mesh generation. Typically used for defining ROI (region of interest) in the mesh where finer resolution is needed.

## Usage
**NOTE**: In `controlDict`, the library should be added to `libs`:
```cpp
libs ( "libdfMeshTools.so" );
```

1. For `cellSources`, the tool can be used in `topoSetDict` as follows:
    ```cpp
    actions
    (

        {
            name    cylinderCut;
            type   cellSet;
            action  new;
            source  cylinderCutToCell;
            sourceInfo
            {
                p1      (0 0 0.015);
                p2      (0 0 0.025);
                radius  0.0296;
            }
        }
    );
    ```

2. For `searchableSurfaces`, the tool can be used in `snappyHexMeshDict` as follows:
    ```cpp
    geometry
    {
        refineRegion
        {
            type            searchableCone;
            point1          (0.0   0 0);
            point2          (0.015 0 0);
            radius1         0.005;
            radius2         0.007;
        }
    }
    ```