# Documentatie PointCloud_Tree_Modelling
---

The repository [PointCloud_Tree_Modelling](https://github.com/Amsterdam-AI-Team/PointCloud_Tree_Modelling) contains methods for the automatic extraction of various characteristics of trees in point clouds using advanced analytical methods. The methods can serve as inspiration, or can be applied as-is under some specific assumptions. This document provides additional documentation for the repository.

---

## Data

When testing the repository on your point clouds, please make sure that:
 - your point cloud represents a single tree (i.e., the tree is segmented out from the background; no ground, no fence...);
 - the tree has an upright orientation (i.e., with Z-axis pointing up).
 - files follow the following naming convention `filtered_tree_<x>_<y>.las`,  where `<x>` and `<y>` are a 6 digit RD coordinate. For example, `filtered_tree_119305_485108.las`.

For processing a complete dataset of point cloud trees in batch, make sure the files are stored in the following structure:
* `name_of_dataset` _The main dataset folder_
   * `group_1` _subfolder containing trees of a particular group_
   * `group_2` _subfolder containing trees of a particular group_

An example of filtered point cloud trees is given below:
![Comparison of datasets (side-view)](./imgs/pc_comparison.png)

---

## AHN Preprocessing

In order to determine to height and location of the tree we use AHN as referece data for the ground level. To prodcue the AHN surface referenece for a tree one can use `clip_ahn_treecodes()` `preprocessing/ahn_preprocessing.py` for a particular treecode `<x>_<y>`


---

## Tree Utility Functions

**Leafwoord Classification**
One can use `leafwood_classification()` to classify the leaves in a point cloud tree. There are two methods available using different point features to determine if a point is either a leaf or wood. The features of choice are _Curvature_ and _Surface Variation_. **Note**: parameters might be optimised to work properly for a particular scan.

**Skeleton Reconstruction**
In order to reconstruct the skeleton of a tree one can use `reconstruct_skeleton()`, which calls a subprocess for the AdTree Library and returns a skeleton graph.

In addition, one can use `skeleton_split()` function to split the stem from the crown using the reconstructed tree skeleton.

To only separate the crown from the stem one can use `tree_separate()`.

**Stem Analysis**
The are multiple functions available that can derive various features from the stem (separated from the pointcloud tree).

  - `get_stem_endpoints()`, fits a cylinder and retrieves the start and end point of the stem.
  - `diameter_at_breastheight()`, fits a circle a breast height and determines the diameter.
  - `stem_to_mesh()`, constructs a mesh of the stem.

**Crown Analysis**
The are multiple functions available that can derive various features from the crown (separated from the pointcloud tree).

  - `crown_height()`, determines the height of the crown from start to top.
  - `crown_baseheight()`, determines the height of the crown above ahn surface.
  - `crown_diameter()`, determines the diameter of the crown at its maximum.
  - `crown_shape()`, determines the shape of the crown (Spherical, Conical, ...).
  - `crown_to_mesh()`, constructs a mesh of the crown using either _Convex Hull_ or _Alphashapes_ and also provides the volume.

**LOD Generation**
There are 3 available LOD generation methods (v2, v3.0, and v3.1). Depending on the version the methods require tree features like: (stem radius, stem basepoint, crown basepoint, crown height, crown mesh). **Note**: the LOD version naming is proposed and not a standard.

