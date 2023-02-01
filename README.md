# PointCloud Tree Modelling

This repository contains methods for the **automatic extraction of various characteristics of trees in point clouds** using advanced analytical methods. The methods can serve as inspiration, or can be applied as-is under some specific assumptions:

1. Usage in The Netherlands (The "[Rijksdriehoek coordinate system](https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten)");
2. Point clouds in LAS format.

Example [notebooks](./pctm/notebooks) are provided to demonstrate the tools.

![Comparison of datasets (side-view)](./imgs/pc_comparison.png)

---

## Project Goal

The goal of this project is to automatically extract various features such as height, width, volume, and other characteristics of trees from point clouds. One of the main challenges in this project is the sparsity and non-uniform density of tree point clouds. This project provides an analysis for multiple data sources.

The first solution we provide is a pipeline that extracts various features from a **stand alone trees**. The input of this pipeline is a fully segmentated tree and produces a list of computed characteristics of the tree. These characteristics can be used for research or other purposes.

An example of a volume estimation for a given point cloud tree:
![convex_hull.png](./imgs/crown_mesh_comparison.png)

For a quick dive into this repository take a look at our [complete solution notebook](./pctm/notebooks/Complete%20Solution.ipynb).

---

## Folder Structure

 * [`datasets`](./dataset) _Demo dataset to get started_
   * [`ahn_surf`](./dataset/ahn_surf) _Example surface reference for tree point clouds_
   * [`ahn`](./dataset/ahn) _Example tree point clouds in AHN scan_
   * [`cyclo media`](./dataset/cyclo) _Example tree point clouds in Cyclo Media scan_
   * [`sonarski`](./dataset/sonarski) _Example tree point clouds in Sonarski scan_
 * [`media`](./imgs) _Visuals_
 * [`pctm`](./pctm/) _Python Library_
   * [`notebooks`](./pctm/notebooks) _Jupyter notebook tutorials_
   * [`scripts`](./pctm/scripts) _Python scripts_
   * [`src`](./pctm/src) _Python source code_
    * [`utils`](./pctm/src/utils) _Utility functions_

---

## Installation

This code has been tested with `Python >= 3.10` on `Linux` and `MacOS`, and should likely work under Windows as well.

1.  To use this code in development mode simply clone the repository and install the dependencies.

    ```bash
    # Clone the repository
    git clone <github-url>

    # Install dependencies
    cd PointCloud_Tree_Modelling
    python -m pip install -r requirements.txt
    ```

2.  Build [AdTree](https://github.com/tudelft3d/AdTree)

    AdTree depends on some third-party libraries and most dependencies are included in the distribution except 
    [Boost](https://www.boost.org/). So you will need to have Boost installed first. 

    Note: AdTree uses a stripped earlier version of [Easy3D](https://github.com/LiangliangNan/Easy3D), which is not 
    compatible with the latest version.

    You need [CMake](https://cmake.org/download/) and of course a compiler to build AdTree:

    - CMake `>= 3.1`
    - a compiler that supports `>= C++11`

    AdTree has been tested on macOS (Xcode >= 8), Windows (MSVC >=2015), and Linux (GCC >= 4.8, Clang >= 3.3). Machines 
    nowadays typically provide higher [support](https://en.cppreference.com/w/cpp/compiler_support), so you should be 
    able to build AdTree on almost all platforms.

    - Use CMake to generate Makefiles and then build (Linux or macOS).
      ```
      $ cd AdTree 
      $ mkdir build
      $ cd build
      $ cmake -DCMAKE_BUILD_TYPE=Release ..
      $ make
      ```

---

## Data
Some test tree point clouds are provided in the '[`dataset`](./dataset)' folder.

**Note:** When testing on your point clouds, please make sure that:
 - your point cloud represents a single tree (i.e., the tree is segmented out from the background; no ground, no fence...);
 - tree point cloud files are stored with `filtered_tree_` prefix.
 - a separate folder can be used inside [`dataset`](./dataset) to group trees.
 - the tree has an upright orientation (i.e., with Z-axis pointing up).

---

## Usage

- Option 1, use the provided [notebooks](./pctm/notebooks) that demonstrate how the tools can be used.

- Option 2, use the command line to process a complete dataset. First, on has to pre-process the AHN data to create surface files using [AHN Preprocessing.ipynb](./pctm/notebooks/AHN%20Preprocessing.ipynb). Then use the following code.
  
  ```bash
  cd pctm/scripts
  python script.py --in_folder '../../dataset' [--lod]
  ```

  The `--lod` argument is optional to produce lod models for the trees.

---

## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License or (at your option) any later version. The full text of the license can be found in the accompanying LICENSE file.

---

## Acknowledgements

This repository was created by [Falke Boskaljon](https://falke-boskaljon.nl/) for the City of Amsterdam. Should you have any questions, comments, or suggestions, please contact us.

Amsterdam Intelligence, City of Amsterdam,

https://www.amsterdamintelligence.com,


