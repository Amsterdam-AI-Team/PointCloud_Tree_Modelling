# PointCloud Tree Modelling

This repository contains methods for the **automatic extraction of various characteristics of trees in point clouds** using advanced analytical methods. The methods can serve as inspiration, or can be applied as-is under some specific assumptions:

1. Usage in The Netherlands (The "[Rijksdriehoek coordinate system](https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten)");
2. Point clouds in LAS format.

Example [notebooks](./pctm/notebooks) are provided to demonstrate the tools.

![Comparison of datasets (side-view)](./imgs/pc_comparison.png)

---

## Project Goal

The goal of this project is to automatically extract various features such as tree height, crown volume, trunk diameter and other characteristics of trees from point clouds. One of the main challenges in this project is the sparsity and non-uniform density of tree point clouds. This project provides an analysis for multiple data sources.

We provide a pipeline that extracts various features from a **stand alone tree**. The input of this pipeline is a fully segmentated tree and produces a list of computed characteristics of the tree. These characteristics can be used for research or other purposes.

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

This code has been tested with `Python >= 3.8` on `Linux` and `MacOS`, and should likely work under Windows as well. There are two ways for installing the repository:

### Using Docker-image
To use this using the docker. Build the provide Dockerfile. code in development mode simply clone the repository and install the dependencies. Note if the image's platform (linux/amd64) does not match the host platform, significant drops in performance can occur.

1. Clone the repository

    ```bash
    # Clone the repository
    $ git clone https://github.com/Amsterdam-AI-Team/PointCloud_Tree_Modelling.git
    ```

2. Build the docker image (the building can take a couple of minutes):

    ```bash
    $ docker build -f Dockerfile . -t treemodelling:latest
    ```

3. Run docker container (as jupyter server on port 8888):

    ```bash
    $ docker run -v `pwd`/pctm:/usr/local/app/pctm -v `pwd`/dataset:/usr/local/app/dataset -it -p 8888:8888 treemodelling:latest
    ```

    The `-v` command is used to mount volumes for both data and code to the container.
    
    One could run the image in iteractive mode using the following run command: 
    ```
    $ docker run -v `pwd`/pctm:/usr/local/app/pctm -v `pwd`/dataset:/usr/local/app/dataset -it --entrypoint /bin/bash treemodelling:latest
    ```


### Build from scratch

1.  Clone the repository and install the dependencies.

    ```bash
    # Clone the repository
    $ git clone https://github.com/Amsterdam-AI-Team/PointCloud_Tree_Modelling.git

    # Install dependencies
    $ cd PointCloud_Tree_Modelling
    $ python -m pip install -r requirements.txt
    ```

2.  Building the AdTree executable 

    In this repository we use a modified version of [AdTree: Accurate, Detailed, and Automatic Modelling of Laser-Scanned Trees.](https://github.com/tudelft3d/AdTree) located in the [`AdTree`](./AdTree) folder. To build the excecutable some third-party libraries and dependencies must be included. Most dependencies are included in the distribution except [Boost](https://www.boost.org/). So you will need to have Boost installed first. 

    You need [CMake](https://cmake.org/download/) and of course a compiler to build AdTree:

    - CMake `>= 3.1`
    - a compiler that supports `>= C++11`

    AdTree has been tested on macOS (Xcode >= 8), Windows (MSVC >=2015), and Linux (GCC >= 4.8, Clang >= 3.3). Machines nowadays typically provide higher [support](https://en.cppreference.com/w/cpp/compiler_support), so you should be able to build AdTree on almost all platforms.

    - Use CMake to generate Makefiles and then build (Linux or macOS).
      ```
      $ mkdir -p AdTree/Release
      $ cd AdTree/Release
      $ cmake -DCMAKE_BUILD_TYPE=Release ..
      $ make
      ```

    - **Note:** In order for the python modules to find the build executable, you should set the correct path in [`pctm/src/config.py`](./pctm/src/config.py). For most systems this is something like `../../AdTree/Release/bin/AdTree`.

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
  $ cd pctm/scripts
  $ python script.py --in_folder '../../dataset' [--lod]
  ```

  The `--lod` argument is optional to produce lod models for the trees.
  
  The output is a csv file containing for each tree in the specified data folder a list of estimated tree features. After processing the csv file is written to the specified dataset folder.

---

## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License or (at your option) any later version. The full text of the license can be found in the accompanying LICENSE file.

---

## Acknowledgements

This repository was created by [Falke Boskaljon](https://falke-boskaljon.nl/) for the City of Amsterdam. It uses a modified version of [AdTree](https://github.com/tudelft3d/AdTree). For demo purposes we include sample data which was made publicly available by [AHN](https://www.ahn.nl/), and provided to the City of Amsterdam by [Cyclomedia](https://www.cyclomedia.com/en/producten/data-visualisatie/lidar-point-cloud) and [Sonarski](https://sonarski.com/).

Should you have any questions, comments, or suggestions, please contact us.

Amsterdam Intelligence, City of Amsterdam\
https://www.amsterdamintelligence.com


