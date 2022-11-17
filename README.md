# PointCloud Tree Analysis

This repository contains methods for the **automatic extraction of various characteristics of trees in PointClouds** using advanced analytical methods. The methods can serve as inspiration, or can be applied as-is under some specific assumptions:

1. Usage in The Netherlands (The "[Rijksdriehoek coordinate system](https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten)");
2. Point clouds in LAS format and tiled following [specific rules](datasets); and
3. Fusion with [AHN](https://www.ahn.nl/) and [BGT](https://www.geobasisregistraties.nl/basisregistraties/grootschalige-topografie) public data sources.

Example [notebooks](notebooks) are provided to demonstrate the tools.

---

## Project Goal

The goal of this project is to automatically extract various features such as height, width, lowest branch, and other characteristics in point clouds. This information can be of value for tree health research and more. One of the main challenges in this project are overlapping trees, since it is difficult to distinguish in 3D point clouds which part belongs to which tree. Moreover, the sparsity and non-uniform density of typical point clouds makes robust methods difficult.

The first solution we provide is a pipeline that extracts various features from a **stand alone tree**. The input of this pipeline is a fully segmentated tree and produces a list of computed characteristics of the tree. These characteristics can be used for research or other purposes.

We also provide some **tree segementation** methods that splits groups of trees into individual ones. For example, one can use the tree segmentation method to retrieve individual trees that can be used as input for the stand alone tree pipeline.

For a quick dive into this repository take a look at our [complete solution notebook](notebooks/0.%20Complete%20solution.ipynb).

---

## Folder Structure

 * [`datasets`](./datasets) _Demo dataset to get started_
   * [`ahn`](./datasets/ahn) _AHN data_
   * [`bgt`](./datasets/bgt) _BGT data_
   * [`pointcloud`](./datasets/pointcloud) _Example urban point cloud_
 * [`media`](./media) _Visuals_
   * [`examples`](./media/examples)
 * [`notebooks`](./notebooks) _Jupyter notebook tutorials_
 * [`scripts`](./scripts) _Python scripts_
 * [`src/upcp`](./src/upcp) _Python source code_
   * [`analysis`](./src/upcp/analysis) _Dataset analysis code_
   * [`fusion`](./src/upcp/fusion) _Data fusion code_
   * [`preprocessing`](./src/upcp/preprocessing) _Pre-processing code_
   * [`region_growing`](./src/upcp/region_growing) _Region growing code_
   * [`scrapers`](./src/upcp/scrapers) _Data scrapers_
   * [`utils`](./src/upcp/utils) _Utility functions_

---

## Installation

This code has been tested with `Python >= 3.8` on `Linux` and `MacOS`, and should likely work under Windows as well.

1.  To use this code in development mode simply clone the repository and install the dependencies.

    ```bash
    # Clone the repository
    git clone https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing.git

    # Install dependencies
    cd Urban_PointCloud_Processing
    python -m pip install -r requirements.txt
    ```

    or, with Conda:
    ```bash
    conda env create -f environment.yml
    ```

2.  Alternatively, the code can be installed as a Python package from source:

    ```bash
    # Install the latest release as Wheel
    python -m pip install https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing/releases/download/v0.1/upcp-0.1-py3-none-any.whl

    # Alternatively, install the latest version from source
    python -m pip install git+https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing.git#egg=upcp

    # Or, after making changes in the code
    cd Urban_PointCloud_Processing
    python -m pip install .
    ```

    If you use the latter and want your code changes to take effect without re-installing the package, use the `--editable` flag for pip.

---

## Usage

We provide tutorial [notebooks](notebooks) that demonstrate how the tools can be used.

For visualisation of the resulting labelled point clouds we suggest [CloudCompare](https://www.danielgm.net/cc/). Simply open the labelled .laz in CloudCompare, select the cloud, and set `Colors` to the custom `Scalar Field` named `label`. For best results, use our [custom "AMS" color scale](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing/raw/main/media/cc_color_scale.xml), and set "steps" to 101.

---

## Acknowledgements

This repository was created by [Falke Boskaljon](https://falke-boskaljon.nl/) for the City of Amsterdam.
