# Building the AdTree executable 

In this repository we use a modified version of [AdTree: Accurate, Detailed, and Automatic Modelling of Laser-Scanned Trees.](https://github.com/tudelft3d/AdTree) located in this folder to create an executable. To build the excecutable some third-party libraries and dependencies must be included. Most dependencies are included in the distribution except [Boost](https://www.boost.org/). So you will need to have Boost installed first. 

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

- **Note:** In order for the python modules to find the build executable, you should set the correct path in [`pctm/src/config.py`](../pctm/src/config.py). For most systems this is something like `../../AdTree/Release/bin/AdTree`.
