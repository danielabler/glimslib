[![Coverage Status](https://coveralls.io/repos/github/danielabler/glimslib/badge.svg?branch=travisci)](https://coveralls.io/github/danielabler/glimslib?branch=travisci)
[![Build Status](https://travis-ci.com/danielabler/glimslib.svg?branch=master)](https://travis-ci.com/danielabler/glimslib)
[![Documentation Status](https://readthedocs.org/projects/glimslib/badge/?version=latest)](https://glimslib.readthedocs.io/en/latest/?badge=latest)

# GlimSLib

This repository provides a library for development and simulation of PDE-based spatial tumor growth models, as well as implementations of specific tumor growth models.
It has been developed as part of the ['Glioma Mass-Effect Simulator' (GlimS)](http://glims.ch) project to investigate the role of tumor-induced mass-effect for tumor evolution and treatment.


## Functionality

GlimSLib aims to support implementation of new and extension of existing tumor growth models by providing a consistent 
interface across model specifications.
Various convenience functions are included in GlimSLib to facilitate model instantiation and analysis:

- Creation of simulation domains from segmented (medical) images
- Initialization of tissue-specific simulation parameters and boundary conditions on these domains.
- Storing, and plotting simulation results.

Models implemented in GlimSLib automatically support 2D and 3D simulations, thanks to the abstractions provided by 
[FEniCS](https://fenicsproject.org) and the [DOLFIN](https://bitbucket.org/fenics-project/dolfin/src/master/) library.  

Inverse-problems can be adressed using the [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) library.

The following growth models have already been implemented in Glims and are included in this repository:

- mechanically-coupled reaction-diffusion model 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 


### Prerequisites

GlimSLib is written in python and requires version 3.5 or higher.
GlimSLib relies on the [FEniCS](https://fenicsproject.org) Finite Element library for solving forward models, and on [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) for inverse-problems.

This version of GlimSLib has been developed against FEniCS 2017.2.0 (python 3.5) and the corresponding [dolfinadjoint/libadjoint](https://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html) version.


### Installing

First, clone this repository into *<path_to_glimslib_dir>* on your local machine.

The easiest way to set up the project dependencies is by using the dockerfile description contained in this repository.


#### Docker 

A docker file description of the development setup is located in */dockerfiles/2017.2.0_libadjoint*.
This file extends the official [dolfin-adjoint](https://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html) images with additional python libraries needed for GlimSLib. 

- Get [Docker](https://docs.docker.com/install/#pre-releases) for your system.
- Build Docker image from dockerfile
  ```
  cd dockerfiles/2017.2.0_libadjoint
  docker build -t glimslib_image .
  ``` 
- Create Docker container from image
  ```
  docker run --name glimslib -w /opt/project -v <path_to_glimslib_dir>:/opt/project -t glimslib_image:latest
  ```
  - working directory */opt/project* in container
  - *<path_to_glimslib_dir>* is directory (on host) where this repository is located
   
  You can stop / restart this container by
  ```
  docker stop glimslib
  docker start glimslib
  ``` 
- Connect to running container
  ```
  docker exec -ti -u fenics glimslib /bin/bash -l
  ```  

For more information about the dockerfile, see */dockerfiles/2017.2.0_libadjoint/README.md*.

The development of [dolfinadjoint/libadjoint](https://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html) has been discontinued.
We plan to update this project to be compatible with [dolfinadjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/download/) (FENICS 2018.1.0, python 3.6).

#### Alternative Installation Methods

Please see the [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) installation pages for alternative installation methods of FEniCS and dolfin-adjoint.

In addition, the python packages listed in *dockerfiles/2017.2.0_libadjoint/requirements.txt* are required.

```
pip install --trusted-host pypi.python.org -r dockerfiles/2017.2.0_libadjoint/requirements.txt
```

### Running the Tests

We use *unittest* for simple function tests. 
Unittests for a class or function are located in the same directory where the class/function is defined.
All unittest files are prefixed with 'test_unit_*' and can therefore be run by:
```
python3 -m unittest discover -p '*test_unit_*'
```

### Running Examples

Computationally more expensive tests and usage examples for specific model implementations are provided in script form in the directory *test_cases/<model_name>*.

For example, to run a forward simulation of the `simulation_tumor_growth` model on a simple 2D domain with 2 subdomains, do:
```
python3 test_cases/test_simulation_tumor_growth/test_case_simulation_tumor_growth_2D_subdomains.py
```
and inspect the output in *output/test_cases/simulation_tumor_growth/test_case_simulation_tumor_growth_2D_subdomains*.


### Documentation

The documentation is based on *sphinx*. 
Compile by:
```
cd docs
make html
```

### Further Configuration Options


#### Parallel Execution

FEniCS and dolfin-adjoint support parallel execution via mpi.
Therefore any GlimSLib model can, in principle, be executed in parallel.

However, various support function in GlimSLib do not currently support mpi. 
For example, GlimSLib does not ensure that restuls of distributed computations are correctly collected in the simulation.helper_classes.Results class. Therefore, plotting during and after computation (simulation.helper_classesPostprocess) may not be working correctly.

Also, FENICS does not support VTK output in mpi run mode.

Therefore execute the `run` command with the following attributes:
```
sim.run(save_method='xdmf', plot=False, output_dir=output_path, clear_all=False) 
```

For example, see *test_cases/test_simulation_tumor_growth/test_case_simulation_tumor_growth_2D_uniform_mpi.py*

```
mpirun -np 4 python3 test_cases/test_simulation_tumor_growth/test_case_simulation_tumor_growth_2D_uniform_mpi.py
```

Imported vtk meshes need to be loaded from hdf5 in parallel execution.
The following scripts illustrate how

- an existing vtk mesh is converted to a FEniCS mesh and saved as hdf5
```
python3 est_cases/test_simulation_tumor_growth/convert_vtk_mesh_to_fenics_hdf5.py
```
- this converted mesh is loaded and used during parallel execution
```
mpirun -np 4 python3 test_cases/test_simulation_tumor_growth/test_case_simulation_tumor_growth_2D_uniform_mpi.py
```

#### FENICS vs dolfin-adjoint

Dolfin-adjoint works by redefining some of the base FEniCS commands. 
Usage of dolfin-adjoint therefore typically requires:

```
from fenics import *
from dolfin_adjoint import *
``` 
I found it preferable to import dolfin-adjoint only when it is actually needed, and work with standard FEniCS otherwise. 
At the same time, we need to make sure that all modules in the library either work with standard FEniCS commands or 
with their dolfin-adjoint extended version. Mixing both imports results in internal errors.

GlimSLib uses the following mechanism for context-dependent global switching between these import options:
We define a pseudo module `fenics_local` in the root of the project which imports either FEniCS, 
or FEniCS and dolfin-adjoint simultaneously, depending on the variable `USE_ADJOINT` in the *config.py* file.
Most modules and functions use import *config.py* to identify path settings. 

To activate dolfin-adjoint, include the following lines before importing any other GlimS modules.
```
import config
config.USE_ADJOINT = True
``` 
Then to import FEniCS / FEniCS with dolfin-adjoint, simply do
```
from fenics_local import *
```
or a more fine-grained import statement.
Such an import statement is also included in any other module that depends FEniCS / FEniCS with dolfin-adjoint
packages.

For usage examples compare 

- no adjoint: *test_cases/test_simulation_tumor_growth/test_case_simulation_tumor_growth_2D_uniform.py*
- with adjoint: *test_cases/test_simulation_tumor_growth/test_case_simulation_tumor_growth_2D_uniform_adjoint.py*


#### Matplotlib Settings & Backend

All project specific matplotlib settings are handled by the *matplotlibrc* file in the project root.
For this file to be considered by matplotlib, the working directory needs to be set to the project root.

The easiest way to install and use this project is via docker containers, see the installation instructions above.
However, interactive plotting from these containers is troublesome; a possible workaround is described in 
*README* file of the included dockerfile specifications. 

To be able to switch globally between interactive and non-interactive plotting, change the plotting 
[backend](https://matplotlib.org/tutorials/introductory/usage.html#what-is-a-backend) in *matplotlibrc* in the
project root, e.g.
* Non-Interactive: 
  * `Agg` for raster graphics
  * `Cairo` for vector graphics
* Interactive:
  * `TkAgg`

If a non-interactive backend is selected, `plt.show()` commands will be suppressed globally and, instead,
the generated images will be saved in a temporary folder *output/tmp_plots* and named by the current date-time.
 
To ensure that the local *matplotlibrc* is used regardless of the working directory in the docker container, the environment
variable `MATPLOTLIBRC` must be set to point to the configuration file in the project directory:
```
ENV MATPLOTLIBRC=/opt/project/matplotlibrc
``` 
The included dockerfile specification takes care of this.

### Development with PyCharm & Docker

PyCharm (Professional) can use a docker image (i.e. docker containers created from this image) as 
[remote interpreter](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html).

In the default configuration, PyCharm's working directory on the docker container is */opt/project*.
However, the "Run File in Console" command calls the remote python interpreter using the host project path by default.
This fails for obvious reasons...

To fix this problem, map the local project path to the project's remote path, i.e. in 
*Preferences* -> *Project* -> *Project Interpreter* -> *Path Mappings* define the mapping
*/local/path/to/project* -> */project/opt*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

