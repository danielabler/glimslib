# GlimS

This repository provides an implementation of a
'Glioma Mass-Effect Simulator' (GlimS) using the
[FEniCS](https://fenicsproject.org) Finite Element library.
The simulation tool has been developed as part of the [GlimS](http://glims.ch)
project to investigate the role of tumor-induced mass-effect for
tumor evolution and treatment.

## Functionality & Design Criteria

The 'Glioma Mass-Effect Simulator' (GlimS) provides a framework for
development and simulation of PDE-based spatial tumor growth models.

It provides convenience functions for
- creating simulation domains from
segmented (medical) images, and for initializing tissue-specific
simulation parameters and boundary conditions on these domains.
- storing, and plotting simulation results.

GlimS is build in a way that allows existing growth models to be extended
and new growth models to be implemented easily while maintaining
the functionality of these convenience functions.
This facilitates the process testing and comparing different model
specifications in a consistent manner.
Various [standard growth models](#implemented-models) are already implemented in
GlimS.

GlimS is designed to support inverse-problems using FENICS adjoint.


## Implemented Models



## Global Switches

### FENICS vs dolfin-adjoint

Dolfin-adjoint works by redefining some of the base FEniCS commands. 
Usage of dolfin-adjoint therefore typically requires:

```
from fenics import *
from dolfin_adjoint import *
``` 
I found it preferable to import dolfin-adjoint only when it is actually needed, and stick to vanilla FEniCS otherwise. 
At the same time, we need to make sure that all modules in the library either work with the base FEniCS commands or 
with their dolfin-adjoint extended version. Mixing both imports results in internal errors.

This library uses the following mechanism for context-depending global switching between these import options:
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


### Matplotlib Settings & Backend

All project specific matplotlib settings are handled by the *matplotlibrc* file in the project root.
For this file to be considered by matplotlib, the working directory needs to be set to the project root.

The easiest way to install and use this project is via docker containers, see [installation instructions](#docker) below.
However, interactive plotting from these containers is troublesome; a possible workaround is described in 
*README* file of the included images. 

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
 
To ensure that the local *matplotlibrc* is used regardless of the working directory on the docker container, the environment
variable `MATPLOTLIBRC` is set to point to the configuration file in the project directory:
```
ENV MATPLOTLIBRC=/opt/project/matplotlibrc
```  

 

# Documentation

The documentation is based on *sphinx*. Compile by:

    cd docs
    make html


We use *unittest* for simple functiont tests. Run by:

    python -m unittest discover -p 'test_units*'


Computationally more expensive tests are provided in script form:
- code verification based on method of manufactured solutions
  for stationary diffusion problems in 2D & 3D

# Installation


## Docker

Instructions for building docker images are included in */dockerfiles/**.
These images extend the official [dolfin-adjoint](https://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html)
images with additional libraries needed for GlimS.

The current code only supports */dockerfiles/2017.2.0_libadjoint* which provides FEniCS 2017.2.0 (python 3.5) and
[dolfinadjoint/libadjoint](https://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html).
As development of [dolfinadjoint/libadjoint](https://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html) 
has been discontinued, we will update this project to be compatible with 
[dolfinadjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/download/) in the future (FENICS 2018.1.0, python 3.6).

Instructions for building and using these docker images can be found in /dockerfiles/*/README.md*.

## Anaconda


# Configuration

## PyCharm & Docker

PyCharm (Professional) can use a docker image (i.e. docker containers created from this image) as 
[remote interpreter](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html).
In the default configuration, PyCharm's working directory on the docker container is */opt/project*.
It "Run File in Console" command however, uses the project path on the host for calling the remote python interpreter, 
which fails for obvious reasons.
This can be fixed by mapping the local project path to the project's remote path, i.e.
*/local/path/to/project* -> */projeect/opt*, in *Preferences* -> *Project* -> *Project Interpreter* -> *Path Mappings*.


