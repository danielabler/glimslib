.. GlimSLib documentation master file, created by
   sphinx-quickstart on Thu May  3 10:14:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GlimSLib documentation
**********************

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This repository provides a library for development and simulation of PDE-based spatial tumor growth models, as well as implementations of specific tumor growth models.
It has been developed as part of the ['Glioma Mass-Effect Simulator' (GlimS)](http://glims.ch) project to investigate the role of tumor-induced mass-effect for tumor evolution and treatment.


Getting Started
===============

Please see the README file for installation and configuration instructions.


Test Cases
==========

Various application test cases for an implementation of a mechanically-coupled reaction-diffusion tumor growth model
are available in :py:meth:`test_cases`.


API
===

The "Simulation" Module
-----------------------

The `Simulation` module consists of a base class (*abstract class*) that implements
methods common to all problems addressed by GlimS.
It also defines various *abstract methods* that must be implemented to instantiate this class.

- Model-specific parameters in :py:meth:`simulation.simulation_base._define_model_params()`.
- Model-specific function space in :py:meth:`simulation.simulation_base._setup_functionspace()`.
- Model-specific governing form in :py:meth:`simulation.simulation_base._setup_problem()`.
- Model-specific parameter estimation problem in :py:meth:`simulation.simulation_base.run_for_adjoint()`.

Implementations of the simulation base class represent Simulation modules for specific PDE-based models.
An example is provided in :py:meth:`simulation.simulation_tumor_growth.py`.

Many functionalities of this base class rely on helper classes defined in :py:meth:`simulation.herlpers.helper_classes.py`.


Simulation Base Class (abstract)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simulation.simulation_base
   :members:
   :private-members:
   :show-inheritance:

Tumor Growth Simulation
^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: simulation.simulation_tumor_growth
   :members:
   :private-members:
   :show-inheritance:


Helper Classes
^^^^^^^^^^^^^^^

.. automodule:: simulation.helpers.helper_classes
   :members:
   :show-inheritance:


The "Visualisation" Module
--------------------------

The `Visualisation` module consists of a set of plotting functions for visualising 2D simulation results,
and overlay of image data with simulation results.
Some of these functions are also directly accessible from instances of the simulation classes via `self.plotting`.

Plotting Functions
^^^^^^^^^^^^^^^^^^^

.. automodule:: visualisation.plotting
   :members:
   :show-inheritance:


Helpers
^^^^^^^^^

.. automodule:: visualisation.helpers
   :members:
   :show-inheritance:



The "Utils" Module
--------------------------

The `Utils` module consists of helper functions for file system manipulation and (mesh) data import/export.

File System Utilities
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: utils.file_utils
   :members:
   :show-inheritance:

Data IO
^^^^^^^

.. automodule:: utils.data_io
   :members:
   :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
