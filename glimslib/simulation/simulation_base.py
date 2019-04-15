"""
Contains the base class from which all specific simulations inherit.
The base class does not provide implementations for the following methods:

- Model-specific parameters in :py:meth:`simulation.simulation_base._define_model_params()`.
- Model-specific function space in :py:meth:`simulation.simulation_base._setup_functionspace()`.
- Model-specific governing form in :py:meth:`simulation.simulation_base._setup_problem()`.
- Model-specific parameter estimation problem in :py:meth:`simulation.simulation_base.run_for_adjoint()`.

These need to be defined for each implementation of this base class.
"""

import logging
import os
from abc import ABC, abstractmethod

from glimslib import fenics_local as fenics
from glimslib.simulation_helpers.helper_classes import SubDomains, FunctionSpace, \
                                            BoundaryConditions, Parameters, Results, Plotting
from glimslib.simulation import config

# FENICS (and related) Logger settings
if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.WARNING)
else:
    fenics.set_log_level(fenics.LogLevel.WARNING)
logger_names = ['FFC', 'UFL', 'dijitso', 'flufl']
for logger_name in logger_names:
    try:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
    except:
        pass


class FenicsSimulation(ABC):
    """
    This class serves as 'master class' for the definition of time-dependent FEniCS problems that may differ in their
    underlying governing forms and input parameters.
    The implementation is optimised for the class of problems of interest in GlimS.
    This implies:

    - support for dolfin-adjoint based parameter assimilation
    - automatic construction of subdomains and boundary specifications from segmentation label maps to facilitate
      image-based modeling

    An implementation of this class requires at least the following abstract methods to be implemented::

        class NewSimulationClass(FenicsSimulation):

            def _setup_functionspace(self):
                ...

            def setup_model_parameters(self, **kwargs):
                ...

            def _setup_problem(self, u_previous):
                ...

            def run_for_adjoint(self, u_previous):
                ...

    It is instantiated and configured by::

        sim = NewSimulationClass()
        sim.setup_global_parameters(...)
        sim.setup_model_parameters(...)
        sim.run(...)

    and run by::

        sim.run(...)

    or::

        sim.run_for_adjoint(...)


     .. warning::
        Parameters for heterogeneous subdomains are internally defined as `DiscontinuousScalar` Expressions,
        see :py:meth:`simulation.helpers.helper_classes.DiscontinuousScalar`.
        This works well for forward simulation, however, FEniCS dolfin-adjoint does not seem to handle them well.

     .. warning::
        Von Neumann 'named_boundary' and 'subdomain_boundary' boundary conditions cannot be mixed in the current
        implementation, see :py:meth:`simulation.helpers.helper_classes.BoundaryConditions` for details.
        Also, in the current implementation, von Neumann BCs can only be applied to mesh boundaries not to interfaces
        between subdomains.
    """

    def __init__(self, mesh, time_dependent=True):
        """
        Init routine.
        :param mesh: The mesh.
        :param time_dependent: Boolean switch indicating whether this simulation is time-dependent.
        """
        self.logger = logging.getLogger(__name__)
        self.mesh = mesh
        self.geometric_dimension = self.mesh.geometry().dim()
        self.time_dependent = time_dependent
        self.projection_parameters = {'solver_type':'cg',
                                        'preconditioner_type':'amg'}
        self.functionspace = FunctionSpace(self.mesh, projection_parameters=self.projection_parameters)
        self._define_model_params()
        if fenics.is_version("<2018.1.x"):
            pass
        else:
            if config.USE_ADJOINT:
                self.tape = fenics.get_working_tape()

    @abstractmethod
    def _define_model_params(self):
        """
        Defines the models parameters expected by :py:meth:`setup_model_parameters`.
        In addition to these, time dependent simulations require parameters:

            - `sim_time`
            - `sim_time_step`
        """
        self.required_params = ['a', 'b']
        self.optional_params = ['c', 'd']

    @abstractmethod
    def _setup_functionspace(self):
        """
        This is a test_cases implementation that needs to be overwritten.
        Any implementation of this function must initialize self.functionspace.
        """
        # Element definitions
        displacement_element = fenics.VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
        concentration_element = fenics.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        element = fenics.MixedElement([displacement_element, concentration_element])
        subspace_names = {0: 'displacement', 1: 'concentration'}
        # Initialisation of self.functionspace
        self.functionspace.init_function_space(element, subspace_names)

    @abstractmethod
    def _setup_problem(self, u_previous):
        """
        This function takes the initial value function u_previous as input and, after its execution,
        is expected to provide a fenics.solver as class instance attribute `self.solver`.
        The function needs to be overloaded with the model-specific problem definition, including:

        - governing form
        - problem
        - solver

        The function is executed by :py:meth:`self.run()`.

        :param u_previous: initial value function, or previous solution
        """

    @abstractmethod
    def run_for_adjoint(self, parameters):
        """
        Run the time-dependent simulation with minimum number of updated parameters for adjoint optimisation.
        :param parameters: list of parameters
        """

    def setup_global_parameters(self, label_function=None, subdomains=None, domain_names=None, boundaries = None,
                                dirichlet_bcs=None, von_neumann_bcs=None):
        """
        This function bundles the setup of 'global' simulation properties that remain unchanged
        when other simulation parameters are modified.

        We separate initialisation of those parameters here from :py:meth:`self.setup_model_parameters` to ensure that all
        simulations can be rerun with different parameter settings but utilising the same (identical) functionspace and
        mesh.

        After execution, the class instance has attributes:

        - `functionspace`, an instance of :py:meth:`simulation.helpers.helper_classes.FunctionSpace`
            with subspaces if appropriate.
        - `subdomains`, an instance of :py:meth:`simulation.helpers.helper_classes.SubDomains`
        - `bcs`, an instance of :py:meth:`simulation.helpers.helper_classes.BoundaryConditions`

        :param label_function: a `fenics.Function()` assigning integer value to each point of domain, e.g. segmentation image.
        :param subdomains: a `fenics.MeshFunction()` assigning value to each cell of domain.
        :param domain_names: a dictionary mapping each value in `label_function` or `subdomains` to a string.
        :param boundaries: da dictionary of `{name : function}` containing boundaries defined by functions
        :param dirichlet_bcs: a dictionary of the form `{bc_name : bc }`.
        :param von_neumann_bcs: a dictionary of the form `{bc_name : bc }`.
        """
        self.logger.info("-- Setting up global parameters")
        self.logger.info("   - assigning _mesh")
        self.geometric_dimension = self.mesh.geometry().dim()

        # Subdomains
        self.subdomains = SubDomains(self.mesh)
        self.subdomains.setup_subdomains(label_function=label_function, subdomains=subdomains, replace=False)
        self.subdomains.setup_boundaries(tissue_map=domain_names, boundary_fct_dict=boundaries)
        self.subdomains.setup_measures()
        # Function space
        self._setup_functionspace()
        # Boundary Conditions
        self.bcs = BoundaryConditions(self.functionspace, self.subdomains)
        self.bcs.setup_dirichlet_boundary_conditions(dirichlet_bcs)
        self.bcs.setup_von_neumann_boundary_conditions(von_neumann_bcs)

    def setup_model_parameters(self, iv_expression, **kwargs):
        """
        Initialises model-specific parameters.

        :param iv_expression:
        :param kwargs: keyword-value pairs corresponding to the required parameters defined in `self.required_params`.
            In time-dependent simulations, additionally the following parameters must be defined:

            - `sim_time`: total simulation time
            - `sim_time_step`: simulation time step
        """
        self._define_model_params()
        self.params = Parameters(self.functionspace, self.subdomains,
                                 time_dependent=self.time_dependent)
        self.params.set_initial_value_expressions(iv_expression)
        self.params.define_required_params(self.required_params)
        self.params.define_optional_params(self.optional_params)
        self.params.init_parameters(kwargs)

    def _update_expressions(self, time):
        """
        Updates parameters and boundary conditions with current time.
        :param time: current time.
        """
        self.params.time_update_parameters(time)
        self.bcs.time_update_bcs(time, kind='dirichlet')
        self.bcs.time_update_bcs(time, kind='von-neumann')

    def _update_mesh_displacements(self, displacement):
        """
        Applies displacement function to mesh.
        .. warning:: This changes the current mesh! Multiple updates result in additive mesh deformations!
        """
        fenics.ALE.move(self.mesh, displacement)
        self.mesh.bounding_box_tree().build(self.mesh)

    def run(self, keep_nth=1, save_method='xdmf', clear_all=False, plot=True,
            output_dir=config.output_dir_simulation_tmp):
        """
        Run the time-dependent simulation.
        :param keep_nth: keep every nth simulation step
        :param save_method : None, 'vtk', 'xdmf'
        """
        if self.geometric_dimension==3:
            plot=False

        self.logger.info("-- Computing solutions: ")
        # Results instance
        self.results = Results(self.functionspace, self.subdomains, output_dir=output_dir)
        self.results.save_solution_start(method=save_method, clear_all=clear_all)
        # Plotting
        self.plotting = Plotting(self.results, output_dir=os.path.join(output_dir, 'plots'))
        # Initial Conditions
        u_previous = self.params.create_initial_value_function()
        self._setup_problem(u_previous)

        if not self.time_dependent:
            self.logger.info("    - solving stationary problem")
            self.solver.solve()
            self.results.add_to_results(0, 0, 0, self.solution)
            self.results.save_solution(0, 0, method=save_method)
            if plot:
                self.plotting.plot_all(0)
            u_previous.vector()[:] = self.solution.vector()
        else:
            # == t=0
            current_sim_time = 0.0
            self._update_expressions(current_sim_time)
            time_step = 0
            recording_step = 0
            u_0 = u_previous
            self.results.add_to_results(0, 0, recording_step, u_0)
            self.results.save_solution(recording_step, current_sim_time, function=u_0, method=save_method)
            if plot:
                self.plotting.plot_all(recording_step)
            continue_simulation = True
            # == t>0
            while (current_sim_time <= self.params.sim_time - 1e-5) and continue_simulation:
                if hasattr(self, 'tape')  :
                    with self.tape.name_scope("Timestep"):
                        current_sim_time += float(self.params.sim_time_step)
                        time_step = time_step + 1
                        self._update_expressions(current_sim_time)
                        self.logger.info("    - solving for time = %.2f / %.2f" % (current_sim_time, self.params.sim_time))
                        try:
                            self.solver.solve()
                        except:
                            self.logger.warning("    - Solver did not converge -- will shutdown simulation")
                            continue_simulation = False
                        if (time_step % keep_nth == 0) and continue_simulation:
                            recording_step = recording_step + 1
                            self.results.add_to_results(current_sim_time, time_step, recording_step, self.solution)
                            self.results.save_solution(recording_step, current_sim_time, method=save_method)
                            if plot:
                                self.plotting.plot_all(recording_step)
                        u_previous.assign(self.solution)
                else:
                    current_sim_time += float(self.params.sim_time_step)
                    time_step = time_step + 1
                    self._update_expressions(current_sim_time)
                    self.logger.info("    - solving for time = %.2f / %.2f" % (current_sim_time, self.params.sim_time))
                    try:
                        self.solver.solve()
                    except:
                        self.logger.warning("    - Solver did not converge -- will shutdown simulation")
                        continue_simulation = False
                    if (time_step % keep_nth == 0) and continue_simulation:
                        recording_step = recording_step + 1
                        self.results.add_to_results(current_sim_time, time_step, recording_step, self.solution)
                        self.results.save_solution(recording_step, current_sim_time, method=save_method)
                        if plot:
                            self.plotting.plot_all(recording_step)
                    u_previous.assign(self.solution)

        self.results.save_solution_end(method=save_method)
        # save entire time series as hdf5
        self.results.save_solution_hdf5()
        return self.solution

    def reload_from_hdf5(self, path_to_hdf5, output_dir=config.output_dir_simulation_tmp):
        self.logger.info("-- Reloading from hdf5: ")
        # Results instance
        self.results = Results(self.functionspace, self.subdomains, output_dir=output_dir)
        self.results.data.load_from_hdf5(path_to_hdf5)
        # Plotting
        self.plotting = Plotting(self.results, output_dir=os.path.join(output_dir, 'plots'))