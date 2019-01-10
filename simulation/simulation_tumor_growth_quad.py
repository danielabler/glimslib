"""
This class implements a mechanically-coupled reaction-diffusion model of tumor growth.

"""
from numpy import zeros
import fenics_local as fenics
import simulation.helpers.math_linear_elasticity
from simulation.simulation_base import FenicsSimulation
from simulation.helpers.helper_classes import PostProcessTumorGrowth
import simulation.helpers.math_linear_elasticity as mle
import simulation.helpers.math_reaction_diffusion as mrd
import simulation.config as config


class TumorGrowth(FenicsSimulation):
    """
    This class implements a mechanically-coupled reaction-diffusion model of tumor growth, with components:

    1. Reaction diffusion model of tumor cell concentration `c` with logistic growth term:

    .. math::
        :nowrap:

        \\begin{equation}
          \\frac{\partial c}{\\partial t} = D \\, \\nabla \\cdot \\nabla c + r \\, c \\, (1-c)
        \\end{equation}

        Model parameters are:

        - diffusion_constant `D` (isotropic, constant) [:math:`L^2/T`]
        - proliferation_rate `r` [:math:`1/T`]

    2. Linear-elstic continuum mechanics model material parameters:

        - young's modulus `E` (isotropic) [:math:`N/m^2`]
        - poisson ratio `nu`

    3. Coupling term, linking tumor cell concentration `c` and isotropic strain.

        - coupling parameter

    .. warning::

        Limitations:

            - Adjoint parameter optimization not possible for subdomain-specific paramaters.
            - von Neumann boundary conditions cannot be applied between subdomain interfaces.


    .. note::
        Parameter values and Units:

        Units of input parameters depend on the spatial units of the domain.
        The duration of a time step depends on the time units chosen for time-dependent parameters.
        Often, spatial units will be in :math:`mm`: and time will be measured in days `d`.

        Typical values for tumor growth models:

            - diffusion constant `D`: 0.02 ... 0.10 mm^2/d
            - proliferation `r`:      0.04 ... 0.08 1/d
            - young's modulus `E`:    1 ... 10 N/m^2 = 0.001 ... 0.010 N/mm^2
            - poisson ratio `nu`:     0.4 ... 0.49

    """

    def _setup_functionspace(self):
        displacement_element = fenics.VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
        concentration_element = fenics.FiniteElement("Lagrange", self.mesh.ufl_cell(), 2)
        element = fenics.MixedElement([displacement_element, concentration_element])
        subspace_names = {0: 'displacement', 1: 'concentration'}
        self.functionspace.init_function_space(element, subspace_names)

    def _define_model_params(self):
        self.required_params = ['diffusion', 'coupling', 'proliferation', 'E', 'poisson']
        self.optional_params = []

    def _setup_problem(self, u_previous):
        dim = self.geometric_dimension
        dx = self.subdomains.dx
        ds = self.subdomains.ds
        dsn = self.subdomains.dsn

        mu = mle.compute_mu(self.params.E, self.params.poisson)
        lmbda = mle.compute_lambda(self.params.E, self.params.poisson)
        diff_const = self.params.diffusion
        prolif_rate = self.params.proliferation
        coupling = self.params.coupling

        # # This is the mechanical body force
        if not hasattr(self,'body_force'):
            self.body_force = fenics.Constant(zeros(dim))

        # This is the RD source term
        if not hasattr(self, 'source_term'):
            self.source_term = fenics.Constant(0.0)

        du = fenics.TrialFunction(self.functionspace.function_space)
        v0, v1 = fenics.TestFunctions(self.functionspace.function_space)
        self.solution = fenics.Function(self.functionspace.function_space)
        self.solution.label = 'solution_function'

        sol0, sol1 = fenics.split(self.solution)
        u_previous0, u_previous1 = fenics.split(u_previous)

        self.logger.info("    - Using non-linear solver")

        dt = fenics.Constant(float(self.params.sim_time_step))

        F_m = fenics.inner(mle.compute_stress(sol0, mu, lmbda), mle.compute_strain(v0)) * dx \
              - fenics.inner(mle.compute_stress(v0, mu, lmbda), mle.compute_growth_induced_strain(sol1, coupling, dim)) * dx \
              - fenics.inner(self.body_force, v0) * dx \
              - self.bcs.implement_von_neumann_bc(v0, subspace_id=0)  # integral over ds already included

        F_rd = sol1 * v1 * dx \
               + dt * diff_const * fenics.inner(fenics.grad(sol1), fenics.grad(v1)) * dx \
               - u_previous1 * v1 * dx \
               - dt * mrd.compute_growth_logistic(sol1, prolif_rate, 1.0) * v1 * dx \
               - dt * self.source_term * v1 * dx \
               - dt * self.bcs.implement_von_neumann_bc(diff_const * v1, subspace_id=1)  # integral over ds already included

        F = F_m + F_rd

        J = fenics.derivative(F, self.solution, du)

        problem = fenics.NonlinearVariationalProblem(F, self.solution, bcs=self.bcs.dirichlet_bcs, J=J)
        solver = fenics.NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm.nonlinear_solver = 'snes'
        prm.snes_solver.report = False
        # prm.snes_solver.linear_solver = "lu"
        # prm.snes_solver.maximum_iterations = 20
        # prm.snes_solver.report = True
        # prm.snes_solver.error_on_nonconvergence = False
        # prm.snes_solver.preconditioner = 'amg'
        # prm = solver.parameters.newton_solver  # short form -> Newton Solver
        # prm.absolute_tolerance = 1E-11
        # prm.relative_tolerance = 1E-8
        # prm.maximum_iterations = 1000
        self.solver = solver

    def run_for_adjoint(self, parameters, output_dir=config.output_dir_simulation_tmp):
        """
        Run the time-dependent simulation with minimum number of updated parameters for adjoint optimisation.
        :param parameters: list of parameters
        """
        self.logger.info("-- Updating parameters for solution")
        self.params.diffusion, self.params.proliferation, self.params.coupling = parameters
        #self.params.diffusion, self.params.proliferation = parameters
        self.logger.info("    - 'diffusion_constant' = %.2f" % self.params.diffusion)
        self.logger.info("    - 'proliferation_rate' = %.2f" % self.params.proliferation)
        self.logger.info("    - 'coupling'           = %.2f" % self.params.coupling)
        self.run(keep_nth=1, save_method=None, clear_all=False, plot=False,
                 output_dir=output_dir)
        return self.solution

    def init_postprocess(self, output_dir=config.output_dir_simulation_tmp):
        self.postprocess = PostProcessTumorGrowth(self.results, self.params, output_dir=output_dir)
