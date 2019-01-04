import fenics_local as fenics
from numpy import zeros
from simulation.simulation_base import FenicsSimulation
import simulation.helpers.math_linear_elasticity as mle
import simulation.helpers.math_reaction_diffusion as mrd



class TumorGrowthBrain(FenicsSimulation):

    def _setup_functionspace(self):
        displacement_element = fenics.VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
        concentration_element = fenics.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        element = fenics.MixedElement([displacement_element, concentration_element])
        subspace_names = {0: 'displacement', 1: 'concentration'}
        self.functionspace.init_function_space(element, subspace_names)

    def _define_model_params(self):
        self.required_params =['E_GM', 'E_WM', 'E_CSF', 'E_VENT',
                               'nu_GM', 'nu_WM', 'nu_CSF', 'nu_VENT',
                               'D_GM', 'D_WM',
                               'rho_GM', 'rho_WM',
                               'coupling']
        self.optional_params = []

    def _setup_problem(self, u_previous):
        dx = self.subdomains.dx
        ds = self.subdomains.ds
        dsn = self.subdomains.dsn
        # Parameters
        mu_GM = mle.compute_mu(self.params.E_GM, self.params.nu_GM)
        lmbda_GM = mle.compute_lambda(self.params.E_GM, self.params.nu_GM)
        mu_WM = mle.compute_mu(self.params.E_WM, self.params.nu_WM)
        lmbda_WM = mle.compute_lambda(self.params.E_WM, self.params.nu_WM)
        mu_CSF = mle.compute_mu(self.params.E_CSF, self.params.nu_CSF)
        lmbda_CSF = mle.compute_lambda(self.params.E_CSF, self.params.nu_CSF)
        mu_VENT = mle.compute_mu(self.params.E_VENT, self.params.nu_VENT)
        lmbda_VENT = mle.compute_lambda(self.params.E_VENT, self.params.nu_VENT)
        mu_OUT = mle.compute_mu(10E6, 0.45)
        lmbda_OUT = mle.compute_lambda(10E6, 0.45)

        # The following terms are added in governing form testing.
        # They are not strictly part of the problem but need to be defined if not present!
        if not hasattr(self, 'body_force'):
            self.body_force = fenics.Constant(zeros(self.geometric_dimension))
        if not hasattr(self, 'rd_source_term'):
            self.rd_source_term = fenics.Constant(0)

        du = fenics.TrialFunction(self.functionspace.function_space)
        v0, v1 = fenics.TestFunctions(self.functionspace.function_space)
        self.solution = fenics.Function(self.functionspace.function_space)
        self.solution.label = 'solution_function'

        sol0, sol1 = fenics.split(self.solution)
        u_previous0, u_previous1 = fenics.split(u_previous)

        # Implement von Neuman Boundary Conditions
        #von_neuman_bc_terms = self._implement_von_neumann_bcs([v0, v1])
        #von_neuman_bc_term_mech, von_neuman_bc_term_rd = von_neuman_bc_terms

        # subspace 0 -> displacements
        # subspace 1 -> concentration

        dx_outside = dx(self.subdomains.get_subdomain_id('outside'))
        dx_CSF = dx(self.subdomains.get_subdomain_id('CSF'))
        dx_WM = dx(self.subdomains.get_subdomain_id('WM'))
        dx_GM = dx(self.subdomains.get_subdomain_id('GM'))
        dx_Ventricles = dx(self.subdomains.get_subdomain_id('Ventricles'))

        dt = fenics.Constant(float(self.params.sim_time_step))
        d = self.solution.geometric_dimension()
        F_m = fenics.inner(mle.compute_stress(sol0, mu_OUT, lmbda_OUT), mle.compute_strain(v0)) * dx_outside \
              - fenics.inner(mle.compute_stress(v0, mu_OUT, lmbda_OUT), mrd.compute_expansion(sol1, self.params.coupling, d)) * dx_outside \
              + fenics.inner(mle.compute_stress(sol0, mu_CSF, lmbda_CSF), mle.compute_strain(v0)) * dx_CSF \
              - fenics.inner(mle.compute_stress(v0, mu_CSF, lmbda_CSF), mrd.compute_expansion(sol1, self.params.coupling, d)) * dx_CSF \
              + fenics.inner(mle.compute_stress(sol0, mu_WM, lmbda_WM), mle.compute_strain(v0)) * dx_WM \
              - fenics.inner(mle.compute_stress(v0, mu_WM, lmbda_WM), mrd.compute_expansion(sol1, self.params.coupling, d)) * dx_WM \
              + fenics.inner(mle.compute_stress(sol0, mu_GM, lmbda_GM), mle.compute_strain(v0)) * dx_GM \
              - fenics.inner(mle.compute_stress(v0, mu_GM, lmbda_GM), mrd.compute_expansion(sol1, self.params.coupling, d)) * dx_GM \
              + fenics.inner(mle.compute_stress(sol0, mu_VENT, lmbda_VENT), mle.compute_strain(v0)) * dx_Ventricles \
              - fenics.inner(mle.compute_stress(v0, mu_VENT, lmbda_VENT), mrd.compute_expansion(sol1, self.params.coupling, d)) * dx_Ventricles \
              #- self._implement_von_neumann_bcs_subspace(v0, 0)

        F_rd = sol1 * v1 * dx \
               + dt * fenics.Constant(0) * fenics.inner(fenics.grad(sol1), fenics.grad(v1)) * dx_outside \
               + dt * fenics.Constant(0) * fenics.inner(fenics.grad(sol1), fenics.grad(v1)) * dx_CSF \
               + dt * self.params.D_WM * fenics.inner(fenics.grad(sol1), fenics.grad(v1)) * dx_WM \
               + dt * self.params.D_GM * fenics.inner(fenics.grad(sol1), fenics.grad(v1)) * dx_GM \
               + dt * fenics.Constant(0) * fenics.inner(fenics.grad(sol1), fenics.grad(v1)) * dx_Ventricles \
               - u_previous1 * v1 * dx \
               - dt * fenics.Constant(0) * v1 * dx_outside \
               - dt * fenics.Constant(0) * v1 * dx_CSF \
               - dt * mrd.compute_growth_logistic(sol1, self.params.rho_WM, 1.0) * v1 * dx_WM \
               - dt * mrd.compute_growth_logistic(sol1, self.params.rho_GM, 1.0) * v1 * dx_GM \
               - dt * fenics.Constant(0) * v1 * dx_Ventricles \
               - dt * self.rd_source_term * v1 * dx \
               #- self._implement_von_neumann_bcs_subspace(dt * self.diff_const * v1, 1)

        # TODO How to ensure that von neumann BC is applied to correct boundaries?
        # TODO How to enforce zero flux across boundaries?
        F = F_m + F_rd

        J = fenics.derivative(F, self.solution, du)

        problem = fenics.NonlinearVariationalProblem(F, self.solution, bcs=self.bcs.dirichlet_bcs, J=J)
        solver = fenics.NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm.nonlinear_solver = 'snes'
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

    def run_for_adjoint(self, parameters, save_method=None):
        """
        Run the time-dependent simulation with minimum number of updated parameters for adjoint optimisation.
        :param parameters: list of parameters
        """
        self.logger.info("-- Updating parameters for solution")
        self.params.D_WM = parameters[0]
        self.logger.info("    - 'diffusion_constant WM' = %.2f" % self.params.D_WM)
        self.params.D_GM = parameters[1]
        self.logger.info("    - 'diffusion_constant GM' = %.2f" % self.params.D_GM)
        self.params.rho_WM = parameters[2]
        self.logger.info("    - 'proliferation_rate WM' = %.2f" % self.params.rho_WM)
        self.params.rho_GM = parameters[3]
        self.logger.info("    - 'proliferation_rate GM' = %.2f" % self.params.rho_GM)
        self.params.coupling = parameters[4]
        self.logger.info("    - 'coupling'              = %.2f" % self.params.coupling)
        self.run(save_method=save_method)
        return self.solution

    def run_for_adjoint2(self, parameters, save_method=None):
        """
        Run the time-dependent simulation with minimum number of updated parameters for adjoint optimisation.
        :param parameters: list of parameters
        """
        self.logger.info("-- Updating parameters for solution")
        self.params.D_WM = parameters[0]
        self.logger.info("    - 'diffusion_constant WM' = %.2f" % self.params.D_WM)
        self.params.D_GM = parameters[1]
        self.logger.info("    - 'diffusion_constant GM' = %.2f" % self.params.D_GM)
        self.params.rho_WM  = parameters[2]
        self.params.rho_GM = parameters[2]
        self.logger.info("    - 'proliferation_rate GM, WM' = %.2f" % self.params.rho_GM)
        self.params.coupling = parameters[3]
        self.logger.info("    - 'coupling'              = %.2f" % self.params.coupling)
        self.run(save_method=save_method)
        return self.solution

    def run_for_adjoint3(self, parameters, save_method=None):
        """
        Run the time-dependent simulation with minimum number of updated parameters for adjoint optimisation.
        :param parameters: list of parameters
        """
        self.logger.info("-- Updating parameters for solution")
        self.params.D_GM = parameters[0]
        self.params.D_WM = parameters[0]
        self.logger.info("    - 'diffusion_constant WM, GM' = %.2f" % self.params.D_GM)
        self.params.rho_WM  = parameters[1]
        self.params.rho_GM = parameters[1]
        self.logger.info("    - 'proliferation_rate GM, WM' = %.2f" % self.params.rho_GM)
        self.params.coupling = parameters[2]
        self.logger.info("    - 'coupling'              = %.2f" % self.params.coupling)
        self.run(save_method=save_method)
        return self.solution

    def run_for_adjoint4(self, parameters, save_method=None):
        """
        Run the time-dependent simulation with minimum number of updated parameters for adjoint optimisation.
        :param parameters: list of parameters
        """
        self.logger.info("-- Updating parameters for solution")
        self.params.coupling = parameters[0]
        self.logger.info("    - 'coupling'              = %.2f" % self.params.coupling)
        self.run(save_method=save_method)
        return self.solution

    def run_for_adjoint5(self, parameters, save_method=None):
        """
        Run the time-dependent simulation with minimum number of updated parameters for adjoint optimisation.
        :param parameters: list of parameters
        """
        self.logger.info("-- Updating parameters for solution")
        self.params.D_GM = parameters[0]
        self.params.D_WM = parameters[0]
        self.params.logger.info("    - 'diffusion_constant WM, GM' = %.2f" % self.params.D_GM)
        self.params.rho_WM  = parameters[1]
        self.params.rho_GM = parameters[1]
        self.logger.info("    - 'proliferation_rate GM, WM' = %.2f" % self.params.rho_GM)
        self.run(save_method=save_method)
        return self.solution