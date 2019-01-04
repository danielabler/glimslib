"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation & adjoint parameter estimation
 - 2D test domain
 - spatially homogeneous parameters
"""

import logging
import os

import test_cases.test_simulation_tumor_growth.testing_config as test_config
import config
config.USE_ADJOINT=True
from simulation.simulation_tumor_growth import TumorGrowth
import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fenics.set_log_level(fenics.PROGRESS)

# ==============================================================================
# Problem Settings
# ==============================================================================

class Boundary(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

nx = ny = nz = 50
mesh = fenics.RectangleMesh(fenics.Point(-5, -5), fenics.Point(5, 5), nx, ny)

boundary = Boundary()
boundary_dict = {'boundary_all': boundary}
dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant((0.0, 0.0)),
                                    'named_boundary': 'boundary_all',
                                    'subspace_id': 0}
                      }
von_neuman_bcs = {}

u_0_conc_expr = fenics.Expression( ('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))'), degree=1, a=1, x0=0.0, y0=0.0)
u_0_disp_expr = fenics.Expression(('0','0'), degree=1)


# ==============================================================================
# Class instantiation & Setup
# ==============================================================================
sim_time = 5
sim_time_step = 1

sim = TumorGrowth(mesh)

sim.setup_global_parameters( boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )


ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}
sim.setup_model_parameters(iv_expression=ivs,
                            diffusion=0.1,
                            coupling=1,
                            proliferation=0.1,
                            E=0.001,
                            poisson=0.4,
                            sim_time=sim_time, sim_time_step=sim_time_step)

# ==============================================================================
# Run Simulation
# ==============================================================================
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_uniform_adjoint')
fu.ensure_dir_exists(output_path)


D_target    = 0.05
rho_target  = 0.05
c_target    = 0.1

u_target = sim.run_for_adjoint([D_target, rho_target, c_target], output_dir=output_path)

# ==============================================================================
# OPTIMISATION
# ==============================================================================

D   = fenics.Constant(0.4)
rho = fenics.Constant(0.4)
c   = fenics.Constant(0.4)

u = sim.run_for_adjoint([D, rho, c], output_dir=output_path)

J = fenics.Functional( fenics.inner(u-u_target, u-u_target)*sim.subdomains.dx)
controls = [fenics.ConstantControl(D), fenics.ConstantControl(rho), fenics.ConstantControl(c)]

def eval_cb(j, a):
    D    = a[0].values()
    rho  = a[1].values()
    c    = a[2].values()
    print(j, D, rho, c)

reduced_functional = fenics.ReducedFunctional(J, controls, eval_cb_post=eval_cb)

m_opt = fenics.minimize(reduced_functional)

# ==============================================================================
# RESULTS
# ==============================================================================
# Plot when adjoint computation has finished to avoid recording of function projections
sim.plotting.plot_all(sim_time)

output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_uniform_adjoint')
fu.ensure_dir_exists(output_path)

sim.init_postprocess(os.path.join(output_path, 'postprocess', 'plots'))

sim.postprocess.plot_all(deformed=False)
sim.postprocess.plot_all(deformed=True)

for var in m_opt:
    print(var.values())