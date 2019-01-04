"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation
 - 2D test domain (uniform)
 - spatially uniform parameters
"""

import logging
import os

import test_cases.test_simulation_tumor_growth.testing_config as test_config

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

# Mesh
nx = ny = 50
mesh = fenics.RectangleMesh(fenics.Point(-5, -5), fenics.Point(5, 5), nx, ny)


# Boundaries & BCs
boundary = Boundary()
boundary_dict = {'boundary_all': boundary}
dirichlet_bcs = {'clamped_boundary': {'bc_value': fenics.Constant((0.0, 0.0)),
                                    'named_boundary': 'boundary_all',
                                    'subspace_id': 0}
                      }

# no flux boundary BC not necessary
von_neuman_bcs = {
                    # 'no_flux_through_boundary': {'bc_value': fenics.Constant(0.0),
                    #                  'named_boundary': 'boundary_all',
                    #                  'subspace_id': 1},
                 }

# Initial Values
# u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.1 ? (1.0) : (0.0)',
#                                         degree=1, x0=0.0, y0=0.0)
u_0_conc_expr = fenics.Expression( ('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))'), degree=1, a=1, x0=0.0, y0=0.0)

u_0_disp_expr = fenics.Constant((0.0, 0.0))

# ==============================================================================
# Class instantiation & Setup
# ==============================================================================
sim_time = 20
sim_time_step = 1

sim = TumorGrowth(mesh)

sim.setup_global_parameters( boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs   )

ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}
sim.setup_model_parameters(iv_expression=ivs,
                            diffusion=0.1,
                            coupling=1,
                            proliferation=0.1,
                            E=0.001,
                            poisson=0.45,
                            sim_time=sim_time, sim_time_step=sim_time_step)

# ==============================================================================
# Run Simulation
# ==============================================================================
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_uniform_mpi')
fu.ensure_dir_exists(output_path)
sim.run(save_method='xdmf',plot=False, output_dir=output_path, clear_all=False)



