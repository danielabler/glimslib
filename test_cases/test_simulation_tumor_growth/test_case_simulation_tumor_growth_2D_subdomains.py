"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation
 - 2D test domain with 2 subdomains
 - spatially heterogeneous parameters
 - diffusion and proliferation parameters mimick interface between CSF and brain tissue
"""

import logging
import os

import test_cases.test_simulation_tumor_growth.testing_config as test_config

from glimslib.simulation.simulation_tumor_growth import TumorGrowth
from glimslib import fenics_local as fenics
import glimslib.utils.file_utils as fu
import glimslib.utils.data_io as dio

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

# LabelMap
label_funspace = fenics.FunctionSpace(mesh, "DG", 1)
label_expr = fenics.Expression('(x[0]>=0.0) ? (1.0) : (2.0)', degree=1)
labels = fenics.project(label_expr, label_funspace)
tissue_map = {0: 'outside',
              1: 'A',
              2: 'B'}

# Boundaries & BCs
boundary = Boundary()
boundary_dict = {'boundary_all': boundary}
dirichlet_bcs = {'clamped_outside': {'bc_value': fenics.Constant((0.0, 0.0)),
                                    'named_boundary': 'boundary_all',
                                    'subspace_id': 0},
                 # Test to show that Dirichlet BCs can be applied to subdomain interfaces
                 # 'clamped_A_B'    : {'bc_value': fenics.Constant((0.0, 0.0)),
                 #                    'subdomain_boundary': 'A_B',
                 #                    'subspace_id': 0}
                      }
von_neuman_bcs = {  # not necessary to impose zero flux at domain boundary
                  # 'no_flux_boundary': {'bc_value': fenics.Constant(0.0),
                  #                    'named_boundary': 'boundary_all',
                  #                    'subspace_id': 1}
   }

# Initial Values
u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.4 ? (1.0) : (0.0)',
                                        degree=1, x0=2.5, y0=2.5)
u_0_disp_expr = fenics.Constant((0.0, 0.0))

# model parameters
youngmod = {'outside': 10E6,
            'A' : 0.001,
            'B'  : 0.001}
poisson = {'outside': 0.49,
           'A' : 0.40,
           'B' : 0.10}
diffusion = {'outside': 0.0,
           'A' : 0.1,
           'B' : 0.0}
prolif  = {'outside': 0.0,
           'A' : 0.1,
           'B' : 0.0}
coupling = {'outside': 0.0,
           'A' : 0.2,
           'B' : 0.0}

# ==============================================================================
# Class instantiation & Setup
# ==============================================================================
sim_time = 10
sim_time_step = 1

sim = TumorGrowth(mesh)

sim.setup_global_parameters(label_function=labels,
                             domain_names=tissue_map,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}
sim.setup_model_parameters(iv_expression=ivs,
                            diffusion=diffusion,
                            coupling=coupling,
                            proliferation=prolif,
                            E=youngmod,
                            poisson=poisson,
                            sim_time=sim_time, sim_time_step=sim_time_step)

# ==============================================================================
# Run Simulation
# ==============================================================================
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_subdomains')
fu.ensure_dir_exists(output_path)
sim.run(save_method='vtk',plot=True, output_dir=output_path, clear_all=True)


# ==============================================================================
# PostProcess
# ==============================================================================

dio.merge_VTUs(output_path, sim_time_step, sim_time, remove=True, reference=None)

sim.init_postprocess(os.path.join(output_path, 'postprocess', 'plots'))
sim.postprocess.plot_all(deformed=False)
sim.postprocess.plot_all(deformed=True)
