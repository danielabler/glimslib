"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation
 - 2D test domain from brain atlas, 4 tissue subdomains + 'outside'
 - spatially heterogeneous parameters
 - no displacement bc between 'outside' and other subdomains
"""

import logging
import os

import test_cases.test_simulation_tumor_growth.testing_config as test_config

from glimslib.simulation.simulation_tumor_growth import TumorGrowth
from glimslib import fenics_local as fenics, config
import glimslib.utils.file_utils as fu
import glimslib.utils.data_io as dio

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fenics.set_log_level(fenics.PROGRESS)

# ==============================================================================
# Load 2D Mesh from IMAGE
# ==============================================================================
path_to_atlas   = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
labelfunction = dio.get_labelfunction_from_image(path_to_atlas, 87)

mesh          = labelfunction.function_space().mesh()

# ==============================================================================
# Problem Settings
# ==============================================================================

class Boundary(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

tissue_id_name_map = {    0: 'outside',
                          1: 'CSF',
                          3: 'WM',
                          2: 'GM',
                          4: 'Ventricles'}

# Boundaries & BCs
boundary = Boundary()
boundary_dict = {'boundary_all': boundary}

dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant((0.0, 0.0)),
                                'subdomain_boundary': 'outside_CSF',
                                'subspace_id': 0},
                'clamped_1': {'bc_value': fenics.Constant((0.0, 0.0)),
                                                'subdomain_boundary': 'outside_WM',
                                                'subspace_id': 0},
                'clamped_2': {'bc_value': fenics.Constant((0.0, 0.0)),
                                                'subdomain_boundary': 'outside_GM',
                                                'subspace_id': 0}
                      }

von_neuman_bcs = {
    'no_flux_boundary': {'bc_value': fenics.Constant(0.0),
                       'named_boundary': 'boundary_all',
                       'subspace_id': 1}
}


# Initial Values
u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1, a=0.5, x0=148, y0=-67)
u_0_disp_expr = fenics.Constant((0.0, 0.0))


# ==============================================================================
# Class instantiation & Setup
# ==============================================================================
sim_time = 10
sim_time_step = 1

sim = TumorGrowth(mesh)

sim.setup_global_parameters(label_function=labelfunction,
                             domain_names=tissue_id_name_map,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

youngmod = {'outside': 1E3,
            'CSF': 1000E-6,
            'WM': 3000E-6,
            'GM': 3000E-6,
            'Ventricles': 1000E-6}

poisson = {'outside': 0.45, #convergence problem if ~0.49
           'CSF': 0.45,
           'WM': 0.45,
           'GM': 0.45,
           'Ventricles': 0.3}

diffusion = {'outside': 0.0,
            'CSF': 0.0,
            'WM': 0.05,
            'GM': 0.01,
            'Ventricles': 0.0}

prolif = {'outside': 0.0,
           'CSF': 0.0,
           'WM': 0.05,
           'GM': 0.05,
           'Ventricles': 0.0}

coupling = {'outside': 0.0,
           'CSF': 0.0,
           'WM': 0.1,
           'GM': 0.1,
           'Ventricles': 0.0}

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
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_atlas')
fu.ensure_dir_exists(output_path)
sim.run(save_method='vtk',plot=True, output_dir=output_path, clear_all=True)

# ==============================================================================
# PostProcess
# ==============================================================================

dio.merge_VTUs(output_path, sim_time_step, sim_time, remove=True, reference=None)

sim.init_postprocess(os.path.join(output_path, 'postprocess', 'plots'))
sim.postprocess.plot_all(deformed=False)
sim.postprocess.plot_all(deformed=True)
