"""
Run forward simulation on 2D atlas
"""
import logging
import os

import test_cases.test_image_based_optimisation.testing_config as test_config

from simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fenics.set_log_level(fenics.PROGRESS)
# ==============================================================================
# Load 2D Mesh from IMAGE
# ==============================================================================

labelfunction, mesh, subdomains, boundaries = dio.load_function_mesh(test_config.path_to_2d_labelfunction)
# verify that loaded correctly
# plott.show_img_seg_f(function=function, show=True,
#                      path=os.path.join(config.output_path, 'image_label_from_fenics_function_2.png'))

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

von_neuman_bcs = {}


# Initial Values
u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1, a=0.5, x0=148, y0=-67)
u_0_disp_expr = fenics.Constant((0.0, 0.0))

ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}

# ==============================================================================
# Parameters
# ==============================================================================
sim_time =200
sim_time_step = 1

E_GM=3000E-6
E_WM=3000E-6
E_CSF=1000E-6
E_VENT=1000E-6
nu_GM=0.45
nu_WM=0.45
nu_CSF=0.45
nu_VENT=0.3
D_GM=0.02
D_WM=0.1
rho_GM=0.1
rho_WM=0.1
coupling = 0.15

# ==============================================================================
# TumorGrowthBrain
# ==============================================================================

sim = TumorGrowthBrain(mesh)

sim.setup_global_parameters(label_function=labelfunction,
                             domain_names=tissue_id_name_map,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

sim.setup_model_parameters(iv_expression=ivs,
                           sim_time=sim_time, sim_time_step=sim_time_step,
                           E_GM=E_GM, E_WM=E_WM, E_CSF=E_CSF, E_VENT=E_VENT,
                           nu_GM=nu_GM, nu_WM=nu_WM, nu_CSF=nu_CSF, nu_VENT=nu_VENT,
                           D_GM=D_GM, D_WM=D_WM,
                           rho_GM=rho_GM, rho_WM=rho_WM,
                           coupling=coupling)

output_path = os.path.join(test_config.output_path, '01_forward_simulation')
fu.ensure_dir_exists(output_path)
#sim.run(save_method=None ,plot=False, output_dir=output_path, clear_all=True)


# ==============================================================================
# Reload and Plot
# ==============================================================================

path_to_h5_file = os.path.join(output_path, 'solution_timeseries.h5')
sim.reload_from_hdf5(path_to_h5_file)

sim.init_postprocess(output_path)
sim.postprocess.save_all(save_method='vtk', clear_all=False, selection=slice(1,-1,1))

selection = slice(1, -1, 1)
sim.postprocess.plot_all(deformed=False, selection=selection, output_dir=os.path.join(output_path, 'postprocess_reloaded', 'plots'))
sim.postprocess.plot_all(deformed=True, selection=selection, output_dir=os.path.join(output_path, 'postprocess_reloaded', 'plots'))

# ==============================================================================
# Save concentration and deformation fields at last time step
# ==============================================================================
rec_steps = sim.results.get_recording_steps()
conc = sim.postprocess.get_solution_concentration(rec_steps[-1])
disp = sim.postprocess.get_solution_displacement(rec_steps[-1])

path_to_conc = os.path.join(output_path, 'concentration_simulated.h5')
dio.save_function_mesh(conc, path_to_conc, labelfunction=None, subdomains=sim.subdomains.subdomains)

path_to_disp = os.path.join(output_path, 'displacement_simulated.h5')
dio.save_function_mesh(disp, path_to_disp, labelfunction=None, subdomains=sim.subdomains.subdomains)