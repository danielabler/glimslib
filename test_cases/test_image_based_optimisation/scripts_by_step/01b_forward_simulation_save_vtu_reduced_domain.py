"""
Reload forward simulation, save vtu
"""
import logging
import os

import test_cases.test_image_based_optimisation.testing_config as test_config

from glimslib.simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from glimslib.simulation_helpers import Boundary

from glimslib import fenics_local as fenics
import glimslib.utils.file_utils as fu
import glimslib.utils.data_io as dio

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.PROGRESS)
else:
    fenics.set_log_level(fenics.LogLevel.PROGRESS)
## ==============================================================================
# Load Reduced Domain!! -- forward simulation used full domain
# ==============================================================================

path_to_hdf5_mesh = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_2d_reduced_domain.h5')
mesh, subdomains, boundaries = dio.read_mesh_hdf5(path_to_hdf5_mesh)

# ==============================================================================
# Problem Settings
# ==============================================================================

tissue_id_name_map = {    1: 'CSF',
                          3: 'WM',
                          2: 'GM',
                          4: 'Ventricles'}

boundary = Boundary()
boundary_dict = {'boundary_all': boundary}

dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant((0.0, 0.0)),
                                'named_boundary': 'boundary_all',
                                'subspace_id': 0}
                      }

von_neuman_bcs = {}

# Initial Values
u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1, a=0.5,
                                  x0=test_config.seed_position[0], y0=test_config.seed_position[1])
u_0_disp_expr = fenics.Constant((0.0, 0.0))

ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}

# ==============================================================================
# Parameters
# ==============================================================================
sim_time = test_config.params_sim["sim_time"]
sim_time_step = test_config.params_sim["sim_time_step"]

E_GM = test_config.params_fix["E_GM"]
E_WM = test_config.params_fix["E_WM"]
E_CSF = test_config.params_fix["E_CSF"]
E_VENT = test_config.params_fix["E_VENT"]
nu_GM = test_config.params_fix["nu_GM"]
nu_WM = test_config.params_fix["nu_WM"]
nu_CSF = test_config.params_fix["nu_CSF"]
nu_VENT = test_config.params_fix["nu_VENT"]

D_WM = test_config.params_target["D_WM"]
D_GM = test_config.params_target["D_GM"]
rho_WM = test_config.params_target["rho_WM"]
rho_GM = test_config.params_target["rho_GM"]
coupling = test_config.params_target["coupling"]

# ==============================================================================
# TumorGrowthBrain
# ==============================================================================

sim = TumorGrowthBrain(mesh)

sim.setup_global_parameters(subdomains=subdomains,
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

output_path = test_config.path_01_forward_simulation_red
fu.ensure_dir_exists(output_path)


# ==============================================================================
# Reload
# ==============================================================================
path_to_h5_file = os.path.join(output_path, 'solution_timeseries.h5')
sim.reload_from_hdf5(path_to_h5_file)

# ==============================================================================
# Save concentration and deformation fields at last time step
# ==============================================================================
rec_steps = sim.results.get_recording_steps()

conc = sim.results.get_solution_function(subspace_name='concentration',recording_step=rec_steps[-1])
disp = sim.results.get_solution_function(subspace_name='displacement',recording_step=rec_steps[-1])

path_to_conc = os.path.join(output_path, 'concentration_simulated.h5')
dio.save_function_mesh(conc, path_to_conc, labelfunction=None, subdomains=sim.subdomains.subdomains)

path_to_disp = os.path.join(output_path, 'displacement_simulated.h5')
dio.save_function_mesh(disp, path_to_disp, labelfunction=None, subdomains=sim.subdomains.subdomains)

# ==============================================================================
# Save as VTK & Plot
# ==============================================================================
sim.init_postprocess(output_path)
selection = [1, test_config.params_sim['sim_time']]
sim.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)

if fenics.is_version("<2018.1.x"):
    sim.postprocess.plot_all(deformed=False, selection=selection, output_dir=os.path.join(output_path, 'postprocess_reloaded', 'plots'))
    sim.postprocess.plot_all(deformed=True, selection=selection, output_dir=os.path.join(output_path, 'postprocess_reloaded', 'plots'))
    sim.postprocess.plot_for_pub(deformed=True, selection=selection, output_dir=os.path.join(output_path, 'postprocess_reloaded', 'plots_for_pub'))


