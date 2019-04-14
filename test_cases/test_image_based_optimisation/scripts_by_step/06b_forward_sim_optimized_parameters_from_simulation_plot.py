import logging
import os

import config
config.USE_ADJOINT = True
import test_cases.test_image_based_optimisation.testing_config as test_config

from simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from simulation.helpers.helper_classes import Boundary

import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio
import visualisation.plotting as plott

output_path = test_config.path_05_forward_simulation_optimized_from_sim
output_path_4 = test_config.path_04_optimization_from_sim
# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.PROGRESS)
else:
    fenics.set_log_level(fenics.LogLevel.PROGRESS)
# ==============================================================================
# Load Domain as in forward simulation
# ==============================================================================

labelfunction, mesh, subdomains, boundaries = dio.load_function_mesh(test_config.path_to_2d_labelfunction)
# verify that loaded correctly
# plott.show_img_seg_f(function=function, show=True,
#                      path=os.path.join(config.output_path, 'image_label_from_fenics_function_2.png'))

# ==============================================================================
# Problem Settings
# ==============================================================================

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

import pickle
path_to_params_dict = os.path.join(output_path_4, 'opt_params.pkl')

if os.path.exists(path_to_params_dict):
    opt_params_dict = pickle.load(open(path_to_params_dict, "rb"))
else:
    raise FileNotFoundError("File %s does not exist"%(path_to_params_dict))

for key, value in opt_params_dict.items():
    print(key, value)

D_WM = opt_params_dict["D_WM"]
D_GM = opt_params_dict["D_GM"]
rho_WM = opt_params_dict["rho_WM"]
rho_GM = opt_params_dict["rho_GM"]
coupling = opt_params_dict["coupling"]

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

