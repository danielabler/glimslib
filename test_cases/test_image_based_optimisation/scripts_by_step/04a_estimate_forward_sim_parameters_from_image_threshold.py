import logging
import os

config.USE_ADJOINT = True
import test_cases.test_image_based_optimisation.testing_config as test_config

from glimslib.simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from glimslib.simulation_helpers import Boundary
from glimslib import fenics_local as fenics
import glimslib.utils.file_utils as fu
import glimslib.utils.data_io as dio

output_path = test_config.path_04_optimization_from_image
fu.ensure_dir_exists(output_path)
# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.WARNING)
else:
    fenics.set_log_level(fenics.LogLevel.WARNING)
# ==============================================================================
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

D_WM = test_config.params_init["D_WM"]
D_GM = test_config.params_init["D_GM"]
rho_WM = test_config.params_init["rho_WM"]
rho_GM = test_config.params_init["rho_GM"]
coupling = test_config.params_init["coupling"]

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
# Load Target Fields
# ==============================================================================
output_path_3 = test_config.path_03_registration

# path_to_sim_conc_reduced = os.path.join(output_path_3, 'concentration_from_simulation_reduced.h5')
# conc_sim = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_sim_conc_reduced)

path_to_sim_disp_reduced = os.path.join(output_path_3, 'displacement_from_registration_reduced.h5')
disp_sim = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=0),path_to_sim_disp_reduced)


# # thresholded but from SIM ... not projected in reduced domain!!!
# path_to_conc_thr_reduced_020 = os.path.join(output_path_3, 'thresholded_concentration_simulation_020.h5')
# conc_thr_020 = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_conc_thr_reduced_020)
#
# path_to_conc_thr_reduced_080 = os.path.join(output_path_3, 'thresholded_concentration_simulation_080.h5')
# conc_thr_080 = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_conc_thr_reduced_080)

# thresholded from SIM in reduced domain
path_to_conc_thr_reduced_020 = os.path.join(output_path_3, 'thresholded_concentration_simulation_reduced_020.h5')
conc_thr_020 = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_conc_thr_reduced_020)

path_to_conc_thr_reduced_080 = os.path.join(output_path_3, 'thresholded_concentration_simulation_reduced_080.h5')
conc_thr_080 = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_conc_thr_reduced_080)

# ==============================================================================
# OPTIMISATION
# ==============================================================================

output_path_adjoint = os.path.join(output_path, 'adjoint')

D_WM_init = test_config.params_init["D_WM"]
D_GM_init = test_config.params_init["D_GM"]
rho_WM_init = test_config.params_init["rho_WM"]
rho_GM_init = test_config.params_init["rho_GM"]
coupling_init = test_config.params_init["coupling"]

params_init = [D_WM_init, D_GM_init, rho_WM_init, rho_GM_init, coupling_init]
params_init = [fenics.Constant(param) for param in params_init]

u = sim.run_for_adjoint(params_init, output_dir=output_path_adjoint)

disp_opt, conc_opt = fenics.split(u)
#conc_opt_proj = sim.functionspace.project_over_space(conc_opt, subspace_id=1)
disp_opt_proj = sim.functionspace.project_over_space(disp_opt, subspace_id=0)

def thresh(f, thresh):
    smooth_f = 0.01
    f_thresh = 0.5 * (fenics.tanh((f - thresh) / smooth_f) + 1)
    return f_thresh

conc_opt_proj_020 = sim.functionspace.project_over_space(thresh(conc_opt, 0.2), subspace_id=1)
conc_opt_proj_080 = sim.functionspace.project_over_space(thresh(conc_opt, 0.8), subspace_id=1)


function_expr =   fenics.inner(conc_opt_proj_020 - conc_thr_020, conc_opt_proj_020 - conc_thr_020) * sim.subdomains.dx \
                + fenics.inner(conc_opt_proj_080 - conc_thr_080, conc_opt_proj_080 - conc_thr_080) * sim.subdomains.dx \
                + fenics.inner(disp_opt_proj - disp_sim, disp_opt_proj - disp_sim) * sim.subdomains.dx

if fenics.is_version("<2018.1.x"):
    J = fenics.Functional(function_expr)
else:
    J = fenics.assemble(function_expr)

controls = [fenics.Control(param) for param in params_init]

import pandas as pd

opt_param_progress_post = []
opt_dj_progress_post = []

def eval_cb_post(j, a):
    values = [param.values()[0] for param in a]
    result = (j, *values)
    global opt_param_progress_post
    opt_param_progress_post.append(result)
    print(result)

def derivative_cb_post(j, dj, m):
    param_values = [param.values()[0] for param in m]
    dj_values = [param.values()[0] for param in dj]
    global opt_dj_progress_post
    result = (j, *dj_values)
    opt_dj_progress_post.append(result)
    print(result)


def create_opt_progress_df(opt_param_list, opt_dj_list, param_name_list):
    columns_params = ['J', *param_name_list]
    columns_dJ = ['J', *['dJd%s'%param for param in param_name_list]]
    # create data frames
    params_df = pd.DataFrame(opt_param_list)
    params_df.columns = columns_params
    dj_df = pd.DataFrame(opt_dj_list)
    dj_df.columns = columns_dJ
    # merge
    opt_df = pd.merge(params_df, dj_df, on='J', how='outer')
    return opt_df

rf = fenics.ReducedFunctional(J, controls, eval_cb_post=eval_cb_post, derivative_cb_post=derivative_cb_post)

bounds = [[0.005, 0.005, 0.005, 0.005, 0.005],
          [0.5, 0.5, 0.5, 0.5, 0.5]]

print("== Start Optimization")
m_opt = fenics.minimize(rf, bounds=bounds,
                        options={'disp': True, 'gtol': 1e-6}, tol=1e-6)

params_dict = {}
for var, name in zip(m_opt,['D_WM', 'D_GM', 'rho_WM', 'rho_GM', 'coupling']):
    print(var.values())
    params_dict[name] = var.values()[0]

import pickle
path_to_params_dict = os.path.join(output_path, 'opt_params.pkl')
with open(path_to_params_dict, 'wb') as handle:
    pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

opt_df = create_opt_progress_df(opt_param_progress_post, opt_dj_progress_post,
                                ['D_WM', 'D_GM', 'rho_WM', 'rho_GM', 'coupling'])

print(opt_df)
opt_df.to_excel(os.path.join(output_path, 'optimization.xls'))

if fenics.is_version(">2017.2.x"):
    sim.tape.visualise()