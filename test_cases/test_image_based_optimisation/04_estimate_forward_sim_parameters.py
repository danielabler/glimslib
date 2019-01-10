import logging
import os

import config
config.USE_ADJOINT = True
import test_cases.test_image_based_optimisation.testing_config as test_config

from simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio

output_path = os.path.join(test_config.output_path, '04_parameter_optimization')
fu.ensure_dir_exists(output_path)
# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fenics.set_log_level(fenics.PROGRESS)
# ==============================================================================
# Load Reduced Domain!! -- forward simulation used full domain
# ==============================================================================

path_to_hdf5_mesh = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_2d_reduced_domain.h5')
mesh, subdomains, boundaries = dio.read_mesh_hdf5(path_to_hdf5_mesh)

# ==============================================================================
# Problem Settings
# ==============================================================================

class Boundary(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

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
u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1, a=0.5, x0=148, y0=-67)
u_0_disp_expr = fenics.Constant((0.0, 0.0))

ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}

# ==============================================================================
# Parameters
# ==============================================================================
sim_time =50
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

output_path_4 = os.path.join(test_config.output_path, '04_parameter_optimization_mpi')

# ==============================================================================
# Run Simulation
# ==============================================================================
output_path_forward = os.path.join(output_path_4, 'forward')
# fu.ensure_dir_exists(output_path_forward)
#
# D_GM_target=0.02
# D_WM_target=0.05
# rho_GM_target=0.1
# rho_WM_target=0.1
# coupling_target=0.15
#
# params_target = [D_WM_target, D_GM_target, rho_WM_target, rho_GM_target, coupling_target]
# u_target = sim.run_for_adjoint(params_target, output_dir=output_path_forward)
#
# #===== THIS DOES WORK
# disp_sim_target_0, conc_sim_target_0 = fenics.split(u_target)
# conc_sim_target = sim.functionspace.project_over_space(conc_sim_target_0, subspace_id=1)
# disp_sim_target = sim.functionspace.project_over_space(disp_sim_target_0, subspace_id=0)
# #=====


#===== THIS DOES NOT WORK
# disp_sim_target = sim.results.get_solution_function(subspace_id=0, recording_step=50)
# conc_sim_target = sim.results.get_solution_function(subspace_id=1, recording_step=50)
#=====

# # save functions on reduced domain
path_to_sim_conc = os.path.join(output_path_4, 'concentration_from_simulation.h5')
#dio.save_function_mesh(conc_sim_target, path_to_sim_conc, subdomains=subdomains)

path_to_sim_disp = os.path.join(output_path_4, 'displacement_from_simulation.h5')
#dio.save_function_mesh(disp_sim_target, path_to_sim_disp, subdomains=subdomains)


# ==============================================================================
# Load Target Fields
# ==============================================================================
# output_path_1 = os.path.join(test_config.output_path, '01_forward_simulation')
# output_path_3 = os.path.join(test_config.output_path, '03_estimate_deformation_from_image')
#
# path_to_sim_conc_reduced = os.path.join(output_path_3, 'concentration_from_simulation_reduced.h5')
# #conc_sim, mesh, subdomains, boundaries = dio.load_function_mesh(path_to_sim_conc_reduced, functionspace='function')
# path_to_sim_disp_reduced = os.path.join(output_path_3, 'displacement_from_simulation_reduced.h5')
# #disp_sim, mesh, subdomains, boundaries = dio.load_function_mesh(path_to_sim_disp_reduced, functionspace='vector')
# path_to_est_disp_reduced = os.path.join(output_path_3, 'displacement_from_registration_reduced.h5')
# #disp_est, mesh, subdomains, boundaries = dio.load_function_mesh(path_to_est_disp_reduced, functionspace='vector')
#
# conc_sim = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_sim_conc_reduced)
# disp_sim = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=0),path_to_sim_disp_reduced)

#
#
# fenics.errornorm(disp_sim_target, disp_sim)
# fenics.errornorm(conc_sim_target, conc_sim)
#
# diff_conc = sim.functionspace.project_over_space(conc_sim-conc_sim_target, subspace_id=1)
# fenics.assemble(diff_conc*fenics.dx)

# ==============================================================================
# OPTIMISATION
# ==============================================================================

output_path_adjoint = os.path.join(output_path_4, 'adjoint')

D_GM_init=0.01
D_WM_init=0.01
rho_GM_init=0.01
rho_WM_init=0.01
coupling_init=0.2

params_init = [D_WM_init, D_GM_init, rho_WM_init, rho_GM_init, coupling_init]
params_init = [fenics.Constant(param) for param in params_init]

u = sim.run_for_adjoint(params_init, output_dir=output_path_adjoint)

# optimization
# conc_sim_proj = sim.functionspace.project_over_space(conc_sim, subspace_id=1, annotate=False)
# disp_sim_proj = sim.functionspace.project_over_space(disp_sim, subspace_id=0, annotate=False)

# plott.show_img_seg_f(function=conc_sim_proj, show=True,
#                      path=os.path.join(output_path_4, 'conc_target.png'),
#                      dpi=300)
# plott.show_img_seg_f(function=disp_sim_proj, show=True,
#                      path=os.path.join(output_path_4, 'disp_target.png'),
#                      dpi=300)
#
#


disp_opt, conc_opt = fenics.split(u)
conc_opt_proj = sim.functionspace.project_over_space(conc_opt, subspace_id=1)
disp_opt_proj = sim.functionspace.project_over_space(disp_opt, subspace_id=0)

#J = fenics.Functional( #fenics.inner(conc_opt - conc_sim, conc_opt - conc_sim)*sim.subdomains.dx
#                      fenics.inner(disp_opt - disp_sim, disp_opt - disp_sim)*sim.subdomains.dx)

conc_sim_target = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_sim_conc)
disp_sim_target = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=0),path_to_sim_disp)

fenics.errornorm(conc_opt_proj, conc_sim_target)
fenics.errornorm(disp_opt_proj, disp_sim_target)


J = fenics.Functional(  fenics.inner(conc_opt_proj - conc_sim_target, conc_opt_proj - conc_sim_target)*sim.subdomains.dx
                      + fenics.inner(disp_opt_proj - disp_sim_target, disp_opt_proj - disp_sim_target)*sim.subdomains.dx)

# import visualisation.plotting as plott
# plott.show_img_seg_f(function=conc_sim_proj, show=True,
#                      path=os.path.join(output_path_4, 'conc_target.png'),
#                      dpi=300)
# plott.show_img_seg_f(function=disp_sim_proj, show=True,
#                      path=os.path.join(output_path_4, 'disp_target.png'),
#                      dpi=300)
#
# m_target, w_target = fenics.split(u_target)
# m, w               = fenics.split(u)
#
#
#
# J = fenics.Functional(  fenics.inner(m - m_target, m - m_target) * sim.subdomains.dx  +   # displacements
#                         fenics.inner(w - w_target, w - w_target) * sim.subdomains.dx
# )


controls = [fenics.ConstantControl(param) for param in params_init]



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


# rf_np = fenics.ReducedFunctionalNumPy(rf)
#
# m = [p.data() for p in rf_np.controls]
# m_global = rf_np.obj_to_array(m)
# J = rf_np.__call__
# dJ = lambda m: rf_np.derivative(m, forget=False, project=False) #make dJ a function
# H = rf_np.hessian
#
#
# ps = [m_global]
# def reporter(p):
#     """Reporter function to capture intermediate states of optimization."""
#     global ps
#     print(p)
#     print(ps)
#     ps.append(p)
#
#
# from scipy.optimize import minimize as scipy_minimize
#
# method = 'L-BFGS-B'
#
# res = scipy_minimize(J, m_global, method=method, callback=reporter)
#
# bounds = [(0.005, 0.2), (0.005, 0.2),
#           (0.005, 0.2), (0.005, 0.2),
#           (0.005, 0.2)]
bounds = [[0.005, 0.005, 0.005, 0.005, 0.005],
          [0.2, 0.2, 0.2, 0.2, 0.2]]
m_opt = fenics.minimize(rf, bounds=bounds)


for var in m_opt:
    print(var.values())

opt_df = create_opt_progress_df(opt_param_progress_post, opt_dj_progress_post,
                                ['D_WM', 'D_GM', 'rho_WM', 'rho_GM', 'coupling'])

print(opt_df)
opt_df.to_excel(os.path.join(output_path_4, 'optimization.xls'))


# ==============================================================================
# Reload and Plot
# ==============================================================================

# path_to_h5_file = os.path.join(output_path, 'solution_timeseries.h5')
# sim.reload_from_hdf5(path_to_h5_file)
#
# sim.init_postprocess(output_path)
# sim.postprocess.save_all(save_method='vtk', clear_all=False, selection=slice(1,-1,1))
#
# selection = slice(1, -1, 1)
# sim.postprocess.plot_all(deformed=False, selection=selection, output_dir=os.path.join(output_path, 'postprocess_reloaded', 'plots'))
# sim.postprocess.plot_all(deformed=True, selection=selection, output_dir=os.path.join(output_path, 'postprocess_reloaded', 'plots'))
#

