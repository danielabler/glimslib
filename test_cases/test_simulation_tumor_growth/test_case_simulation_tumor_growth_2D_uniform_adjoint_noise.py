"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation & adjoint parameter estimation
 - 2D test domain
 - spatially homogeneous parameters
"""

import logging
import os

import test_cases.test_simulation_tumor_growth.testing_config as test_config

config.USE_ADJOINT=True
from glimslib.simulation.simulation_tumor_growth import TumorGrowth
from glimslib import fenics_local as fenics, visualisation as plott
import glimslib.utils.file_utils as fu

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
sim_time = 10
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
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_uniform_adjoint_noise')
fu.ensure_dir_exists(output_path)


D_target    = 0.05
rho_target  = 0.05
c_target    = 0.1

u_target = sim.run_for_adjoint([D_target, rho_target, c_target], output_dir=output_path)

# ==============================================================================
# OPTIMISATION
# ==============================================================================

D   = fenics.Constant(0.1)
rho = fenics.Constant(0.01)
c   = fenics.Constant(0.05)

u = sim.run_for_adjoint([D, rho, c], output_dir=output_path)




import numpy as np
def add_noise(function, noise_level):
    noise = fenics.Function(function.function_space())
    noise.vector()[:] = noise_level * np.random.randn(len(function.function_space().dofmap().dofs()))
    fun_with_noise = fenics.project(function + noise, function.function_space(), annotate=False)
    return fun_with_noise

# 0) base Functional
#J = fenics.Functional( fenics.inner(u-u_target, u-u_target)*sim.subdomains.dx)

# 1) Optimization, target fields projected into function space
# m_target_tmp, w_target_tmp = fenics.split(u_target)
# m_target = sim.functionspace.project_over_space(m_target_tmp, subspace_id=0, annotate=False)
# w_target = sim.functionspace.project_over_space(w_target_tmp, subspace_id=1, annotate=False)
# m, w = fenics.split(u)
#
# J = fenics.Functional(   fenics.inner(m - m_target, m - m_target)*sim.subdomains.dx
#                        + fenics.inner(w - w_target, w - w_target) * sim.subdomains.dx)

# 2) Noise
m_target_tmp, w_target_tmp = fenics.split(u_target)
m_target = sim.functionspace.project_over_space(m_target_tmp, subspace_id=0, annotate=False)
w_target = sim.functionspace.project_over_space(w_target_tmp, subspace_id=1, annotate=False)
# this level of noise still results in good approximation of original parameters
# conc_target_noise = add_noise(w_target, 0.01)
# disp_target_noise = add_noise(m_target, 0.005)
# conc_target_noise = add_noise(w_target, 0.02)
# disp_target_noise = add_noise(m_target, 0.01)
conc_target_noise = add_noise(w_target, 0.1)
disp_target_noise = add_noise(m_target, 0.05)

disp, conc = fenics.split(u)
alpha = 0.5
J = fenics.Functional(   fenics.inner(conc - conc_target_noise, conc - conc_target_noise) * sim.subdomains.dx
                       + fenics.inner(disp - disp_target_noise, disp - disp_target_noise) * sim.subdomains.dx
                       + alpha * fenics.inner(u , u) * sim.subdomains.dx
                         )

# 2) Noise
m_target_tmp, w_target_tmp = fenics.split(u_target)
m_target = sim.functionspace.project_over_space(m_target_tmp, subspace_id=0, annotate=False)
w_target = sim.functionspace.project_over_space(w_target_tmp, subspace_id=1, annotate=False)
# this level of noise still results in good approximation of original parameters
# conc_target_noise = add_noise(w_target, 0.01)
# disp_target_noise = add_noise(m_target, 0.005)
# conc_target_noise = add_noise(w_target, 0.02)
# disp_target_noise = add_noise(m_target, 0.01)
conc_target_noise = add_noise(w_target, 0.1)
disp_target_noise = add_noise(m_target, 0.05)

disp, conc = fenics.split(u)
alpha = 0.5
J = fenics.Functional(   fenics.inner(conc - conc_target_noise, conc - conc_target_noise) * sim.subdomains.dx
                       + fenics.inner(disp - disp_target_noise, disp - disp_target_noise) * sim.subdomains.dx
                       + alpha * fenics.inner(u , u) * sim.subdomains.dx
                         )


controls = [fenics.ConstantControl(D), fenics.ConstantControl(rho), fenics.ConstantControl(c)]

#
# class ParameterEstimation():
#
#     def __init__(self, sim):
#         self.sim = sim
#
#     def set

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



reduced_functional = fenics.ReducedFunctional(J, controls,
                                              derivative_cb_post=derivative_cb_post, eval_cb_post=eval_cb_post)

dJdu = fenics.compute_gradient(J, controls)
bounds = [[0.05, 0.01, 0.05],
          [0.5, 0.2, 0.05]]
m_opt = fenics.minimize(reduced_functional, bounds=bounds,
                        method = 'SLSQP', options={'disp':True})

opt_df = create_opt_progress_df(opt_param_progress_post, opt_dj_progress_post, ['D', 'rho', 'c'])
opt_df.to_excel(os.path.join(output_path, 'optimization.xls'))
opt_df.to_pickle(os.path.join(output_path, 'optimization.pkl'))

# ==============================================================================
# Plotting
# ==============================================================================

plott.show_img_seg_f(function=m_target, path=os.path.join(output_path, 'disp_target_orig.png'))
plott.show_img_seg_f(function=w_target, path=os.path.join(output_path, 'conc_target_orig.png'))

conc_target_noise = add_noise(w_target, 0.1)
plott.show_img_seg_f(function=conc_target_noise, path=os.path.join(output_path, 'conc_target_noise.png'))

disp_target_noise = add_noise(m_target, 0.05)
plott.show_img_seg_f(function=disp_target_noise, path=os.path.join(output_path, 'disp_target_noise.png'))

# ==============================================================================
# RESULTS
# ==============================================================================
# Plot when adjoint computation has finished to avoid recording of function projections
# sim.plotting.plot_all(sim_time)
#
# output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_uniform_adjoint')
# fu.ensure_dir_exists(output_path)
#
# sim.init_postprocess(os.path.join(output_path, 'postprocess', 'plots'))
#
# sim.postprocess.plot_all(deformed=False)
# sim.postprocess.plot_all(deformed=True)
#
for var in m_opt:
    print(var.values())