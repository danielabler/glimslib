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



def save_function_to_xdmf(path_to_file, function, function_name):
    if os.path.isfile(path_to_file):
        os.remove(path_to_file)
    mesh = function.function_space().mesh()
    output_xdmf_file = fenics.XDMFFile(mesh.mpi_comm(), path_to_file)
    output_xdmf_file.write(mesh)
    output_xdmf_file.write_checkpoint(function, function_name, 0)
    output_xdmf_file.close()


def load_function_from_xdmf(path_to_file, functionname, functionspace, step=0):
    mpi_comm_world = fenics.mpi_comm_world()
    function_file = fenics.XDMFFile(mpi_comm_world, path_to_file)
    mesh = fenics.Mesh()
    function_file.read(mesh)
    f = fenics.Function(functionspace)
    function_file.read_checkpoint(f, functionname, step)
    return f

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
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_uniform_adjoint_reloaded')
fu.ensure_dir_exists(output_path)


D_target    = 0.05
rho_target  = 0.05
c_target    = 0.1

u_target = sim.run_for_adjoint([D_target, rho_target, c_target], output_dir=output_path)
disp_sim_target_0, conc_sim_target_0 = fenics.split(u_target)

conc_sim_target = sim.functionspace.project_over_space(conc_sim_target_0, subspace_id=1)
disp_sim_target = sim.functionspace.project_over_space(disp_sim_target_0, subspace_id=0)

fenics.parameters["adjoint"]["stop_annotating"] = True
import visualisation.plotting as plott
plott.show_img_seg_f(function=conc_sim_target, show=True,
                     path=os.path.join(output_path, 'conc_target_sim.png'),
                     dpi=300)
plott.show_img_seg_f(function=disp_sim_target, show=True,
                     path=os.path.join(output_path, 'disp_target_sim.png'),
                     dpi=300)

# path_to_sim_conc = os.path.join(output_path, 'concentration_from_simulation.h5')
# dio.save_function_mesh(conc_sim_target, path_to_sim_conc)
# path_to_sim_disp = os.path.join(output_path, 'displacement_from_simulation.h5')
# dio.save_function_mesh(disp_sim_target, path_to_sim_disp)
#
# path_to_sim_mixed = os.path.join(output_path, 'mixed_from_simulation.h5')
# #save_function_to_xdmf(path_to_sim_mixed, u_target, 'u')
# dio.save_function_mesh(u_target, path_to_sim_mixed)
# fenics.parameters["adjoint"]["stop_annotating"] = False


path_to_xdmf = os.path.join(output_path, 'xdmf_from_simulation.xdmf')
with fenics.XDMFFile(path_to_xdmf) as outfile:
    outfile.write_checkpoint(disp_sim_target, "disp")
    outfile.write_checkpoint(conc_sim_target, "conc")




#
#
# conc_sim = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path_to_sim_conc)
# disp_sim = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=0),path_to_sim_disp)
# plott.show_img_seg_f(function=conc_sim, show=True,
#                      path=os.path.join(output_path, 'conc_target_reloaded.png'),
#                      dpi=300)
# plott.show_img_seg_f(function=disp_sim, show=True,
#                      path=os.path.join(output_path, 'disp_target_reloaded.png'),
#                      dpi=300)


#mixed_sim = load_function_from_xdmf(path_to_sim_mixed,'u', sim.functionspace.get_functionspace())
# fenics.errornorm(conc_sim_target, conc_sim)
# fenics.errornorm(disp_sim_target, disp_sim)
#fenics.errornorm(u_target, mixed_sim)

#fenics.assemble(fenics.inner(mixed_sim-u_target,mixed_sim-u_target)*fenics.dx)
# ==============================================================================
# OPTIMISATION
# ==============================================================================

D   = fenics.Constant(0.1)
rho = fenics.Constant(0.1)
c   = fenics.Constant(0.5)

u = sim.run_for_adjoint([D, rho, c], output_dir=output_path)

# fenics.parameters["adjoint"]["stop_annotating"] = True
# mixed_sim = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(),path_to_sim_mixed)
# fenics.parameters["adjoint"]["stop_annotating"] = False

# fenics.assemble(fenics.inner(mixed_sim-u,mixed_sim-u)*fenics.dx)
#
# fenics.assemble(fenics.inner(u_target-u,u_target-u)*fenics.dx)


disp, conc = fenics.split(u)

conc_opt = sim.functionspace.project_over_space(conc, subspace_id=1)
disp_opt = sim.functionspace.project_over_space(disp, subspace_id=0)

conc_sim = fenics.Function(sim.functionspace.get_functionspace(subspace_id=1))
disp_sim = fenics.Function(sim.functionspace.get_functionspace(subspace_id=0))

with fenics.XDMFFile(path_to_xdmf) as infile:
    infile.read_checkpoint(disp_sim, "disp")
    infile.read_checkpoint(conc_sim, "conc")

plott.show_img_seg_f(function=conc_sim, show=True,
                     path=os.path.join(output_path, 'conc_target_reloaded.png'),
                     dpi=300)
plott.show_img_seg_f(function=disp_sim, show=True,
                     path=os.path.join(output_path, 'disp_target_reloaded.png'),
                     dpi=300)


fenics.errornorm(conc_sim_target, conc_sim)
fenics.errornorm(disp_sim_target, disp_sim)



fenics.errornorm(conc_opt, conc_sim)
fenics.errornorm(disp_opt, disp_sim)
#fenics.errornorm(u, mixed_sim)



J = fenics.Functional(   fenics.inner(conc_opt - conc_sim, conc_opt - conc_sim) * fenics.dx
                       + fenics.inner(disp_opt - disp_sim, disp_opt - disp_sim) * fenics.dx)

fenics.adj_html(os.path.join(output_path, "forward.html"), "forward")
fenics.adj_html(os.path.join(output_path, "adjoint.html"), "adjoint")

#J = fenics.Functional(   fenics.inner(u - mixed_sim, u - mixed_sim) * fenics.dx)


controls = [fenics.ConstantControl(D), fenics.ConstantControl(rho), fenics.ConstantControl(c)]

#
# class ParameterEstimation():
#
#     def __init__(self, sim):
#         self.sim = sim
#
#     def set

import pandas as pd

# opt_param_progress_post = []
# opt_dj_progress_post = []
#
# def eval_cb_post(j, a):
#     values = [param.values()[0] for param in a]
#     result = (j, *values)
#     global opt_param_progress_post
#     opt_param_progress_post.append(result)
#     print(result)
#
# def derivative_cb_post(j, dj, m):
#     param_values = [param.values()[0] for param in m]
#     dj_values = [param.values()[0] for param in dj]
#     global opt_dj_progress_post
#     result = (j, *dj_values)
#     opt_dj_progress_post.append(result)
#     print(result)
#
#
# def create_opt_progress_df(opt_param_list, opt_dj_list, param_name_list):
#     columns_params = ['J', *param_name_list]
#     columns_dJ = ['J', *['dJd%s'%param for param in param_name_list]]
#     # create data frames
#     params_df = pd.DataFrame(opt_param_list)
#     params_df.columns = columns_params
#     dj_df = pd.DataFrame(opt_dj_list)
#     dj_df.columns = columns_dJ
#     # merge
#     opt_df = pd.merge(params_df, dj_df, on='J', how='outer')
#     return opt_df
#


reduced_functional = fenics.ReducedFunctional(J, controls)#,
                                              #derivative_cb_post=derivative_cb_post, eval_cb_post=eval_cb_post)

dJdu = fenics.compute_gradient(J, controls)
bounds = [[0.05, 0.01, 0.05],
          [0.5, 0.2, 0.5]]
m_opt = fenics.minimize(reduced_functional, bounds=bounds,
                        options={'disp':True, 'gtol':1e-10}, tol = 1e-10)

# opt_df = create_opt_progress_df(opt_param_progress_post, opt_dj_progress_post, ['D', 'rho', 'c'])
# opt_df.to_excel(os.path.join(output_path, 'optimization.xls'))
# opt_df.to_pickle(os.path.join(output_path, 'optimization.pkl'))

# ==============================================================================
# Plotting
# ==============================================================================
#
# import visualisation.plotting as plott
#
# plott.show_img_seg_f(function=m_target, path=os.path.join(output_path, 'disp_target_orig.png'))
# plott.show_img_seg_f(function=w_target, path=os.path.join(output_path, 'conc_target_orig.png'))
#
# conc_target_noise = add_noise(w_target, 0.1)
# plott.show_img_seg_f(function=conc_target_noise, path=os.path.join(output_path, 'conc_target_noise.png'))
#
# disp_target_noise = add_noise(m_target, 0.05)
# plott.show_img_seg_f(function=disp_target_noise, path=os.path.join(output_path, 'disp_target_noise.png'))

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