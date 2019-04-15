import logging
import os

config.USE_ADJOINT = True
import test_cases.test_image_based_optimisation.testing_config as test_config

from glimslib.simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from glimslib import fenics_local as fenics, visualisation as plott
import glimslib.utils.data_io as dio

output_path = test_config.path_06_comparison_from_Image

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.PROGRESS)
else:
    fenics.set_log_level(fenics.LogLevel.PROGRESS)

# ==============================================================================
# Problem Settings
# ==============================================================================

class Boundary(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

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
# Optimized
# ==============================================================================

path_to_hdf5_mesh = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_2d_reduced_domain.h5')
mesh_reduced, subdomains, boundaries = dio.read_mesh_hdf5(path_to_hdf5_mesh)

sim_opt = TumorGrowthBrain(mesh_reduced)

tissue_id_name_map_reduced = {    1: 'CSF',
                          3: 'WM',
                          2: 'GM',
                          4: 'Ventricles'}


sim_opt.setup_global_parameters(subdomains=subdomains,
                             domain_names=tissue_id_name_map_reduced,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

sim_opt.setup_model_parameters(iv_expression=ivs,
                           sim_time=sim_time, sim_time_step=sim_time_step,
                           E_GM=E_GM, E_WM=E_WM, E_CSF=E_CSF, E_VENT=E_VENT,
                           nu_GM=nu_GM, nu_WM=nu_WM, nu_CSF=nu_CSF, nu_VENT=nu_VENT,
                           D_GM=D_GM, D_WM=D_WM,
                           rho_GM=rho_GM, rho_WM=rho_WM,
                           coupling=coupling)


output_path_opt = test_config.path_05_forward_simulation_optimized_from_image

path_to_h5_file = os.path.join(output_path_opt, 'solution_timeseries.h5')
sim_opt.reload_from_hdf5(path_to_h5_file)
sim_opt.init_postprocess(output_path_opt)
# ==============================================================================
# Original
# ==============================================================================

sim_orig = TumorGrowthBrain(mesh_reduced)

sim_orig.setup_global_parameters(subdomains=subdomains,
                             domain_names=tissue_id_name_map_reduced,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

sim_orig.setup_model_parameters(iv_expression=ivs,
                           sim_time=sim_time, sim_time_step=sim_time_step,
                           E_GM=E_GM, E_WM=E_WM, E_CSF=E_CSF, E_VENT=E_VENT,
                           nu_GM=nu_GM, nu_WM=nu_WM, nu_CSF=nu_CSF, nu_VENT=nu_VENT,
                           D_GM=D_GM, D_WM=D_WM,
                           rho_GM=rho_GM, rho_WM=rho_WM,
                           coupling=coupling)

output_path_orig = test_config.path_01_forward_simulation_red

path_to_h5_file_orig = os.path.join(output_path_orig, 'solution_timeseries.h5')
sim_orig.reload_from_hdf5(path_to_h5_file_orig)
sim_orig.init_postprocess(output_path_orig)

# ==============================================================================
# Project Original to reduced domain
# ==============================================================================

def interpolate_non_matching(source_function, target_funspace):
    function_new = fenics.Function(target_funspace)
    fenics.LagrangeInterpolator.interpolate(function_new, source_function)
    return function_new

#-- chose simulation mesh as reference
funspace_disp_reduced = fenics.VectorFunctionSpace(mesh_reduced, 'Lagrange', 1)
funspace_conc_reduced = fenics.FunctionSpace(mesh_reduced, 'Lagrange', 2)

comp_time_step = test_config.params_sim['sim_time']

conc_orig_full_domain = sim_orig.postprocess.get_solution_concentration(comp_time_step).copy()
disp_orig_full_domain = sim_orig.postprocess.get_solution_displacement(comp_time_step).copy()

conc_orig = interpolate_non_matching(conc_orig_full_domain, funspace_conc_reduced)
disp_orig = interpolate_non_matching(disp_orig_full_domain, funspace_disp_reduced)

# ==============================================================================
# Compare forward to opt
# ==============================================================================

conc_opt = sim_opt.postprocess.get_solution_concentration(comp_time_step).copy()
disp_opt = sim_opt.postprocess.get_solution_displacement(comp_time_step).copy()


print("Errornorm Concentration: ", fenics.errornorm(conc_orig, conc_opt))
plott.show_img_seg_f(function=conc_orig, path=os.path.join(output_path,'conc_forward.png'))
plott.show_img_seg_f(function=conc_opt, path=os.path.join(output_path,'conc_opt.png'))
conc_diff = fenics.project(conc_orig-conc_opt, funspace_conc_reduced)
plott.show_img_seg_f(function=conc_diff, path=os.path.join(output_path,'conc_diff.png'))

print("Errornorm Displacement: ",fenics.errornorm(disp_orig, disp_opt))
plott.show_img_seg_f(function=disp_orig, path=os.path.join(output_path,'disp_forward.png'))
plott.show_img_seg_f(function=disp_opt, path=os.path.join(output_path,'disp_opt.png'))
disp_diff = fenics.project(disp_orig-disp_opt, funspace_disp_reduced)
plott.show_img_seg_f(function=disp_diff, path=os.path.join(output_path,'disp_diff.png'))

plot_params = {'show_axes': False,
              'show_ticks': False,
              'show_title': False,
              'show_cbar': False}

sim_opt.init_postprocess(output_path)
sim_opt.postprocess.plot_function(conc_diff, 250,
                              name='concentration (target - optimized)',
                              file_name='conc_diff_2.png',
                              show_labels=False,
                              range_f = [-1,1], colormap='RdBu_r', cmap_ref=0,
                              show_axes=False, show_title=False, show_ticks=False)


sim_opt.postprocess.plot_function(disp_diff, 250,
                              name='displacement (target - optimized)',
                              file_name='disp_diff_2.png',
                              show_labels=False,
                              range_f = [-1,1], colormap='RdBu_r', cmap_ref=0,
                              show_axes=False, show_title=False, show_ticks=False)