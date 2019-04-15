"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth_brain`:
 - forward simulation
 - 2D test domain from brain atlas, 4 tissue subdomains + 'outside'
 - spatially heterogeneous parameters, as defined in simulation.simulation_tumor_growth_brain
 - no displacement bc between 'outside' and other subdomains

 !! Run test_cases/test_case_simulation_tumor_growth/convert_vtk_mesh_to_fenics_hdf5.py (without mpi) before starting this simulation to produce 'brain_atlas_mesh_2d.hdf5' and  'brain_atlas_labelfunction_2d.hdf5'!!

"""

import logging
import os

import test_cases.test_simulation_tumor_growth_brain.testing_config as test_config

from glimslib.simulation import TumorGrowthBrain
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
path_to_atlas = os.path.join(config.test_data_dir, 'brain_atlas_mesh_2d.hdf5')
mesh, subdomains, boundaries = dio.read_mesh_hdf5(path_to_atlas)

path_to_label = os.path.join(config.test_data_dir, 'brain_atlas_labelfunction_2d.hdf5')
V = fenics.FunctionSpace(mesh, "Lagrange", 1)
labelfunction = dio.read_function_hdf5("labelfunction", V, path_to_label)

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
sim_time = 2
sim_time_step = 1

E_GM=3000E-6
E_WM=3000E-6
E_CSF=1000E-6
E_VENT=1000E-6
nu_GM=0.45
nu_WM=0.45
nu_CSF=0.45
nu_VENT=0.3
D_GM=0.01
D_WM=0.05
rho_GM=0.05
rho_WM=0.05


youngmod = {'outside': 10E3,
            'CSF': E_CSF,
            'WM': E_WM,
            'GM': E_GM,
            'Ventricles': E_VENT}

poisson = {'outside': 0.45, #convergence problem if ~0.49
           'CSF': nu_CSF,
           'WM': nu_WM,
           'GM': nu_GM,
           'Ventricles': nu_VENT}

diffusion = {'outside': 0.0,
            'CSF': 0.0,
            'WM': D_WM,
            'GM': D_GM,
            'Ventricles': 0.0}

prolif = {'outside': 0.0,
           'CSF': 0.0,
           'WM': rho_WM,
           'GM': rho_GM,
           'Ventricles': 0.0}

coupling = 0.1

# ==============================================================================
# TumorGrowth
# ==============================================================================

sim_TG = TumorGrowth(mesh)

sim_TG.setup_global_parameters(label_function=labelfunction,
                             domain_names=tissue_id_name_map,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

sim_TG.setup_model_parameters(iv_expression=ivs,
                            diffusion=diffusion,
                            coupling=coupling,
                            proliferation=prolif,
                            E=youngmod,
                            poisson=poisson,
                            sim_time=sim_time, sim_time_step=sim_time_step)

output_path_TG = os.path.join(test_config.output_path, 'test_case_comparison_2D_atlas_2', 'simulation_tumor_growth')
fu.ensure_dir_exists(output_path_TG)
#sim_TG.run(save_method='xdmf',plot=False, output_dir=output_path_TG, clear_all=True)

# ==============================================================================
# TumorGrowthBrain
# ==============================================================================

sim_TGB = TumorGrowthBrain(mesh)

sim_TGB.setup_global_parameters(label_function=labelfunction,
                             domain_names=tissue_id_name_map,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

sim_TGB.setup_model_parameters(iv_expression=ivs,
                           sim_time=sim_time, sim_time_step=sim_time_step,
                           E_GM=E_GM, E_WM=E_WM, E_CSF=E_CSF, E_VENT=E_VENT,
                           nu_GM=nu_GM, nu_WM=nu_WM, nu_CSF=nu_CSF, nu_VENT=nu_VENT,
                           D_GM=D_GM, D_WM=D_WM,
                           rho_GM=rho_GM, rho_WM=rho_WM,
                           coupling=coupling)

output_path_TGB = os.path.join(test_config.output_path, 'test_case_comparison_2D_atlas_2', 'simulation_tumor_growth_brain')
fu.ensure_dir_exists(output_path_TGB)
sim_TGB.run(save_method='xdmf',plot=False, output_dir=output_path_TGB, clear_all=True)

# ==============================================================================
# Load Results
# ==============================================================================
selection = slice(1,-1,50)

path_to_h5_TG = os.path.join(output_path_TG, 'solution_timeseries.h5')
sim_TG.reload_from_hdf5(path_to_h5_TG)

path_to_h5_TGB = os.path.join(output_path_TGB, 'solution_timeseries.h5')
sim_TGB.reload_from_hdf5(path_to_h5_TGB)


sim_TG.init_postprocess(output_path_TG)
sim_TG.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)
sim_TG.postprocess.plot_all(deformed=False, selection=selection, output_dir=os.path.join(output_path_TG, 'postprocess_reloaded', 'plots'))
sim_TG.postprocess.plot_all(deformed=True, selection=selection, output_dir=os.path.join(output_path_TG, 'postprocess_reloaded', 'plots'))


sim_TGB.init_postprocess(output_path_TGB)
sim_TGB.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)
sim_TGB.postprocess.plot_all(deformed=False, selection=selection, output_dir=os.path.join(output_path_TGB, 'postprocess_reloaded', 'plots'))
sim_TGB.postprocess.plot_all(deformed=True, selection=selection, output_dir=os.path.join(output_path_TGB, 'postprocess_reloaded', 'plots'))



# ==============================================================================
# Compare
# ==============================================================================

from glimslib.simulation_helpers import Comparison

comp = Comparison(sim_TG, sim_TGB)
df = comp.compare(slice(1,300,100))