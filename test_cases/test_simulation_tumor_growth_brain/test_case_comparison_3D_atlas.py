"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth_brain`:
 - forward simulation
 - 2D test domain from brain atlas, 4 tissue subdomains + 'outside'
 - spatially heterogeneous parameters, as defined in simulation.simulation_tumor_growth_brain
 - no displacement bc between 'outside' and other subdomains

 !! Run test_cases/test_case_simulation_tumor_growth/convert_vtk_mesh_to_fenics_hdf5.py (without mpi) before starting this simulation to produce 'brain_atlas_mesh_3d.hdf5' !!
"""

import logging
import os

import test_cases.test_simulation_tumor_growth_brain.testing_config as test_config

from glimslib.simulation import TumorGrowthBrain
from glimslib.simulation.simulation_tumor_growth import TumorGrowth
from glimslib import fenics_local as fenics
import glimslib.utils.file_utils as fu
import glimslib.utils.data_io as dio

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fenics.set_log_level(fenics.PROGRESS)

# ==============================================================================
# Load 3D MESH
# ==============================================================================

path_to_hdf5_mesh = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_3d.hdf5')
# read mesh from hdf5 file
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


# Boundaries & BCs
boundary = Boundary()
boundary_dict = {'boundary_all': boundary}
dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant((0.0, 0.0, 0.0)),
                                    'boundary_name': 'boundary_all',
                                    'subspace_id': 0}
                      }

# Simulation() generates boundaries between subdomains,
# we want to apply no-flux von Neuman BC between tissue and CSF

boundaries_no_flux = ['WM_Ventricles', 'GM_Ventricles', 'CSF_WM', 'CSF_GM']
von_neuman_bcs = {}
for boundary in boundaries_no_flux:
    bc_name = "no_flux_%s"%boundary
    bc_dict = {'bc_value' : fenics.Constant(0.0),
               'boundary_name' : boundary,
               'subspace_id'   : 1}
    von_neuman_bcs[bc_name] = bc_dict

# Initial Values
u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2) - a*pow(x[2]-z0,2))', degree=1, a=0.5,
                                  x0=118, y0=-109, z0=72)
# u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)+pow(x[2]-z0,2)) < 15 ? (1.0) : (0.0)',
#                                 degree=1, x0=118, y0=-109, z0=72)

u_0_disp_expr = fenics.Expression(('0.0','0.0','0.0'), degree=1)

ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}

# ==============================================================================
# Parameters
# ==============================================================================
sim_time = 500
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


youngmod = {'CSF': E_CSF,
            'WM': E_WM,
            'GM': E_GM,
            'Ventricles': E_VENT}

poisson = {'CSF': nu_CSF,
           'WM': nu_WM,
           'GM': nu_GM,
           'Ventricles': nu_VENT}

diffusion = {'CSF': 0.0,
            'WM': D_WM,
            'GM': D_GM,
            'Ventricles': 0.0}

prolif = { 'CSF': 0.0,
           'WM': rho_WM,
           'GM': rho_GM,
           'Ventricles': 0.0}

coupling = 0.1

# ==============================================================================
# TumorGrowth
# ==============================================================================

sim_TG = TumorGrowth(mesh)

sim_TG.setup_global_parameters(subdomains=subdomains,
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

output_path_TG = os.path.join(test_config.output_path, 'test_case_comparison_3D_atlas', 'simulation_tumor_growth')
fu.ensure_dir_exists(output_path_TG)
#sim_TG.run(save_method='xdmf',plot=False, output_dir=output_path_TG, clear_all=False)

# ==============================================================================
# TumorGrowthBrain
# ==============================================================================

sim_TGB = TumorGrowthBrain(mesh)

sim_TGB.setup_global_parameters(subdomains=subdomains,
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

output_path_TGB = os.path.join(test_config.output_path, 'test_case_comparison_3D_atlas', 'simulation_tumor_growth_brain')
fu.ensure_dir_exists(output_path_TGB)
#sim_TGB.run(save_method='xdmf',plot=False, output_dir=output_path_TGB, clear_all=False)

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

sim_TGB.init_postprocess(output_path_TGB)
sim_TGB.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)


# ==============================================================================
# Compare
# ==============================================================================
from glimslib.simulation_helpers import Comparison

comp = Comparison(sim_TG, sim_TGB)
df = comp.compare(slice(300,300,100))