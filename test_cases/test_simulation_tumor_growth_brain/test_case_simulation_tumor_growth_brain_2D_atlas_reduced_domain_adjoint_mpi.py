"""
Example for adjoint solution on 2D atlas.

!! Run test_cases/test_case_simulation_tumor_growth/convert_vtk_mesh_to_fenics_hdf5.py (without mpi) before starting this simulation to produce 'brain_atlas_mesh_2d.hdf5' and  'brain_atlas_labelfunction_2d.hdf5'!!

"""

import logging
import os

config.USE_ADJOINT=True
import test_cases.test_simulation_tumor_growth.testing_config as test_config

from glimslib.simulation import TumorGrowthBrain
from glimslib import fenics_local as fenics, config
import glimslib.utils.file_utils as fu
import glimslib.utils.data_io as dio

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fenics.set_log_level(fenics.CRITICAL)

# ==============================================================================
# Load 2D Mesh from IMAGE
# ==============================================================================
path_to_hdf5_mesh = os.path.join(config.test_data_dir, 'brain_atlas_mesh_2d_reduced_domain.h5')
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

von_neuman_bcs = {}


# Initial Values
u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1, a=0.5, x0=148, y0=-67)
u_0_disp_expr = fenics.Constant((0.0, 0.0))

ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}
# ==============================================================================
# Parameters
# ==============================================================================
sim_time = 300
sim_time_step = 1
E_GM=3000E-6
E_WM=3000E-6
E_CSF=1000E-6
E_VENT=1000E-6
nu_GM=0.4
nu_WM=0.4
nu_CSF=0.4
nu_VENT=0.3
D_GM=0.01
D_WM=0.05
rho_GM=0.05
rho_WM=0.05
coupling = 0.1

# ==============================================================================
# Class instantiation & Setup
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
# Run Simulation
# ==============================================================================
output_path_forward = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_brain_2D_atlas_reduced_domain_adjoint_mpi', 'forward')
fu.ensure_dir_exists(output_path_forward)

D_GM_target=0.02
D_WM_target=0.1
rho_GM_target=0.1
rho_WM_target=0.1
coupling_target=0.15

params_target = [D_WM_target, D_GM_target, rho_WM_target, rho_GM_target, coupling_target]
u_target = sim.run_for_adjoint(params_target, output_dir=output_path_forward)
sim.run(save_method='xdmf',plot=False, output_dir=output_path_forward, clear_all=True)

# ==============================================================================
# OPTIMISATION
# ==============================================================================
output_path_adjoint = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_brain_2D_atlas_reduced_domain_adjoint_mpi', 'adjoint')

D_GM_init=0.01
D_WM_init=0.01
rho_GM_init=0.01
rho_WM_init=0.01
coupling_init=0.2
params_init = [D_WM_init, D_GM_init, rho_WM_init, rho_GM_init, coupling_init]
params_init = [fenics.Constant(param) for param in params_init]

u = sim.run_for_adjoint(params_init, output_dir=output_path_adjoint)


J = fenics.Functional( fenics.inner(u-u_target, u-u_target)*sim.subdomains.dx)
controls = [fenics.ConstantControl(param) for param in params_init]

def eval_cb(j, a):
    params = [param.values() for param in a]
    print(j, *params)

reduced_functional = fenics.ReducedFunctional(J, controls, eval_cb_post=eval_cb)

m_opt = fenics.minimize(reduced_functional)

for var in m_opt:
    print(var.values())


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
# for var in m_opt:
#     print(var.values())


# ==============================================================================
# Reload and Plot
# ==============================================================================
#
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
