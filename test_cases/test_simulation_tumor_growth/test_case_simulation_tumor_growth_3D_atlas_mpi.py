"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation, MPI enabled
 - 3D test domain from brain atlas, 4 tissue subdomains
 - spatially heterogeneous parameters
 - von Neumann BCs on tissue boundaries

!! Run test_cases/test_case_simulation_tumor_growth/convert_vtk_mesh_to_fenics_hdf5.py (without mpi) before starting this simulation to produce 'brain_atlas_mesh_3d.hdf5' !!
"""

import logging
import os

import meshio as mio

import test_cases.test_simulation_tumor_growth.testing_config as test_config

from simulation.simulation_tumor_growth import TumorGrowth
import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio

# ==============================================================================
# Logging settings
# ==============================================================================

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fenics.set_log_level(fenics.PROGRESS)

# ==============================================================================
# Load 3D Mesh
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


# ==============================================================================
# Class instantiation & Setup
# ==============================================================================
sim_time = 5
sim_time_step = 1

sim = TumorGrowth(mesh)

sim.projection_parameters = {'solver_type':'cg',
                             'preconditioner_type':'amg'}

sim.setup_global_parameters( subdomains=subdomains,
                             domain_names=tissue_id_name_map,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

youngmod = {'WM'  : 3000E-6,
            'GM'  : 3000E-6,
            'CSF' : 1000E-6,
            'Ventricles': 1000E-6}
poisson = { 'WM'  : 0.4,
            'GM'  : 0.4,
            'CSF' : 0.47,
            'Ventricles': 0.3}
diffusion = { 'WM'  : 0.1,
            'GM'  : 0.02,
            'CSF' : 0.0,
            'Ventricles': 0.0}
prolif  = { 'WM'  : 0.05,
            'GM'  : 0.05,
            'CSF' : 0.0,
            'Ventricles':0.0}
coupling = {
           'CSF': 0.0,
           'WM': 0.1,
           'GM': 0.1,
           'Ventricles': 0.0}


ivs = {0:u_0_disp_expr, 1:u_0_conc_expr}
sim.setup_model_parameters(iv_expression=ivs,
                            diffusion=diffusion,
                            coupling=coupling,
                            proliferation=prolif,
                            E=youngmod,
                            poisson=poisson,
                            sim_time=sim_time, sim_time_step=sim_time_step)

# ==============================================================================
# Run Simulation
# ==============================================================================
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_3D_atlas_mpi')
fu.ensure_dir_exists(output_path)

# runs simulation with above settings
#sim.run(save_method='xdmf',plot=False, output_dir=output_path, clear_all=False, keep_nth=1)


# ==============================================================================
# Reload and Plot (uncomment when using mpi)
# ==============================================================================

# reload simulations

path_to_h5_file = os.path.join(output_path, 'solution_timeseries.h5')
sim.reload_from_hdf5(path_to_h5_file)

# write results as vtu

sim.init_postprocess(output_path)
sim.postprocess.save_all(save_method='vtk', clear_all=False, selection=slice(1,-1,1))


