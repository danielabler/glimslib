"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation
 - 3D test domain from brain atlas, 4 tissue subdomains
 - spatially heterogeneous parameters
 - von Neumann BCs on tissue boundaries
"""

import logging
import os

import meshio as mio

import test_cases.test_simulation_tumor_growth.testing_config as test_config

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
# Load 3D Mesh from VTU
# ==============================================================================

inpath = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_3d.vtu')
mymesh = mio.read(inpath)
mesh, subdomains = dio.convert_meshio_to_fenics_mesh(mymesh)

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
sim_time = 50
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
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_3D_atlas')
fu.ensure_dir_exists(output_path)
sim.run(save_method='vtk',plot=False, output_dir=output_path, clear_all=True, keep_nth=1)

# ==============================================================================
# PostProcess
# ==============================================================================

dio.merge_VTUs(output_path, sim_time_step, sim_time, remove=True, reference=None)



