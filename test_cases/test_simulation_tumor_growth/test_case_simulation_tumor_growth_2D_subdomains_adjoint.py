"""
Example demonstrating usage of :py:meth:`simulation.simulation_tumor_growth`:
 - forward simulation & adjoint optimisation simulation
 - 2D test domain with 2 subdomains
 - Mixed spatially homogeneous and heterogeneous parameters.
 - Diffusion and proliferation parameters mimick interface between CSF and brain tissue
 - Optimisation of spatially homogenous parameters in heterogeneous domain.
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

# Mesh
nx = ny = 50
mesh = fenics.RectangleMesh(fenics.Point(-5, -5), fenics.Point(5, 5), nx, ny)

# LabelMap
label_funspace = fenics.FunctionSpace(mesh, "DG", 1)
label_expr = fenics.Expression('(x[0]>=0.0) ? (1.0) : (2.0)', degree=1)
labels = fenics.project(label_expr, label_funspace)
tissue_map = {0: 'outside',
              1: 'A',
              2: 'B'}

# Boundaries & BCs
boundary = Boundary()
boundary_dict = {'boundary_all': boundary}
dirichlet_bcs = {'clamped_outside': {'bc_value': fenics.Constant((0.0, 0.0)),
                                    'named_boundary': 'boundary_all',
                                    'subspace_id': 0},
                 # Test to show that Dirichlet BCs can be applied to subdomain interfaces
                 # 'clamped_A_B'    : {'bc_value': fenics.Constant((0.0, 0.0)),
                 #                    'subdomain_boundary': 'A_B',
                 #                    'subspace_id': 0}
                      }
von_neuman_bcs = {  # not necessary to impose zero flux at domain boundary
                  # 'no_flux_boundary': {'bc_value': fenics.Constant(0.0),
                  #                    'named_boundary': 'boundary_all',
                  #                    'subspace_id': 1}
   }

# Initial Values
u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.4 ? (1.0) : (0.0)',
                                        degree=1, x0=2.5, y0=2.5)
u_0_disp_expr = fenics.Constant((0.0, 0.0))

# model parameters
youngmod = {'outside': 10E6,
            'A' : 0.001,
            'B'  : 0.001}
poisson = {'outside': 0.49,
           'A' : 0.40,
           'B' : 0.10}
diffusion = {'outside': 0.0,
           'A' : 0.1,
           'B' : 0.0}
prolif  = {'outside': 0.0,
           'A' : 0.1,
           'B' : 0.0}
coupling = {'outside': 0.0,
           'A' : 0.2,
           'B' : 0.0}

# ==============================================================================
# Class instantiation & Setup
# ==============================================================================
sim_time = 5
sim_time_step = 1

sim = TumorGrowth(mesh)

sim.setup_global_parameters(label_function=labels,
                             domain_names=tissue_map,
                             boundaries=boundary_dict,
                             dirichlet_bcs=dirichlet_bcs,
                             von_neumann_bcs=von_neuman_bcs
                             )

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
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_subdomains_adjoint', 'forward')
fu.ensure_dir_exists(output_path)

D_target  = 0.3
rho_target  = 0.1
coupling_target = 0.2

u_target = sim.run_for_adjoint([D_target, rho_target, coupling_target], output_dir=output_path)


# ==============================================================================
# OPTIMISATION
# ==============================================================================
output_path = os.path.join(test_config.output_path, 'test_case_simulation_tumor_growth_2D_subdomains_adjoint', 'adjoint')
fu.ensure_dir_exists(output_path)

D  = fenics.Constant(0.1)
rho  = fenics.Constant(0.01)
coupling = fenics.Constant(0.1)

u = sim.run_for_adjoint([D, rho, coupling], output_dir=output_path)


J = fenics.Functional( fenics.inner(u-u_target, u-u_target)*sim.subdomains.dx)
controls = [fenics.ConstantControl(D), fenics.ConstantControl(rho), fenics.ConstantControl(coupling) ]

def eval_cb(j, a):
    D, rho, coupling    = a
    print(j, D.values(), rho.values(), coupling.values())

reduced_functional = fenics.ReducedFunctional(J, controls, eval_cb_post=eval_cb)
m_opt = fenics.minimize(reduced_functional)


# ==============================================================================
# RESULTS
# ==============================================================================
# Plot when adjoint computation has finished to avoid recording of function projections
sim.plotting.plot_all(sim_time)

sim.init_postprocess(os.path.join(output_path, 'postprocess', 'plots'))

sim.postprocess.plot_all(deformed=False)
sim.postprocess.plot_all(deformed=True)

for var in m_opt:
    print(var.values())


# ==============================================================================
# COMMENTS concerning OPTIMISATION of spatially HETEROGENEOUS PARAMETERS
# ==============================================================================

# Dolfin adjoint seems not to be working for parameters that are defined as expressions over subdomains, such as
# instances of the Discontinous Scalara class, even, if the parameters are supplied as scalars to the simulation
# function call such as here:

# import collections
# class DiscontinuousScalar(fenics.Expression):
#     """
#     Creates scalar with different values in each subdomains.
#     """
#     def __init__(self, cell_function, scalars, **kwargs):
#         self.cell_function = cell_function
#         self.coeffs = scalars
#
#     def eval_cell(self, values, x, cell):
#         subdomain_id = self.cell_function[cell.index]
#         local_coeff = self.coeffs[subdomain_id]
#         local_coeff.eval_cell(values, x, cell)
#
# def run_for_adjoint(param, **kwargs):
#     mapdict_ordered = collections.OrderedDict(sim.subdomains.tissue_id_name_map)
#     param_list = []
#     param_list.append(fenics.Constant(0))
#     if isinstance(param, fenics.ConstantControl):
#         param_list.append(param)
#     else:
#         param_list.append(fenics.Constant(param))
#     param_list.append(fenics.Constant(0.9))
#     d = DiscontinuousScalar(sim.subdomains.subdomains, param_list, degree=1)
#     u = sim.run_for_adjoint_diff(d, **kwargs)
#     return u

# With this formulation, fenics.compute_gradient returns None and thus the minimzation routine fails.
#
# dJdnu = fenics.compute_gradient(J, controls)
#
# Replaying the forward model also fails:
#
# success = fenics.replay_dolfin(tol=0.0, stop=True)
#
# Adjoint information:
# fenics.adj_html("forward.html", "forward")
# fenics.adj_html("adjoint.html", "adjoint")

