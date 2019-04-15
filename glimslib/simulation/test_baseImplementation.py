from unittest import TestCase

from glimslib import fenics_local as fenics
from glimslib.simulation.simulation_tumor_growth import TumorGrowth


class Boundary(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class TestBaseImplementation(TestCase):

    def setUp(self):
        # Domain
        nx = ny = nz = 10
        self.mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        self.sim = TumorGrowth(self.mesh)
        label_funspace = fenics.FunctionSpace(self.mesh, "DG", 1)
        label_expr = fenics.Expression('(x[0]>=0.5) ? (1.0) : (2.0)', degree=1)
        self.labels = fenics.project(label_expr, label_funspace)
        self.tissue_map = {0: 'outside',
                              1: 'tissue',
                              2: 'tumor'}
        boundary = Boundary()
        self.boundary_dict = {'boundary_1': boundary,
                              'boundary_2': boundary}
        self.dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant((0.0, 0.0)),
                                            'boundary': Boundary(),
                                            'subspace_id': 0},
                              'clamped_1': {'bc_value': fenics.Constant((0.0, 0.0)),
                                            'boundary_id': 0,
                                            'subspace_id': 0},
                              'clamped_2': {'bc_value': fenics.Constant((0.0, 0.0)),
                                            'boundary_name': 'boundary_1',
                                            'subspace_id': 0}
                              }
        self.von_neuman_bcs = {'no_flux': {'bc_value': fenics.Constant(0.0),
                                           'boundary_id': 0,
                                           'subspace_id': 1},
                               'no_flux_2': {'bc_value': fenics.Constant(0.0),
                                             'boundary_name': 'boundary_1',
                                             'subspace_id': 1},
                               }
        self.u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.1 ? (1.0) : (0.0)',
                                                degree=1, x0=0.25, y0=0.5)
        self.u_0_disp_expr = fenics.Constant((0.0, 0.0))

        self.youngmod = {'outside': 10E6,
                    'tissue': 1,
                    'tumor': 1000}
        self.poisson = {'outside': 0.4,
                   'tissue': 0.4,
                   'tumor': 0.49}
        self.diffusion = {'outside': 0.0,
                     'tissue': 1.0,
                     'tumor': 0.1}
        self.prolif = {'outside': 0.0,
                  'tissue': 0.1,
                  'tumor': 1.0}
        self.coupling = {'outside': 0.0,
                    'tissue': 0.0,
                    'tumor': 0.5}

    def test_setup_global_parameters(self):
        # subdomains from labels
        self.sim.setup_global_parameters(label_function=self.labels,
                                         domain_names=self.tissue_map,
                                         boundaries=self.boundary_dict,
                                         dirichlet_bcs=self.dirichlet_bcs,
                                         von_neumann_bcs=self.von_neuman_bcs
                                         )
        # subdomains
        self.assertTrue(hasattr(self.sim,'subdomains'))
        self.assertTrue(hasattr(self.sim.subdomains, 'subdomain_boundaries'))
        # functionspaces
        self.assertTrue(hasattr(self.sim.functionspace,'element'))
        self.assertTrue(hasattr(self.sim.functionspace, 'subspaces'))
        # boundary conditions
        self.assertTrue(hasattr(self.sim.bcs, 'dirichlet_bcs'))
        self.assertEqual(len(self.sim.bcs.dirichlet_bcs), 3)
        self.assertTrue(hasattr(self.sim.bcs, 'von_neumann_bcs'))
        self.assertEqual(len(self.sim.bcs.von_neumann_bcs), 2)

    def test_setup_model_parameters(self):
        self.sim.setup_global_parameters(label_function=self.labels,
                                         domain_names=self.tissue_map,
                                         boundaries=self.boundary_dict,
                                         dirichlet_bcs=self.dirichlet_bcs,
                                         von_neumann_bcs=self.von_neuman_bcs
                                         )
        ivs = {0:self.u_0_disp_expr, 1:self.u_0_conc_expr}
        self.sim.setup_model_parameters(iv_expression=ivs,
                                        diffusion=1,
                                        coupling=1,
                                        proliferation=1,
                                        E=self.params,
                                        poisson=self.params,
                                        otherparam=1,
                                        sim_time=10, sim_time_step=1)
        self.assertTrue(self.sim.params.get_iv(0), self.u_0_disp_expr)
        self.assertTrue(hasattr(self.sim.params, 'E'))
        self.assertFalse(hasattr(self.sim.params, 'otherparam'))



