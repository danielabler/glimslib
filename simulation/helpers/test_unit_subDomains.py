from unittest import TestCase
from simulation.helpers.helper_classes import SubDomains
import fenics_local as fenics
import numpy as np


class TestSubDomains(TestCase):

    def setUp(self):
        nx = ny = nz = 5
        self.ny = ny
        self.nx = nx
        mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        self.subdomains = SubDomains(mesh)
        # 'LabelMap'
        label_funspace = fenics.FunctionSpace(mesh, "DG", 1)
        label_expr = fenics.Expression('(x[0]>=0) ? (1.0) : (2.0)', degree=1)
        labels = fenics.project(label_expr, label_funspace)
        self.labels = labels
        # tissue_id_name_map
        self.tissue_id_name_map = {   0: 'outside',
                                      1: 'tissue',
                                      2: 'tumor'}
        self.parameter = {'outside': 0.0,
                             'tissue': 1.0,
                             'tumor': 0.1}
        class Boundary(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        self.boundary = Boundary()

        boundary_dict = {'boundary_1': self.boundary}
        self.boundary_dict = boundary_dict

    def test_setup_subdomains(self):
        self.subdomains.setup_subdomains(label_function=self.labels)
        self.assertTrue(hasattr(self.subdomains,'subdomains'))
        subdomain_ids = np.unique(self.subdomains.subdomains.array())
        self.assertEqual(set(subdomain_ids),set([1, 2]))
        self.subdomains.setup_subdomains(replace=True)
        subdomain_ids = np.unique(self.subdomains.subdomains.array())
        self.assertEqual(set(subdomain_ids), set([0]))

    def test_setup_boundaries(self):
        self.subdomains.setup_subdomains(label_function=self.labels)
        self.subdomains.setup_boundaries(tissue_map=self.tissue_id_name_map)
        # Checks
        self.assertTrue(hasattr(self.subdomains, 'subdomain_boundaries'))
        self.assertTrue(hasattr(self.subdomains, 'subdomain_boundaries_id_dict'))
        boundary_ids = np.unique(self.subdomains.subdomain_boundaries.array())
        self.assertEqual(set(boundary_ids),set([2, 3]))
        self.assertEqual(set(self.subdomains.subdomain_boundaries_id_dict.values()),
                         set([0, 1, 2, 3]))

        self.subdomains.setup_boundaries(boundary_fct_dict=self.boundary_dict)
        self.assertTrue(hasattr(self.subdomains, 'named_boundaries'))
        self.assertTrue(hasattr(self.subdomains, 'named_boundaries_function_dict'))
        self.assertTrue(hasattr(self.subdomains, 'named_boundaries_id_dict'))

    def test_setup_measures(self):
        self.subdomains.setup_subdomains(label_function=self.labels)
        self.subdomains.setup_boundaries(tissue_map=self.tissue_id_name_map,
                                         boundary_fct_dict=self.boundary_dict)
        self.subdomains.setup_measures()
        self.assertTrue(hasattr(self.subdomains, 'dx'))
        self.assertTrue(hasattr(self.subdomains, 'ds'))
        self.assertTrue(hasattr(self.subdomains, 'dsn'))
        # number of boundary elements 'tissue_tumor' should equal number of cells in y direction
        tissue_tumor_boundary_id = self.subdomains.subdomain_boundaries_id_dict['tissue_tumor']
        self.assertEqual(sum(self.subdomains.ds.subdomain_data().array() == tissue_tumor_boundary_id),self.ny)
        # number of boundary elements 'boundary_1' should equal number of cells on boundary
        boundary_1_id = self.subdomains.named_boundaries_id_dict['boundary_1']
        self.assertEqual(sum(self.subdomains.dsn.subdomain_data().array() == boundary_1_id), 2*(self.nx+self.ny))

    def test_create_discontinuous_scalar_from_parameter_map(self):
        self.subdomains.setup_subdomains(label_function=self.labels)
        self.subdomains.setup_boundaries(tissue_map=self.tissue_id_name_map,
                                         boundary_fct_dict=self.boundary_dict)
        self.subdomains.create_discontinuous_scalar_from_parameter_map(self.parameter, 'diffusion')
        self.assertTrue(hasattr(self.subdomains, 'diffusion'))

    def test_get_subdomain_id(self):
        self.subdomains.setup_subdomains(label_function=self.labels)
        self.subdomains.setup_boundaries(tissue_map=self.tissue_id_name_map)
        self.assertEqual(self.subdomains.get_subdomain_id('tissue'), 1)

