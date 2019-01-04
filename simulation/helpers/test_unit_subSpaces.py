from unittest import TestCase

from simulation.helpers.helper_classes import SubSpaces

class TestSubSpaces(TestCase):

    def setUp(self):
        self.subspaces = SubSpaces({0:'subspace_0', 1:'subspace_1'})
        self.bcs = {'clamped': {'bc_value': 'testvalue',
                                     'boundary': 'testboundary',
                                     'subspace_id': 0},
                    'domain_all': {'boundary': 'testboundary',
                                          'boundary_id': 1,
                                          'subspace_id': 1},
                    'no_flux': {'bc_value': 'testvalue',
                                      'boundary_id': 1,
                                      'subspace_id': 1},
                    }

    def test_get_subspace_id(self):
        id = self.subspaces.get_subspace_id('subspace_1')
        self.assertEqual(id,1)

    def test_set_elements(self):
        content = {0:'for_subspace_0', 1:'for_subspace_1'}
        self.subspaces.set_elements(content)
        self.assertTrue(hasattr(self.subspaces, '_elements'))
        self.assertTrue(type(self.subspaces._elements)==dict)
        content = ['for_subspace_0', 'for_subspace_1']
        self.subspaces.set_elements(content, replace=True)
        self.assertTrue(hasattr(self.subspaces, '_elements'))
        self.assertTrue(type(self.subspaces._elements) == dict)

    def test_get_element(self):
        content = {0: 'for_subspace_0', 1: 'for_subspace_1'}
        self.subspaces.set_elements(content)
        el_subs_1 = self.subspaces.get_element(subspace_id=1)
        self.assertEqual(el_subs_1,'for_subspace_1')
        el_subs_2 = self.subspaces.get_element(subspace_id=2)
        self.assertEqual(el_subs_2,None)

    def test_set_inital_value_expressions(self):
        content = {0: 'for_subspace_0', 1: 'for_subspace_1'}
        self.subspaces.set_inital_value_expressions(content)
        self.assertTrue(hasattr(self.subspaces, '_inital_value_expressions'))
        self.assertTrue(type(self.subspaces._inital_value_expressions) == dict)
        content = ['for_subspace_0', 'for_subspace_1']
        self.subspaces.set_inital_value_expressions(content, replace=True)
        self.assertTrue(hasattr(self.subspaces, '_inital_value_expressions'))
        self.assertTrue(type(self.subspaces._inital_value_expressions) == dict)

    def test_get_inital_value_expression(self):
        content = {0: 'for_subspace_0', 1: 'for_subspace_1'}
        self.subspaces.set_inital_value_expressions(content)
        el_subs_1 = self.subspaces.get_inital_value_expression(subspace_id=1)
        self.assertEqual(el_subs_1, 'for_subspace_1')
        el_subs_2 = self.subspaces.get_inital_value_expression(subspace_id=2)
        self.assertEqual(el_subs_2, None)

    def test_set_functionspaces(self):
        content = {0: 'for_subspace_0', 1: 'for_subspace_1'}
        self.subspaces.set_functionspaces(content)
        self.assertTrue(hasattr(self.subspaces, '_functionspaces'))
        self.assertTrue(type(self.subspaces._functionspaces) == dict)
        content = ['for_subspace_0', 'for_subspace_1']
        self.subspaces.set_functionspaces(content, replace=True)
        self.assertTrue(hasattr(self.subspaces, '_functionspaces'))
        self.assertTrue(type(self.subspaces._functionspaces) == dict)

    def test_get_functionspace(self):
        content = {0: 'for_subspace_0', 1: 'for_subspace_1'}
        self.subspaces.set_functionspaces(content)
        el_subs_1 = self.subspaces.get_functionspace(subspace_id=1)
        self.assertEqual(el_subs_1, 'for_subspace_1')
        el_subs_2 = self.subspaces.get_functionspace(subspace_id=2)
        self.assertEqual(el_subs_2, None)

    def test_get_dirichlet_bcs(self):
        self.subspaces.set_dirichlet_bcs(self.bcs)
        subspace_1 = self.subspaces.get_dirichlet_bcs(subspace_id=1)
        self.assertTrue(len(subspace_1)==2)
        subspace_2 = self.subspaces.get_dirichlet_bcs(subspace_id=2)
        self.assertEqual(subspace_2, None)

    def test_get_von_neumann_bcs(self):
        self.subspaces.set_von_neumann_bcs(self.bcs)
        subspace_1 = self.subspaces.get_von_neumann_bcs(subspace_id=1)
        self.assertTrue(len(subspace_1)==2)
        subspace_2 = self.subspaces.get_von_neumann_bcs(subspace_id=2)
        self.assertEqual(subspace_2, None)

    def test_set_dirichlet_bcs(self):
        self.subspaces.set_dirichlet_bcs(self.bcs)
        self.assertTrue(hasattr(self.subspaces,'_bcs_dirichlet'))
        self.assertTrue(len(self.subspaces._bcs_dirichlet)==2)
        self.assertTrue(len(self.subspaces._bcs_dirichlet[1]) == 2)

    def test_set_von_neumann_bcs(self):
        self.subspaces.set_von_neumann_bcs(self.bcs)
        self.assertTrue(hasattr(self.subspaces, '_bcs_von_neumann'))
        self.assertTrue(len(self.subspaces._bcs_von_neumann) == 2)
        self.assertTrue(len(self.subspaces._bcs_von_neumann[1]) == 2)

    def test_project_over_subspace(self):
        import fenics_local as fenics
        nx = ny = nz = 10
        mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        E1 = fenics.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        V1 = fenics.FunctionSpace(mesh, E1)
        E2 = fenics.VectorElement("Lagrange", mesh.ufl_cell(), 1)
        V2 = fenics.FunctionSpace(mesh, E2)
        expr_1 = fenics.Expression(('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))'), degree=1, a=1, x0=0.0, y0=0.0)
        expr_2 = fenics.Expression(('x[0]', 'x[1]'), degree=1)
        self.subspaces.set_functionspaces([V1, V2])
        f_1 = self.subspaces.project_over_subspace(expr_1, 0)
        self.assertEqual(type(f_1),fenics.functions.function.Function)
        f_2 = self.subspaces.project_over_subspace(expr_2, 0)
        self.assertEqual(f_2, None)
        f_1 = self.subspaces.project_over_subspace(expr_1, 1)
        self.assertEqual(f_1, None)
        f_2 = self.subspaces.project_over_subspace(expr_2, 1)
        self.assertEqual(type(f_2), fenics.functions.function.Function)

