from unittest import TestCase
from glimslib.simulation_helpers.helper_classes import FunctionSpace
from glimslib import fenics_local as fenics


class TestFunctionSpace(TestCase):

    def setUp(self):
        nx = ny = nz = 5
        self.mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        self.displacement_element = fenics.VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.concentration_element = fenics.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.element = fenics.MixedElement([self.displacement_element, self.concentration_element])

    def test_get_element(self):
        # single element
        subspace_name = 'displacement'
        functionspace_single = FunctionSpace(self.mesh)
        functionspace_single.init_function_space(self.displacement_element, subspace_name)
        self.assertEqual(functionspace_single.get_element(), self.displacement_element)
        self.assertEqual(functionspace_single.get_element(subspace_id=1), self.displacement_element)
        # multiple elements
        subspace_names = {0: 'displacement', 1: 'concentration'}
        functionspace_double = FunctionSpace(self.mesh)
        functionspace_double.init_function_space(self.element, subspace_names)
        self.assertEqual(functionspace_double.get_element(), self.element)
        self.assertEqual(functionspace_double.get_element(subspace_id=0), self.displacement_element)
        self.assertEqual(functionspace_double.get_element(subspace_id=1), self.concentration_element)

    def test_get_functionspace(self):
        # single element
        subspace_name = 'displacement'
        functionspace_single = FunctionSpace(self.mesh)
        functionspace_single.init_function_space(self.displacement_element, subspace_name)
        V = functionspace_single.get_functionspace()
        self.assertEqual(functionspace_single.get_functionspace(subspace_id=1), V)
        # multiple elements
        subspace_names = {0: 'displacement', 1: 'concentration'}
        functionspace_double = FunctionSpace(self.mesh)
        functionspace_double.init_function_space(self.element, subspace_names)
        V = functionspace_double.get_functionspace()
        self.assertNotEqual(functionspace_double.get_functionspace(1), V)

    def test_split_function(self):
        subspace_names = {0: 'displacement', 1: 'concentration'}
        functionspace = FunctionSpace(self.mesh)
        functionspace.init_function_space(self.element, subspace_names)

        u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.1 ? (1.0) : (0.0)', degree=1,
                                          x0=0.25,
                                          y0=0.5)
        u_0_disp_expr = fenics.Constant((0.0, 0.0))
        U_orig = functionspace.project_over_space(function_expr={0: u_0_disp_expr, 1: u_0_conc_expr})

        U = functionspace.split_function(U_orig)
        self.assertEqual(U_orig, U)
        U_1 = functionspace.split_function(U_orig, subspace_id=1)
        U_0 = functionspace.split_function(U_orig, subspace_id=0)


