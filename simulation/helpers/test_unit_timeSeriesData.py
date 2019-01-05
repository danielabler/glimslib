from unittest import TestCase
import os

import fenics_local as fenics
from simulation.helpers.helper_classes import FunctionSpace, TimeSeriesDataTimePoint, TimeSeriesData


class TestTimeSeriesData(TestCase):

    def setUp(self):
        # Domain
        nx = ny = nz = 10
        mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        # function spaces
        displacement_element = fenics.VectorElement("Lagrange", mesh.ufl_cell(), 1)
        concentration_element = fenics.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        element = fenics.MixedElement([displacement_element, concentration_element])
        subspace_names = {0: 'displacement', 1: 'concentration'}
        functionspace = FunctionSpace(mesh)
        functionspace.init_function_space(element, subspace_names)
        # build a 'solution' function
        u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.1 ? (1.0) : (0.0)', degree=1,
                                          x0=0.25,
                                          y0=0.5)
        u_0_disp_expr = fenics.Constant((0.0, 0.0))
        self.U = functionspace.project_over_space(function_expr={0: u_0_disp_expr, 1: u_0_conc_expr})
        self.tsd = TimeSeriesData(functionspace=functionspace, name='solution')


    def test_add_observation(self):
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=1)
        self.assertEqual(len(self.tsd.data), 1)
        self.assertEqual(self.tsd.data.get(1).get_time(),1)

        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=1, replace=False)

        self.tsd.add_observation(field=self.U, time=1, time_step=2, recording_step=1, replace=True)
        self.assertEqual(self.tsd.data.get(1).get_time_step(), 2)
        self.assertEqual(len(self.tsd.data), 1)

        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=2, replace=False)
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=3, replace=False)
        self.assertEqual(len(self.tsd.data), 3)

    def test_get_observation(self):
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=1, replace=False)
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=2, replace=False)
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=3, replace=False)
        res = self.tsd.get_observation(2)
        self.assertEqual(res.get_recording_step(),2)
        res = self.tsd.get_observation(5)
        self.assertTrue(res is None)

    def test_get_most_recent_observation(self):
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=1, replace=False)
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=2, replace=False)
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=3, replace=False)
        res = self.tsd.get_most_recent_observation()
        self.assertEqual(res.get_recording_step(), 3)

    def test_get_solution_function(self):
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=1, replace=False)
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=2, replace=False)
        self.tsd.add_observation(field=self.U, time=1, time_step=1, recording_step=3, replace=False)
        u = self.tsd.get_solution_function(subspace_id=None, recording_step=2)
        u1 = self.tsd.get_solution_function(subspace_id=1, recording_step=2)
        u0 = self.tsd.get_solution_function(subspace_id=0, recording_step=2)
        self.assertEqual(u.function_space(), self.U.function_space())
        self.assertNotEqual(u, self.U)