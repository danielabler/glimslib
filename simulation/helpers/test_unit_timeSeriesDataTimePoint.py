from unittest import TestCase
import os

import fenics_local as fenics
from simulation.helpers.helper_classes import FunctionSpace, TimeSeriesDataTimePoint


class TestResults(TestCase):

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

    def test_set_field(self):
        observation = TimeSeriesDataTimePoint(time=1.05, time_step=2, recording_step=1)
        observation.set_field(self.U)
        self.assertTrue(hasattr(observation, 'field'))
        self.assertEqual(observation.field, self.U)

    def test_get_field(self):
        observation = TimeSeriesDataTimePoint(time=1.05, time_step=2, recording_step=1)
        observation.set_field(self.U)
        self.assertTrue(hasattr(observation, 'field'))
        self.assertEqual(observation.get_field(), self.U)

    def test_get_time(self):
        observation = TimeSeriesDataTimePoint(time=1.05, time_step=2, recording_step=1)
        self.assertEqual(observation.get_time(), 1.05)

    def test_get_time_step(self):
        observation = TimeSeriesDataTimePoint(time=1.05, time_step=2, recording_step=1)
        self.assertEqual(observation.get_time_step(), 2)

    def test_get_recording_step(self):
        observation = TimeSeriesDataTimePoint(time=1.05, time_step=2, recording_step=1)
        self.assertEqual(observation.get_recording_step(), 1)