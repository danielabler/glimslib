from unittest import TestCase
import os

from glimslib import fenics_local as fenics
from glimslib.simulation_helpers.helper_classes import FunctionSpace, Results


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
        self.results = Results(functionspace, subdomains=None)

    def test_add_to_results(self):
        self.results.add_to_results(current_sim_time=1,current_time_step=1,recording_step=1,
                                    field=self.U)
        self.assertTrue(hasattr(self.results, 'data'))
        self.assertEqual(self.results.data.get_time_series(self.results.ts_name).get_observation(1).get_time_step(), 1)
        self.results.add_to_results(current_sim_time=1, current_time_step=2, recording_step=1,
                                     field=self.U, replace=False)
        self.assertEqual(self.results.data.get_time_series(self.results.ts_name).get_observation(1).get_time_step(), 1)
        self.results.add_to_results(current_sim_time=1, current_time_step=2, recording_step=1,
                                    field=self.U, replace=True)
        self.assertEqual(self.results.data.get_time_series(self.results.ts_name).get_observation(1).get_time_step(), 2)
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=2,
                                     field=self.U)
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=3,
                                     field=self.U)
        self.assertEqual(len(self.results.data.get_all_recording_steps(self.results.ts_name)), 3)

    def test_get_result(self):
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=1,
                                    field=self.U)
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=2,
                                    field=self.U)
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=3,
                                    field=self.U)
        res = self.results.get_result(2)
        self.assertEqual(res.get_recording_step(),2)
        res = self.results.get_result(5)
        self.assertTrue(res is None)

    def test_get_solution_function(self):
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=1,
                                    field=self.U)
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=2,
                                    field=self.U)
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=3,
                                    field=self.U)
        u = self.results.get_solution_function(subspace_id=None, recording_step=2)
        u1 = self.results.get_solution_function(subspace_id=1, recording_step=2)
        u0 = self.results.get_solution_function(subspace_id=0, recording_step=2)


    def test_save_solution(self):
        self.results.add_to_results(current_sim_time=1, current_time_step=1, recording_step=1,
                                    field=self.U)
        method = 'vtk'
        self.results.save_solution_start(method, clear_all=True)
        self.results.save_solution(recording_step=1, time=1, function=self.U, method=method)
        self.results.save_solution(recording_step=2, time=10, function=self.U, method=method)
        self.results.save_solution_end(method)
        self.assertTrue(os.path.isfile(os.path.join(self.results.output_dir, 'concentration', 'concentration_00001.pvd')))
        self.assertTrue(os.path.isfile(os.path.join(self.results.output_dir, 'concentration', 'concentration_00002.pvd')))
        method = 'xdmf'
        self.results.save_solution_start(method, clear_all=True)
        self.results.save_solution(recording_step=1, time=1, function=self.U, method=method)
        self.results.save_solution(recording_step=2, time=10, function=self.U, method=method)
        self.results.save_solution_end(method)
        self.assertTrue(
            os.path.isfile(os.path.join(self.results.output_dir, 'solution.h5')))


