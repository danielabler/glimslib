from unittest import TestCase
import os
import numpy as np

import glimslib.utils.file_utils as fu
from glimslib import fenics_local as fenics, config
from glimslib.simulation_helpers.helper_classes import FunctionSpace, TimeSeriesMultiData


class TestTimeSeriesMultiData(TestCase):

    def setUp(self):
        # Domain
        nx = ny = nz = 10
        mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        # function spaces
        displacement_element = fenics.VectorElement("Lagrange", mesh.ufl_cell(), 1)
        concentration_element = fenics.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        element = fenics.MixedElement([displacement_element, concentration_element])
        subspace_names = {0: 'displacement', 1: 'concentration'}
        self.functionspace = FunctionSpace(mesh)
        self.functionspace.init_function_space(element, subspace_names)
        # build a 'solution' function
        u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.1 ? (1.0) : (0.0)', degree=1,
                                          x0=0.25,
                                          y0=0.5)
        u_0_disp_expr = fenics.Constant((1.0, 0.0))
        self.U = self.functionspace.project_over_space(function_expr={0: u_0_disp_expr, 1: u_0_conc_expr})

    def test_register_time_series(self):
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        self.assertTrue(hasattr(tsmd, tsmd.time_series_prefix+'solution'))
        tsmd.register_time_series(name='solution2', functionspace=self.functionspace)
        self.assertTrue(hasattr(tsmd, tsmd.time_series_prefix+'solution2'))

    def test_get_time_series(self):
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        self.assertEqual(tsmd.get_time_series('solution'), getattr(tsmd, tsmd.time_series_prefix+'solution'))

    def test_add_observation(self):
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=1)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=1)
        self.assertEqual(tsmd.get_time_series('solution').get_observation(1).get_time_step(), 1)

    def test_get_observation(self):
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=1)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=1)
        self.assertEqual(tsmd.get_time_series('solution').get_observation(1),
                         tsmd.get_observation('solution', 1))
        self.assertEqual(tsmd.get_observation('solution3', 1), None)

    def test_get_solution_function(self):
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=1)
        u = tsmd.get_solution_function('solution', subspace_id=None, recording_step=1)
        self.assertEqual(u.function_space(), self.U.function_space())
        self.assertNotEqual(u, self.U)

    def test_get_all_time_series(self):
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        tsmd.register_time_series(name='solution2', functionspace=self.functionspace)
        ts_dict = tsmd.get_all_time_series()
        self.assertEqual(len(ts_dict), 2)
        self.assertTrue('solution' in ts_dict.keys())
        self.assertTrue('solution2' in ts_dict.keys())

    def test_save_to_hdf5(self):
        path_to_file = os.path.join(config.output_dir_testing, 'timeseries_to_hdf5.h5')
        fu.ensure_dir_exists(path_to_file)
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=1)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=2)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=3)
        tsmd.register_time_series(name='solution2', functionspace=self.functionspace)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=1)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=2)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=3)
        tsmd.save_to_hdf5(path_to_file, replace=True)
        # path_to_file2 = os.path.join(config.output_dir_testing, 'timeseries_to_hdf5_manual.h5')
        # hdf = fenics.HDF5File(self.functionspace._mesh.mpi_comm(), path_to_file2, "w")
        # function = tsmd.get_solution_function('solution', recording_step=1)
        # hdf.write(function, 'solution', 1)
        # hdf.write(function, 'solution', 2)
        # hdf.close()

    def test_load_from_hdf5(self):
        path_to_file = os.path.join(config.output_dir_testing, 'timeseries_to_hdf5_for_reading.h5')
        fu.ensure_dir_exists(path_to_file)
        # create file
        tsmd = TimeSeriesMultiData()
        tsmd.register_time_series(name='solution', functionspace=self.functionspace)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=1)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=2)
        tsmd.add_observation('solution', field=self.U, time=1, time_step=1, recording_step=3)
        tsmd.register_time_series(name='solution2', functionspace=self.functionspace)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=1)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=2)
        tsmd.add_observation('solution2', field=self.U, time=1, time_step=1, recording_step=3)
        tsmd.save_to_hdf5(path_to_file, replace=True)
        # read file
        tsmd2 = TimeSeriesMultiData()
        tsmd2.register_time_series(name='solution', functionspace=self.functionspace)
        tsmd2.register_time_series(name='solution2', functionspace=self.functionspace)
        tsmd2.load_from_hdf5(path_to_file)
        self.assertEqual(len(tsmd2.get_all_time_series()),2)
        self.assertEqual(len(tsmd2.get_time_series('solution').get_all_recording_steps()),3)
        self.assertEqual(len(tsmd2.get_time_series('solution2').get_all_recording_steps()), 3)
        u_reloaded = tsmd2.get_solution_function(name='solution')
        # print(u_reloaded.vector().array())
        # print(self.U.vector().array())
        array_1 = u_reloaded.vector().get_local()
        array_2 = self.U.vector().get_local()
        self.assertTrue(np.allclose(array_1, array_2))



