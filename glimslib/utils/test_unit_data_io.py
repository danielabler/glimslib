import os
from unittest import TestCase
import glimslib.utils.data_io as dio
from glimslib import fenics_local as fenics, config, visualisation as plott
import  SimpleITK as sitk
import glimslib.utils.file_utils as fu


class FenicsImageInterface(TestCase):

    def setUp(self):
        self.nx = 40
        self.ny = 20
        mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), self.nx, self.ny)
        # function spaces
        U = fenics.VectorFunctionSpace(mesh, "Lagrange", 1)
        V = fenics.FunctionSpace(mesh, "Lagrange", 1)
        u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 1 ? (1.0) : (0.0)', degree=1,
                                          x0=1,
                                          y0=1)
        u_0_disp_expr = fenics.Constant((1.0, 1.0))
        self.conc = fenics.project(u_0_conc_expr, V)
        self.disp = fenics.project(u_0_disp_expr, U)
        # 3D
        mesh3d = fenics.BoxMesh(fenics.Point(-2, -2, -2), fenics.Point(2, 2, 2), 10, 20, 30)
        # function spaces
        U3 = fenics.VectorFunctionSpace(mesh3d, "Lagrange", 1)
        V3 = fenics.FunctionSpace(mesh3d, "Lagrange", 1)
        u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)+pow(x[2]-z0,2)) < 1 ? (1.0) : (0.0)', degree=1,
                                          x0=1, y0=1, z0=1            )
        u_0_disp_expr = fenics.Constant((1.0, 1.0, 1.0))
        self.conc3 = fenics.project(u_0_conc_expr, V3)
        self.disp3 = fenics.project(u_0_disp_expr, U3)
        self.test_path = os.path.join(config.output_dir_testing, 'test_data_io')
        fu.ensure_dir_exists(self.test_path)



    def test_function_to_image_2d_scalar(self):
        n_rep = 10
        img_list = []
        fun_list = []
        fun_list.append(self.conc)
        for i in range(1, n_rep):
            # write to image
            img = dio.create_image_from_fenics_function(fun_list[i - 1], size_new=None)
            path_to_img = os.path.join(self.test_path, 'image_from_function_2d_scalar_%i.nii' % i)
            sitk.WriteImage(img, path_to_img)
            img_list.append(img)
            # read from image
            img_read = sitk.ReadImage(path_to_img)
            function = dio.create_fenics_function_from_image(img_read)
            fun_list.append(function)
            # plot function
            path_to_fun_plot = os.path.join(self.test_path, 'conc_%i.png' % i)
            plott.show_img_seg_f(function=fun_list[i], show=True, path=path_to_fun_plot)
            # compare function with previous one
            self.assertLess(fenics.errornorm(fun_list[i - 1], fun_list[i]),1E-5)

    def test_function_to_image_2d_vector(self):
        n_rep = 10
        img_list = []
        fun_list = []
        fun_list.append(self.disp)
        for i in range(1, n_rep):
            # write to image
            img = dio.create_image_from_fenics_function(fun_list[i - 1], size_new=None)
            path_to_img = os.path.join(self.test_path, 'image_from_function_2d_vector_%i.nii' % i)
            sitk.WriteImage(img, path_to_img)
            img_list.append(img)
            # read from image
            img_read = sitk.ReadImage(path_to_img)
            function = dio.create_fenics_function_from_image(img_read)
            fun_list.append(function)
            # plot function
            path_to_fun_plot = os.path.join(self.test_path, 'disp_%i.png' % i)
            plott.show_img_seg_f(function=fun_list[i], show=True, path=path_to_fun_plot)
            # compare function with previous one
            self.assertLess(fenics.errornorm(fun_list[i - 1], fun_list[i]),1E-5)


    def test_function_to_image_3d_scalar(self):
        n_rep = 10
        img_list = []
        fun_list = []
        fun_list.append(self.conc3)
        for i in range(1, n_rep):
            # write to image
            img = dio.create_image_from_fenics_function(fun_list[i - 1], size_new=None)
            path_to_img = os.path.join(self.test_path, 'image_from_function_3d_scalar_%i.nii' % i)
            sitk.WriteImage(img, path_to_img)
            img_list.append(img)
            # read from image
            img_read = sitk.ReadImage(path_to_img)
            function = dio.create_fenics_function_from_image(img_read)
            fun_list.append(function)
            # plot function
            # path_to_fun_plot = os.path.join(config.output_dir_testing, 'conc_%i.png' % i)
            # plott.show_img_seg_f(function=fun_list[i], show=True, path=path_to_fun_plot)
            # compare function with previous one
            self.assertLess(fenics.errornorm(fun_list[i - 1], fun_list[i]),1E-5)

    def test_function_to_image_3d_vector(self):
        n_rep = 10
        img_list = []
        fun_list = []
        fun_list.append(self.disp3)
        for i in range(1, n_rep):
            # write to image
            img = dio.create_image_from_fenics_function(fun_list[i - 1], size_new=None)
            path_to_img = os.path.join(self.test_path, 'image_from_function_3d_vector_%i.nii' % i)
            sitk.WriteImage(img, path_to_img)
            img_list.append(img)
            # read from image
            img_read = sitk.ReadImage(path_to_img)
            function = dio.create_fenics_function_from_image(img_read)
            fun_list.append(function)
            # plot function
            # path_to_fun_plot = os.path.join(config.output_dir_testing, 'conc_%i.png' % i)
            # plott.show_img_seg_f(function=fun_list[i], show=True, path=path_to_fun_plot)
            # compare function with previous one
            self.assertLess(fenics.errornorm(fun_list[i - 1], fun_list[i]),1E-5)