from unittest import TestCase
from glimslib import fenics_local as fenics
from glimslib.simulation_helpers.helper_classes import Parameters, SubDomains, FunctionSpace, DiscontinuousScalar


class Boundary(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class TestSimulationParameters(TestCase):


    def setUp(self):
        # Domain
        nx = ny = nz = 10
        self.mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        # function spaces
        self.displacement_element = fenics.VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.concentration_element = fenics.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.element = fenics.MixedElement([self.displacement_element, self.concentration_element])
        subspace_names = {0: 'displacement', 1: 'concentration'}
        self.functionspace = FunctionSpace(self.mesh)
        self.functionspace.init_function_space(self.element, subspace_names)
        # subdomains
        label_funspace = fenics.FunctionSpace(self.mesh, "DG", 1)
        label_expr = fenics.Expression('(x[0]>=0) ? (1.0) : (2.0)', degree=1)
        labels = fenics.project(label_expr, label_funspace)
        self.labels = labels
        self.tissue_id_name_map = {0: 'outside',
                                   1: 'tissue',
                                   2: 'tumor'}
        self.parameter = {'outside': 0.0,
                          'tissue': 1.0,
                          'tumor': 0.1}
        self.boundary = Boundary()
        boundary_dict = {'boundary_1': self.boundary,
                         'boundary_2': self.boundary}
        self.boundary_dict = boundary_dict
        self.subdomains = SubDomains(self.mesh)
        self.subdomains.setup_subdomains(label_function=self.labels)
        self.subdomains.setup_boundaries(tissue_map=self.tissue_id_name_map,
                                         boundary_fct_dict=self.boundary_dict)
        self.subdomains.setup_measures()
        # parameter instance
        self.params = Parameters(self.functionspace, self.subdomains)


    def test_set_initial_value_expressions(self):
        u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.1 ? (1.0) : (0.0)',
                                          degree=1, x0=0.25, y0=0.5)
        u_0_disp_expr = fenics.Constant((0.0, 0.0))
        ivs = {1 : u_0_conc_expr, 0 : u_0_disp_expr}
        self.params.set_initial_value_expressions(ivs)
        self.assertEqual(self.params.get_iv(1), u_0_conc_expr)
        self.assertEqual(self.params.get_iv(0), u_0_disp_expr)

    def test_define_required_params(self):
        req_params = ['a', 'b', 'c']
        self.params = Parameters(self.functionspace, self.subdomains, time_dependent=False)
        self.params.define_required_params(req_params)
        self.assertEqual(sorted(req_params), sorted(self.params.params_required))
        self.params = Parameters(self.functionspace, self.subdomains, time_dependent=True)
        self.params.define_required_params(req_params)
        self.assertEqual(sorted(req_params+['sim_time', 'sim_time_step']),
                              sorted(self.params.params_required))

    def test_define_optional_params(self):
        opt_params = ['d', 'e', 'f']
        self.params = Parameters(self.functionspace, self.subdomains, time_dependent=False)
        self.params.define_optional_params(opt_params)
        self.assertEqual(sorted(opt_params), sorted(self.params.params_optional))

    def test_check_param_arguments(self):
        self.params = Parameters(self.functionspace, self.subdomains, time_dependent=False)
        req_params = ['a', 'b', 'c']
        self.params.define_required_params(req_params)
        kw_args = {'a' : 1, 'b': 2, 'c' : 3}
        test = self.params._check_param_arguments(kw_args)
        self.assertTrue(test)
        self.params = Parameters(self.functionspace, self.subdomains, time_dependent=True)
        req_params = ['a', 'b', 'c']
        self.params.define_required_params(req_params)
        kwargs = {'a': 1, 'b': 2, 'c': 3}
        test = self.params._check_param_arguments(kw_args)
        self.assertTrue(~test)

    def test_set_parameters(self):
        self.params = Parameters(self.functionspace, self.subdomains, time_dependent=False)
        req_params = ['a', 'b', 'c']
        self.params.define_required_params(req_params)
        opt_params = ['d']
        self.params.define_optional_params(opt_params)
        input_params = {'a':1, 'b' : self.parameter, 'c':1 ,'d':1, 'e':1}
        self.params.init_parameters( input_params)
        self.assertTrue(hasattr(self.params,'a'))
        self.assertTrue(hasattr(self.params, 'b'))
        self.assertTrue(hasattr(self.params, 'b_dict'))
        self.assertEqual(type(self.params.b), DiscontinuousScalar)
        self.assertTrue(hasattr(self.params, 'c'))
        self.assertTrue(hasattr(self.params, 'd'))
        self.assertFalse(hasattr(self.params, 'e'))

    def test_create_initial_value_function(self):
        self.params = Parameters(self.functionspace, self.subdomains, time_dependent=False)
        u_0_conc_expr = fenics.Expression('sqrt(pow(x[0]-x0,2)+pow(x[1]-y0,2)) < 0.1 ? (1.0) : (0.0)',
                                          degree=1, x0=0.25, y0=0.5)
        u_0_disp_expr = fenics.Constant((0.0, 0.0))
        ivs = {1: u_0_conc_expr, 0: u_0_disp_expr}
        self.params.set_initial_value_expressions(ivs)
        u = self.params.create_initial_value_function()


