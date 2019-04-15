from unittest import TestCase
from glimslib import fenics_local as fenics
from glimslib.simulation_helpers.helper_classes import BoundaryConditions, SubDomains, FunctionSpace


class BoundaryPos(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1]>0

class BoundaryNeg(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1]<0


class TestBoundaryConditions(TestCase):

    def setUp(self):
        # Domain
        nx = ny = nz = 10
        self.nx, self.ny = nx, ny
        self.mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny)
        # function spaces
        self.displacement_element = fenics.VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.concentration_element = fenics.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.element = fenics.MixedElement([self.displacement_element, self.concentration_element])
        subspace_names = {0: 'displacement', 1: 'concentration'}
        self.functionspace = FunctionSpace(self.mesh)
        self.functionspace.init_function_space(self.element, subspace_names)
        # define 'solution' with concentration=1 everywhere
        self.conc_expr =  fenics.Constant(1.0)
        self.conc = self.functionspace.project_over_space(self.conc_expr, subspace_name='concentration')
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
        self.boundary_pos = BoundaryPos()
        self.boundary_neg = BoundaryNeg()
        boundary_dict = {'boundary_pos': self.boundary_pos,
                         'boundary_neg': self.boundary_neg}
        self.boundary_dict = boundary_dict
        self.subdomains = SubDomains(self.mesh)
        self.subdomains.setup_subdomains(label_function=self.labels)
        self.subdomains.setup_boundaries(tissue_map=self.tissue_id_name_map,
                                         boundary_fct_dict=self.boundary_dict)
        self.subdomains.setup_measures()
        # BCs
        self.bcs = BoundaryConditions(self.functionspace, self.subdomains)
        self.dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant((0.0, 0.0)),
                                          'boundary': BoundaryPos(),
                                          'subspace_id': 0},
                              'clamped_1': {'bc_value': fenics.Constant((0.0, 0.0)),
                                            'subdomain_boundary': 'tissue_tumor',
                                            'subspace_id': 0},
                              'clamped_pos': {'bc_value': fenics.Constant((0.0, 0.0)),
                                            'named_boundary': 'boundary_pos',
                                            'subspace_id': 0},
                              'clamped_neg': {'bc_value': fenics.Constant((0.0, 0.0)),
                                            'named_boundary': 'boundary_neg',
                                            'subspace_id': 0}
                              }
        self.von_neuman_bcs = {
                               'flux_boundary_pos': {'bc_value': fenics.Constant(1.0),
                                           'named_boundary': 'boundary_pos',
                                           'subspace_id': 1},
                                'flux_boundary_neg': {'bc_value': fenics.Constant(-5.0),
                                                      'named_boundary': 'boundary_neg',
                                                      'subspace_id': 1}
                               # 'no_flux_domain_boundary': {'bc_value': fenics.Constant(1.0),
                               #               'subdomain_boundary': 'tissue_tumor',
                               #               'subspace_id': 1},
                              }

    def test_setup_dirichlet_boundary_conditions(self):
        self.bcs.setup_dirichlet_boundary_conditions(self.dirichlet_bcs)
        self.assertTrue(hasattr(self.bcs, 'dirichlet_bcs'))
        self.assertEqual(len(self.bcs.dirichlet_bcs),4)

    def test_setup_von_neumann_boundary_conditions(self):
        self.bcs.setup_von_neumann_boundary_conditions(self.von_neuman_bcs)
        self.assertTrue(hasattr(self.bcs, 'von_neumann_bcs'))
        self.assertEqual(len(self.bcs.von_neumann_bcs), 2)

    def test_implement_von_neumann_bcs(self):
        # seutp bcs
        self.bcs.setup_von_neumann_boundary_conditions(self.von_neuman_bcs)
        #v0, v1 = fenics.TestFunctions(self.functionspace.function_space)
        #param = self.subdomains.create_discontinuous_scalar_from_parameter_map(self.parameter, 'param')
        #bc_terms_0 = self.bcs.implement_von_neumann_bc(param * v0, subspace_id=0)
        #bc_terms_1 = self.bcs.implement_von_neumann_bc(param * v1, subspace_id=1)

        param = fenics.Constant(1.0)
        # compute surface integral based on implementation functions
        bc_boundary_auto_form = self.bcs.implement_von_neumann_bc(param * self.conc, subspace_id=1)
        bc_boundary_auto = fenics.assemble(bc_boundary_auto_form)
        # compute surface integral by defining form manually
        boundary_pos_id = self.subdomains.named_boundaries_id_dict['boundary_pos']
        boundary_neg_id = self.subdomains.named_boundaries_id_dict['boundary_neg']
        bc_boundary_pos_form = param * self.conc * fenics.Constant(1.0) * self.subdomains.dsn(boundary_pos_id)
        bc_boundary_neg_form = param * self.conc * fenics.Constant(-5.0) * self.subdomains.dsn(boundary_neg_id)
        bc_boundary = fenics.assemble(bc_boundary_pos_form) + fenics.assemble(bc_boundary_neg_form)
        self.assertAlmostEqual(bc_boundary_auto, bc_boundary)



# von Neumann BCs seem to apply only to mesh boundary, i.e. surface integral only for mesh boundary
# possibly this is because the surface integral involves a direction vector that is undefined for internal boundaries ?!
# see integral over internal boundaries: https://fenicsproject.org/qa/11837/integration-on-an-internal-boundary-ds/
# and over subdomain: https://fenicsproject.org/qa/4482/integrals-over-sub-domain/
# make test case for computation of von neumann BC acorss multiple parts of domain boundary
# https://fenicsproject.org/qa/6967/solving-a-pde-with-neumann-bc-between-subdomains/
