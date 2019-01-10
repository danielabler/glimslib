import logging
import sys
import itertools
import numpy as np
import collections
import copy
import os
import shutil
import shelve
from abc import ABC, abstractmethod

import pandas as pd

import fenics_local as fenics
import simulation.helpers.math_linear_elasticity

import utils.file_utils as fu
import utils.data_io as dio
import simulation.config as config
import visualisation.plotting as plott
import visualisation.helpers as vh

import simulation.helpers.math_linear_elasticity as mle
from simulation.helpers import math_linear_elasticity as mle, math_reaction_diffusion as mrd


class AnyDimPoint(fenics.Point):
    """
    This class extends `Point` with a more convenient constructor for 1D/2D/3D.
    From https://github.com/geo-fluid-dynamics/phaseflow-fenics
    :param coordinates: tuple of floats
    """

    def __init__(self, coordinates):

        if type(coordinates) is type(0.):
            coordinates = (coordinates,)

        if len(coordinates) == 1:

            fenics.Point.__init__(self, coordinates[0])

        elif len(coordinates) == 2:

            fenics.Point.__init__(self, coordinates[0], coordinates[1])

        elif len(coordinates) == 3:

            fenics.Point.__init__(self, coordinates[0], coordinates[1], coordinates[2])

class DiscontinuousScalar(fenics.Expression):
    """
    Creates scalar with different values in each subdomains.
    """
    def __init__(self, cell_function, scalars, **kwargs):
        self.cell_function = cell_function
        self.coeffs = scalars

    def eval_cell(self, values, x, cell):
        subdomain_id = self.cell_function[cell.index]
        local_coeff = self.coeffs[subdomain_id]
        local_coeff.eval_cell(values, x, cell)

class SubSpaces():
    """
    Helper class for management of Fenics functions that live on multiple subspaces.
    The following attributes are foreseen:

    - available finite elements  `self._elements = {subspace_id : subspace element}`
    - available function spaces `self._functionspaces = { }`
    - initial value expressions `self._inital_value_expressions = { }`
    - Dirichlet BCs by subspace `self._bcs_dirichlet = { }`
    - Von Neumann BCs by subspace `self._bcs_von_neumann = { }`
    """

    def __init__(self, names={}):
        """
        Init routine.
        :param names: dictionary mapping subspace ids to names `{ subspace id : subspace name}`.
        """
        self.logger = logging.getLogger(__name__)

        self.names = names
        self.n = len(self.names)
        self._attribute_prefix = '_'

    def get_subspace_names(self):
        return self.names.values()

    def get_subspace_name(self, subspace_id):
        return self.names.get(subspace_id)

    def get_subspace_ids(self):
        return self.names.keys()

    def get_subspace_id(self, subspace_name):
        """
        Returns id for subspace of name `subspace_name`.
        :param subspace_name: name of subspace
        :return: subspace id
        """
        subspace_name_dict = {value: key for key, value in self.names.items()}
        if subspace_name in subspace_name_dict.keys():
            subspace_id = subspace_name_dict[subspace_name]
        else:
            self.logger.warning("Functionspace does not have '%s' subspace." % (subspace_name))
            subspace_id = None
        return subspace_id

    def _set_subspace_attribute(self, name, content, replace=False):
        """
        Sets subspace attribute.
        :param name: attribute name
        :param content: attribute content
        :param replace: boolean flag, indicating whether an existing attribute should be replaced.
        """
        internal_name = self._attribute_prefix + name
        if (type(content) is not list) and (type(content) is not dict):
            self.logger.error('Expect either list or dictionary')
        elif len(content) != self.n:
            self.logger.error('Expect content with %i items, but this has %i items.'%(self.n, len(content)))
        else:
            if type(content) is list:
                dictionary = { i : item for i, item in enumerate(content) }
            else:
                dictionary = content

            if not hasattr(self, internal_name):
                setattr(self, internal_name, dictionary)
            else:
                self.logger.warning("Attribute '%s' already exists."%internal_name)
                if replace:
                    self.logger.warning("... replacing existing attribute '%s'." % internal_name)
                    setattr(self, internal_name, dictionary)
                else:
                    self.logger.warning("... do nothing.")


    def _get_subspace_attribute(self, name, subspace_id=None, subspace_name=None):
        """
        Retrieves content of attribute `name` by `subspace_id`, or by `subspace_name`.
        :param name: attribute name
        :param subspace_id: id
        :param subspace_name: name
        :return: content of attribute `name`.
        """
        if (subspace_id is None) and (subspace_name is None):
            self.logger.error("No subspace or subspace name specified")
        else:
            if subspace_id is None:
                subspace_id = self.get_subspace_id(subspace_name)
        internal_name = self._attribute_prefix + name
        if hasattr(self, internal_name):
            attribute = getattr(self, internal_name)
            if subspace_id in attribute.keys():
                return attribute[subspace_id]
            else:
                self.logger.warning("Attribute '%s' has no information for subspace '%i'"%(name, subspace_id))
        else:
            self.logger.warning("Attribute '%s' does not exist.")
        return None

    def get_element(self, subspace_id=None, subspace_name=None):
        return self._get_subspace_attribute(name='elements', subspace_id=subspace_id, subspace_name=subspace_name)

    def get_inital_value_expression(self, subspace_id=None, subspace_name=None):
        return self._get_subspace_attribute(name='inital_value_expressions', subspace_id=subspace_id, subspace_name=subspace_name)

    def get_functionspace(self, subspace_id=None, subspace_name=None):
        return self._get_subspace_attribute(name='functionspaces', subspace_id=subspace_id, subspace_name=subspace_name)

    def get_dirichlet_bcs(self, subspace_id=None, subspace_name=None):
        return self._get_subspace_attribute(name='bcs_dirichlet', subspace_id=subspace_id, subspace_name=subspace_name)

    def get_von_neumann_bcs(self, subspace_id=None, subspace_name=None):
        return self._get_subspace_attribute(name='bcs_von_neumann', subspace_id=subspace_id, subspace_name=subspace_name)

    def set_elements(self, content, replace=False):
        self._set_subspace_attribute('elements', content, replace=replace)

    def set_inital_value_expressions(self, content, replace=False):
        self._set_subspace_attribute('inital_value_expressions', content, replace=replace)

    def set_functionspaces(self, content, replace=False):
        self._set_subspace_attribute('functionspaces', content, replace=replace)


    def _rearrange_dict_by_subspace(self, dict_in):
        """
        Sorts dictrionary by value of entry 'subspace_id'.
        """
        dict_by_subspace = {}
        for subspace_id, name in self.names.items():
            dict_by_subspace[subspace_id] = []
            for name, item_dict in dict_in.items():
                item_subspace_id = item_dict.get('subspace_id')
                if item_subspace_id == subspace_id:
                    item_dict['name'] = name
                    dict_by_subspace[subspace_id].append(item_dict)
        return dict_by_subspace

    def set_dirichlet_bcs(self, content, replace=False):
        dict_by_subspace = self._rearrange_dict_by_subspace(content)
        self._set_subspace_attribute('bcs_dirichlet', dict_by_subspace, replace=replace)

    def set_von_neumann_bcs(self, content, replace=False):
        dict_by_subspace = self._rearrange_dict_by_subspace(content)
        self._set_subspace_attribute('bcs_von_neumann', dict_by_subspace, replace=replace)

    def project_over_subspace(self, function_expr, subspace_id=None, subspace_name=None, **kwargs):
        """
        Projects function or expression over subspace.
        `**kwargs` can be used to inject projection parameters, such as

            - solver_type="cg", preconditioner_type="amg"
        """
        if (subspace_id is None) and (subspace_name is None):
            self.logger.error("No subspace or subspace name specified")
        else:
            if subspace_id is None:
                subspace_id = self.get_subspace_id(subspace_name)
            funspace_sub = self.get_functionspace(subspace_id=subspace_id)
            try:
                self.logger.info('Trying to project over subspace %i'%subspace_id)
                function = fenics.project(function_expr, funspace_sub, **kwargs)
                return function
            except:
                self.logger.warning("Cannot project functions over subspace %i:"%subspace_id)
                self.logger.warning(sys.exc_info()[0])
                return None

class FunctionSpace():
    """
    Helper class for management of Fenics function space.
    """

    def __init__(self, mesh, projection_parameters={}):
        """
        Init routine.
        :param element: fenics element or dictionary of { subspace name : fenics element}
        :param projection_parameters: parameters for projection of functions. E.g. solver_type="cg", preconditioner_type="amg"
        """
        self.logger = logging.getLogger(__name__)
        self._mesh = mesh
        self._projection_parameters = projection_parameters

    def init_function_space(self, element, name):
        self.element = element
        self._init_subspaces(name)
        self._setup_function_space()

    def _init_subspaces(self, name):
        """
        Checks if mixed function space and initialises instance of SubSpaces
        """
        if type(name) == dict:
            self.has_subspaces = True
            self.subspaces = SubSpaces(name)
            self.subspaces.set_elements(self.element.sub_elements())

        else:
            self.has_subspaces = False
            self.name = name

    def _setup_function_space(self):
        """ Set the function space from self._mesh and self.element. """
        self.logger.info("   - setting up global function space")
        self.function_space = fenics.FunctionSpace(self._mesh, self.element)
        # Also create global functionspaces for subspace elements -- needed for adjoint
        if self.has_subspaces:
            subspace_functionspaces = {}
            for subspace_id, subspace_name in self.subspaces.names.items():
                element = self.subspaces.get_element(subspace_id=subspace_id)
                functionspace = fenics.FunctionSpace(self._mesh, element)
                subspace_functionspaces[subspace_id] = functionspace
            self.subspaces.set_functionspaces(subspace_functionspaces)

    def get_element(self, subspace_id=None, subspace_name=None):
        if self.has_subspaces:
            if (subspace_id is None) and (subspace_name is None):
                self.logger.warning("No subspaces specified ... returning main element")
                element = self.element
            else:
                element = self.subspaces.get_element(subspace_id=subspace_id, subspace_name=subspace_name)
        else:
            element = self.element

        return element

    def get_functionspace(self, subspace_id=None, subspace_name=None):
        """
        Returns a separate instance of the subspace obtained by creating fenics.FunctionSpace(Element) from
        respective subspace element.
        This function should be used for plotting on subspaces.
        :param subspace_id:
        :param subspace_name:
        :return:
        """
        if self.has_subspaces:
            if (subspace_id is None) and (subspace_name is None):
                self.logger.warning("No subspaces specified ... returning main functionspace")
                functionspace = self.function_space
            else:
                functionspace = self.subspaces.get_functionspace(subspace_id=subspace_id, subspace_name=subspace_name)
        else:
            functionspace = self.function_space

        return functionspace

    def get_functionspace_orig_subspace(self, subspace_id=None, subspace_name=None):
        """
        Returns actual subspace of main functionspace: self.function_space.sub(subspace_id)
        This function should be used e.g. for initialization of boundary conditions.
        :param subspace_id: subspace id
        :param subspace_name: subspace name
        :return: reference to function_space.sub(subspace_id)
        """
        if (subspace_id is None) and (subspace_name is None):
            self.logger.error("No subspace or subspace name specified")
        else:
            if subspace_id is None:
                subspace_id = self.get_subspace_id(subspace_name)
        subspace = self.function_space.sub(subspace_id)
        return subspace

    def project_over_space(self, function_expr, subspace_id=None, subspace_name=None, **kwargs):
        if self.has_subspaces:
            if type(function_expr)==dict:
                self.logger.info("Assembling function over subspaces")
                function = self._project_combine_multiple_subspaces(function_expr_subspace_dict=function_expr, **kwargs)
            else:
                if (subspace_id is None) and (subspace_name is None):
                    self.logger.info("Projecting over main function space")
                    function = fenics.project(function_expr, self.function_space, **self._projection_parameters, **kwargs)
                else:
                    function = self.subspaces.project_over_subspace(function_expr,
                                                                subspace_id=subspace_id, subspace_name=subspace_name,
                                                                    **self._projection_parameters, **kwargs)
        else:
            function = fenics.project(function_expr, self.function_space, **self._projection_parameters)
        return function

    def _project_combine_multiple_subspaces(self, function_expr_subspace_dict, **kwargs):
        U = fenics.Function(self.function_space)
        for subspace, function_expr in function_expr_subspace_dict.items():
            if type(subspace)==str:
                subspace_id = self.subspaces.names.get(subspace)
            else:
                subspace_id = subspace
            f = self.project_over_space(function_expr, subspace_id=subspace_id, **kwargs)
            V = self.subspaces.get_functionspace(subspace_id=subspace_id)
            assigner = fenics.FunctionAssigner(self.function_space.sub(subspace_id), V)
            assigner.assign(U.sub(subspace_id), f)
        return U

    def split_function(self, function, subspace_id=None, subspace_name=None):
        """
        Splits function into subspace functions. Does not project onto other functionspace.
        Returns function of subspace `subspace_id` if functionspace has subspaces, or original function otherwise.
        :param function:
        :param subspace_id:
        :param subspace_name:
        :return: original
        """
        if self.has_subspaces:
            if (subspace_id is None) and (subspace_name is None):
                self.logger.info("No subspace specified, returning entire function")
                function_sub = function
            else:
                function_split = fenics.split(function)
                if (subspace_id is None) and (subspace_name is not None):
                    subspace_id = self.subspaces.get_subspace_id(subspace_name=subspace_name)
                function_sub = function_split[subspace_id]
        else:
            self.logger.info("Only single functionspace -- Returning entire function")
            function_sub = function
        return function_sub

class SubDomains():
    """
    Helper class for management of subdomains in Fenics.
    """

    def __init__(self, mesh):
        """
        Init routine.
        :param n: number of subspace
        """
        self.logger = logging.getLogger(__name__)
        self._mesh = mesh

    def setup_subdomains(self, label_function=None, subdomains=None, replace=False):
        """
        Creates self.subdomains containing fenics.MeshFunction() which indicates domain subdomains.
        This MeshFunction can be provided directly, or be generated from a fenics function.
        :param label_function: fenics function indicating subdomain membership (node)
        :param subdomains: MeshFunction indicating subdomain membership (cell)
        """
        if not hasattr(self, 'subdomains'):
            if subdomains is not None:
                self.subdomains = subdomains
            elif label_function is not None:
                self._setup_subdomains_from_labelmapfunction(label_function)
            else:
                self.subdomains = fenics.MeshFunction("size_t", self._mesh, self._mesh.geometry().dim())
                self.subdomains.set_all(0)
        else:
            self.logger.warning("'subdomains' already exists.")
            if replace:
                self.logger.warning("... replacing existing 'subdomains'.")
                if subdomains is not None:
                    self.subdomains = subdomains
                elif label_function is not None:
                    self._setup_subdomains_from_labelmapfunction(label_function)
                else:
                    self.subdomains = fenics.MeshFunction("size_t", self._mesh, self._mesh.geometry().dim())
                    self.subdomains.set_all(0)
            else:
                self.logger.warning("... do nothing.")

    def _setup_subdomains_from_labelmapfunction(self, label_function):
        """
        Creates subdomains MeshFunction from labelmap functions.
        :param label_function:   A fenics function defining labels over the domain.
        """
        self.label_function = label_function
        self.logger.info("   - Creating SubDomains from labelmap")
        subdomains = fenics.MeshFunction("size_t", self._mesh, self._mesh.geometry().dim())
        subdomains.set_all(0)
        # mark subdomains
        for cell in fenics.cells(self._mesh):
            subdomains[cell.index()] = int(self.label_function(cell.midpoint()))
        self.subdomains = subdomains
        self.logger.info("     ... created subdomains.")

    def setup_boundaries(self, tissue_map=None, boundary_fct_dict=None):
        """
        Create domain boundaries from tissue_map and other named boundaries from boundary_fct_dict.
        :param tissue_map:
        :param boundary_fct_dict:
        """
        if tissue_map is not None:
            self._setup_boundaries_from_subdomains(tissue_map)
        if boundary_fct_dict is not None:
            self._setup_boundaries_from_functions(boundary_fct_dict)

    def _setup_boundaries_from_subdomains(self, tissue_id_name_map):
        """
        Creates boundaries fenics.MeshFunction() from subdomain information
        :param tissue_id_name_map: dictionary of form `{ tissue_id : tissue_name }`
        """
        self.logger.info("   - Creating Boundaries from tissue_id_name_map")
        if hasattr(self, 'subdomains'):
            self.tissue_id_name_map = tissue_id_name_map
            # Define boundaries
            boundaries = fenics.MeshFunction('size_t', self._mesh, self._mesh.geometry().dim() - 1)

            # generate all possible boundaries
            boundary_types = list(itertools.combinations(self.tissue_id_name_map.keys(), 2))
            boundary_names = list(itertools.combinations(self.tissue_id_name_map.values(), 2))
            boundary_names_string = list(map('_'.join, boundary_names))
            boundary_type_dict = dict(zip(boundary_types, boundary_names_string))
            boundary_id_dict = dict(zip(boundary_names_string, range(len(boundary_type_dict.keys()))))

            value_no_boundary = max(boundary_id_dict.values()) + 1
            # Assign values from boundary_id_dict as boundary ids
            ids = []
            id_list = []
            for f in fenics.facets(self._mesh):
                domains = []
                for c in fenics.cells(f):
                    domains.append(self.subdomains[c])  # this gives the subdomain_id of the current cell
                domains = list(set(domains))
                if len(domains) > 1:
                    for boundary_type, boundary_name in boundary_type_dict.items():
                        if tuple(domains) == boundary_type:
                            boundaries[f] = boundary_id_dict[boundary_name]
                            ids.append(boundary_id_dict[boundary_name])
                            id_list.append((f, boundary_id_dict[boundary_name]))
                else:
                    boundaries[f] = value_no_boundary

            boundary_id_dict['no_boundary'] = value_no_boundary

            self.subdomain_boundaries = boundaries
            self.subdomain_boundaries_id_dict = boundary_id_dict

            self.logger.info("     ... found boundaries %s" %
                             (np.unique(self.subdomain_boundaries.array())))
        else:
            self.logger.warning("Need subdomains to define boundaries. No subdomains defined.")

    def _setup_boundaries_from_functions(self, boundary_dict):
        """
        Creates boundaries fenics.MeshFunction() from functions.

        :param boundary_dict: dictionary of form `{boundary_name : boundary}`.

        Where boundary is a function or expression that defines a boundary, such as::

            class BoundaryR(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and x[0] >= (1.0 - DOLFIN_EPS)
        """
        self.logger.info("   - Creating Boundaries from boundary dictionary")
        boundaries = fenics.MeshFunction('size_t', self._mesh, self._mesh.geometry().dim() - 1)
        boundaries.set_all(0)
        boundary_id = 0
        boundary_id_dict = {}
        for name, boundary_def in boundary_dict.items():
            boundary_id = boundary_id + 1
            self.logger.info("      - boundary '%s' with id=%d" % (name, boundary_id))
            boundary_def.mark(boundaries, boundary_id)
            boundary_id_dict[name] = boundary_id

        self.named_boundaries_id_dict = boundary_id_dict
        self.named_boundaries_function_dict = boundary_dict
        self.named_boundaries = boundaries

    def add_named_boundaries(self, boundary_dict):
        """
        Appends named boundaries.
        :param boundary_dict:
        :return:
        """
        # TODO extend function to allow adding new named boundaries after initialisation
        pass

    def setup_measures(self):
        """
        Redefine measures based on subdomains and boundaries.
        Requires self.subdomains and self.boundaries to be present.
        """
        self.logger.info("   - setting up measures dx, ds, dsn")
        if hasattr(self, 'subdomains'):
            self.dx = fenics.dx(subdomain_data=self.subdomains)
        else:
            self.logger.warning("Need 'subdomains' to redefine 'dx' measure. Use standard measure")
            self.dx = fenics.dx

        if hasattr(self, 'subdomain_boundaries'):
            self.ds = fenics.ds(subdomain_data=self.subdomain_boundaries)
        else:
            self.logger.warning(
                "Need 'subdomain_boundaries' to redefine 'ds' measure. Use standard measure")
            self.ds = fenics.ds

        if hasattr(self, 'named_boundaries'):
            self.dsn = fenics.ds(subdomain_data=self.named_boundaries)
        else:
            self.logger.warning("Need 'named_boundaries' to redefine 'ds' measure. Use standard measure")
            self.dsn = fenics.ds

    def _create_param_list_for_discontinuous_scalar(self, param_dict):
        if hasattr(self, 'tissue_id_name_map'):
            mapping_dict = self.tissue_id_name_map
            mapdict_ordered = collections.OrderedDict(mapping_dict)
            param_list = []
            if not 0 in mapdict_ordered.keys():
                param_list.append(fenics.Constant(0))
            for material_id, material_name in mapdict_ordered.items():
                mat_prop = param_dict[material_name]
                param_list.append(fenics.Constant(mat_prop))
            return param_list
        else:
            self.logger.warning("No subdomains have been defined, cannot assign parameter values")

    def create_discontinuous_scalar_from_parameter_map(self, param_dict, name, replace=False):
        """
        Creates an instance of `DiscontinuousScalar` from dictionary {<subdomain_name> : value},
        where <subdomain_name> is the name of one of the problems subdomains as defined in
        `self.tissue_id_name_map`.
        :param param_dict: dictionary {<subdomain_name> : value}
        :param name: name of parameter
        :param replace: boolean flag indicating whether already existing parameter should be replaced
        :return: instance of `DiscontinuousScalar`
        """
        disc_param = None
        param_list = self._create_param_list_for_discontinuous_scalar(param_dict)
        if hasattr(self, 'subdomains'):
            if not hasattr(self, name):
                disc_param = DiscontinuousScalar(self.subdomains, param_list, degree=1)
                setattr(self, name, disc_param)
            else:
                self.logger.warning("Parameter '%s' already exists."%name)
                if replace:
                    disc_param = DiscontinuousScalar(self.subdomains, param_list, degree=1)
                    setattr(self, name, disc_param)
                else:
                    self.logger.warning("... do nothing.")
        else:
            self.logger.warning("No subdomains have been defined, cannot assign parameter values")
        return disc_param

    def _create_tissue_name_id_map(self):
        if hasattr(self, 'tissue_id_name_map'):
           self.tissue_name_id_map = { value : key for key, value in self.tissue_id_name_map.items()}

    def get_subdomain_id(self, subdomain_name):
        if not hasattr(self, 'tissue_name_id_map'):
            self._create_tissue_name_id_map()
        if subdomain_name in self.tissue_name_id_map.keys():
            return self.tissue_name_id_map.get(subdomain_name)
        else:
            self.logger.error("Subdomain '%s' does not exist"%(subdomain_name))


class BoundaryConditions():
    """
    Helper class for management of Fenics Boundary Conditions.
    """
    def __init__(self, functionspace, subdomains):
        """
        Init routine.
        :param functionspace: Instance of FunctionSpace
        :param subdomains: Instance of SubDomains
        """
        self.logger = logging.getLogger(__name__)
        self._functionspace = functionspace
        self._subdomains = subdomains

    def setup_dirichlet_boundary_conditions(self, dirichlet_bcs={}):
        """
        Creates fenics Dirichlet Boundary conditions based from dictionary of boundary specifications::

            bcs = { 'description' :
                  {  dirichlet_bc_specification_dict  }}

        where `dirichlet_bc_specification_dict` can have one of the following forms:

        - Boundary defined by function::

            {'bc_value': fenics.Constant((0.0, 0.0)),
             'boundary': Boundary(),
             'subspace_id': 0}

        - Boundary defined by reference to if od existing subdomain boundary::

             {'bc_value': fenics.Constant((0.0, 0.0)),
             'subdomain_boundary': 0,
             'subspace_id': 0}

        - Boundary defined by reference to named boundary::

             {'bc_value': fenics.Constant((0.0, 0.0)),
             'named_boundary': 'boundary_1',
             'subspace_id': 0}

        Implementation in :py:meth:`_construct_dirichlet_bc`.

        Dirichlet BCs can be applied on named_boundary and subdomain_boundary boundaries simultaneously.
        """
        if len(dirichlet_bcs) > 0:
            self.dirichlet_bcs_dict = dirichlet_bcs

            self.dirichlet_bcs = []
            for bc_name, bc_dict in self.dirichlet_bcs_dict.items():
                self.logger.info("     - Dirichlet BC '%s'" % bc_name)
                bc = self._construct_dirichlet_bc(bc_dict)
                if bc is not None:
                    self.dirichlet_bcs.append(bc)

    def _construct_dirichlet_bc(self, dirichlet_bc):
        """
        Constructs a fenics.DirichletBC from dictionary:
            - `subspace_id`     : id of subspace, only needed if function_space has subspaces
            - `bc_value`        : expression, function specifying value at boundary
            - `boundary`        : a fenics.SubDomain function or a fenics.MeshFunction
            - `boundary_id`     : integer, the id of a 'boundary' in self.subdomain_boundaries
            - `boundary_name`   : string, name of boundary in self.boundary_id_dict
        Required parameters:
            - 'bc_value'
            - one of: 'boundary', 'boundary_id', 'boundary_name'
            - 'subspace_id' (if mixed element function space)
        """
        # Check whether function-space contains MixedElement subspaces.
        if self._functionspace.has_subspaces:  # subspaces
            if 'subspace_id' in dirichlet_bc.keys():
                subspace_id = dirichlet_bc['subspace_id']
                funspace_bc = self._functionspace.get_functionspace_orig_subspace(subspace_id=subspace_id)
            else:
                self.logger.error("Dirichlet BC dictionary does not contain id of function sub space 'subspace_id'")
        else: # no subspaces
            funspace_bc = self._functionspace.get_functionspace()

        # Check for value.
        bc_value = dirichlet_bc['bc_value']
        if not 'bc_value' in dirichlet_bc.keys():
            self.logger.error("Dirichlet BC dictionary does not contain BC value 'bc_value'")

        bc = None
        # Check for type of boundary definition
        if 'boundary' in dirichlet_bc.keys():
            self.logger.info("       - Dirichlet BC from 'boundary' subdomain specification")
            bc = fenics.DirichletBC(funspace_bc, bc_value, dirichlet_bc['boundary'])
        else:
            if 'subdomain_boundary' in dirichlet_bc.keys():
                self.logger.info("       - Dirichlet BC from existing 'subdomain_boundary'")
                boundary_name = dirichlet_bc.get('subdomain_boundary')
                if boundary_name in self._subdomains.subdomain_boundaries_id_dict.keys():
                    boundary_id = self._subdomains.subdomain_boundaries_id_dict.get(boundary_name)
                    bc = fenics.DirichletBC(funspace_bc, bc_value, self._subdomains.subdomain_boundaries, boundary_id)
            elif 'named_boundary' in dirichlet_bc.keys():
                self.logger.info("       - Dirichlet BC from existing 'named_boundary'")
                boundary_id = self._subdomains.named_boundaries_id_dict.get(dirichlet_bc.get('named_boundary'))
                if boundary_id is not None:
                    bc = fenics.DirichletBC(funspace_bc, bc_value, self._subdomains.named_boundaries, boundary_id)
            else:
                self.logger.warning("       - Dirichlet BC incomplete -- skipping")
                bc = None
                boundary_id = None

        return bc

    def setup_von_neumann_boundary_conditions(self, von_neumann_bcs={}):
        """
        Parses bc_dict and creates `self.bcs_von_neumann` which contains::

            {bc_name : {bc_value    : function / expression,
                        subspace_id : integer,
                        measure     : correct measure and component of measure for this bc }
                        }

        Von Neumann term in variational form has form:
            delta_t * v ( \sum_i  g_diff_N_i * ds(boundary_i) ).

        Here we parse bc_dicts and compile standard format that will be used to create BC in
        implementation-specific :py:meth:`setup_problem` function.

        .. warning::
            Von Neuman BCs are implemented as surface integral (boundary measure ds) in the variational form of the problem.
            It seems that FENICS does not allow multiple boundary measures to be present in the same variational form.
            This means that all integrals need to be defined against the same boundary measure, and therefore
            named_boundary and subdomain_boundaries cannot be applied simultaneously in the current implementation.
            This could be solved by combining the boundaries from multiple measures into a single new boundary measure.

        .. warning::
            The current implementation uses the 'ds' boundary measure which only works for surface integrals over exterior
            facets. However, von Neumann boundary conditions across subdomain interfaces involve surface integrals over
            interior facets. These integrals require the 'dS' (capital S!) measure to be used instead of the 'ds' measure.
            In addition, the 'side of evaluation' must be indicated, e.g.: u('+')*dS(2) or u('-')*dS(2).
            See the following Q&A for further information:

                - https://fenicsproject.org/qa/11837/integration-on-an-internal-boundary-ds/
                - https://fenicsproject.org/qa/13400/facetnormal-of-internal-boundary/

        .. note::
            Consequences for GlimS simulations:

            Currently, GlimS uses a reaction-diffusion type model to simulate invasive tumor growth.
            This model assumes that tumor cells cannot leave the simulation domain, i.e. zero flux through the domain boundary.
            In fenics is a natural boundary condition that does not need to be imposed explicitly.
            Further, it is assumed that tumor cells cannot enter certain subdomains of the simulation domain, for example
            those representing CSF. This could be imposed as BC at the domain interface, however, a more natural approach
            is to assume that the material parameters of the 'isolated' subdomain are not compatible with the spread and
            evoluation of tumor cells, i.e. zero diffusion and proliferation.
            Indeed, this specification results in the desired behavior, see example 'test_case_simulation_tumor_growth_2D_subdomains'.


        """
        if len(von_neumann_bcs) > 0:
            self.von_neumann_bcs_dict = von_neumann_bcs
            # Then we create a von Neumann BC specification that includes bc_value, subspace_id, measure.
            bcs_von_neumann = {}
            for bc_name, bc_dict in self.von_neumann_bcs_dict.items():
                self.logger.info("   - Creating von Neumann BC '%s'" % bc_name)
                bc_specification = self._construct_von_neumann_bc(bc_dict)
                if bc_specification is not None:
                    bcs_von_neumann[bc_name] = bc_specification
            self.von_neumann_bcs = bcs_von_neumann

    def _construct_von_neumann_bc(self, bc_dict):
        """
        The output dictionary has the form::

            {bc_value       : function / expression,
             subspace_id    : integer,
             measure        : correct measure and component of measure for this bc }
        """
        # check if bc value is specified, raise error if not
        if 'bc_value' in bc_dict.keys():
            bc_value = bc_dict['bc_value']
        else:
            bc_value = None
            self.logger.error("Von Neumann BC dictionary does not contain BC value, key 'bc_value'")

        if self._functionspace.has_subspaces:  # subspaces
            if 'subspace_id' in bc_dict.keys():
                subspace_id = bc_dict['subspace_id']
                #funspace_bc = self._functionspace.get_functionspace_orig_subspace(subspace_id=subspace_id)
            else:
                subspace_id = None
                self.logger.error("Von Neumann BC dictionary does not contain id of function subspace 'subspace_id'")
        else: # no subspaces
            #funspace_bc = self._functionspace.get_functionspace()
            subspace_id = None

        # Check for type of boundary definition
        boundary_id=None
        measure=None

        if 'boundary' in bc_dict.keys():
            self.logger.error("You specified a function based boundary that has not been set up. "
                              "The current implemention requires such boundaries to be defined as 'named boundaries'"
                              "upon initialisation.")
        else:
            if 'subdomain_boundary' in bc_dict.keys():
                self.logger.info("       - Von Neumann BC from existing 'subdomain_boundary'")
                boundary_name = bc_dict.get('subdomain_boundary')
                if boundary_name in self._subdomains.subdomain_boundaries_id_dict.keys():
                    boundary_id = self._subdomains.subdomain_boundaries_id_dict.get(boundary_name)
                    measure = self._subdomains.ds(boundary_id)
            elif 'named_boundary' in bc_dict.keys():
                self.logger.info("       - Von Neumann BC from existing 'named_boundary'")
                boundary_id = self._subdomains.named_boundaries_id_dict.get(bc_dict.get('named_boundary'))
                measure = self._subdomains.dsn(boundary_id)
            else:
                self.logger.warning("       - Von Neumann BC incomplete -- skipping")


        if (bc_value is not None) and (measure is not None) and (subspace_id is not None):
            bc_von_neumann = {'bc_value': bc_value,
                              'measure': measure,
                              'subspace_id': subspace_id}
            return bc_von_neumann
        else:
            return None

    def time_update_bcs(self, time, kind='dirichlet'):
        """
        Function for updating time-dependent conditions with time at current time step.
        :param time: current time step
        :param kind: kind of BC, `dirichlet` or `von-neumann`.
        """
        bcs_dict = {}
        if kind == 'dirichlet':
            if hasattr(self, 'dirichlet_bcs_dict'):
                bcs_dict = self.dirichlet_bcs_dict
        elif kind == 'von-neumann':
            if hasattr(self, 'von_neumann_bcs_dict'):
                bcs_dict = self.von_neumann_bcs_dict

        for bc_name, bc in bcs_dict.items():
            try:
                bc['bc_value'].t = time
                self.logger.info("      - Updating %s BC '%s' at time %.2f " % (kind, bc_name, time))
            except Exception:
                self.logger.debug(
                    "Updating expression for %s BC '%s' at time %.2f raised exception" % (kind, bc_name, time))

    def implement_von_neumann_bc(self, product_component, subspace_id=None):
        """
        `test_functions` is a list::

            [test_function_subspace_0, test_function_subspace_1, ...]
        """
        bc_term = 0.0
        if self._functionspace.has_subspaces and (subspace_id is not None):
            if not hasattr(self, 'von_neumann_bcs_by_subspace'):
                self._create_von_neumann_bcs_by_subspace_dict()
            # create variational terms for each subspace
            if len(self.von_neumann_bcs_by_subspace):
                subspace_bcs = self.von_neumann_bcs_by_subspace.get(subspace_id)
                if len(subspace_bcs)>0:
                    bc_term = self._implement_von_neumann_bcs_subspace(product_component, subspace_bcs, subspace_id=subspace_id)
                    self.logger.debug("Final von Neumann BC term (function space %d): %s" % (subspace_id, bc_term))
        elif (not self._functionspace.has_subspaces) and (subspace_id is None):  # no subspaces
            bc_term = self._implement_von_neumann_bcs_subspace(product_component, self.bcs_von_neumann.values())
            self.logger.debug("Final von Neumann BC term (main function space): %s" % bc_term)
        else:
            self.logger.error("Choice of subspace ID not compatible with functionspace")
        return bc_term

    def _create_von_neumann_bcs_by_subspace_dict(self):
        self.von_neumann_bcs_by_subspace = {}
        if hasattr(self, 'von_neumann_bcs'):
            for subspace_id in self._functionspace.subspaces.get_subspace_ids():
                self.von_neumann_bcs_by_subspace[subspace_id] = []
                for name, bc_dict in self.von_neumann_bcs.items():
                    v_n_subspace_id = bc_dict.get('subspace_id')
                    if v_n_subspace_id == subspace_id:
                        bc_dict['name'] = name
                        self.von_neumann_bcs_by_subspace[subspace_id].append(bc_dict)

    def _implement_von_neumann_bcs_subspace(self, product_components, subspace_bcs, subspace_id=None):
        """
        Creates \sum_i  ( g_diff_N_i * product_components * ds(boundary_i) ) for subspace i.

        """
        subspace_dim = self._functionspace.get_element(subspace_id=subspace_id).value_size()
        bc_terms = []
        for bc in subspace_bcs:
            self.logger.info("     - implementing von Neumann BC '%s'" % (bc.get('name')))
            if subspace_dim == 1:
                bc_terms.append( bc.get('bc_value') * product_components * bc.get('measure') )
            else:
                bc_terms.append( fenics.inner(bc.get('bc_value'), product_components) * bc.get('measure'))
        return sum(bc_terms)

class Parameters():
    """
    Helper class for management of simulation parameters.
    """
    def __init__(self, functionspace, subdomains, time_dependent=False):
        """
        Init routine.
        :param time_dependent: Boolean switch indicating whether this simulation is time-dependent.
        :param functionspace: Instance of FunctionSpace.
        :param subdomains: Instance of SubDomains.
        """
        self.logger = logging.getLogger(__name__)
        self.time_dependent = time_dependent
        self._functionspace = functionspace
        self._subdomains = subdomains
        self._iv_base_name = 'iv'

        if self.time_dependent:
            self.sim_time = 1
            self.sim_time_step = 1

    def get_iv_map(self, return_name=True):
        iv_map = {}
        if self._functionspace.has_subspaces:
            for subspace_id in self._functionspace.subspaces.get_subspace_ids():
                if return_name:
                    iv_map[subspace_id] = self._get_iv_name(subspace_id)
                else:
                    iv_map[subspace_id] = self.get_iv(subspace_id)
        else:
            if return_name:
                iv_map = self._get_iv_name()
            else:
                iv_map = self.get_iv()
        return iv_map

    def _get_iv_name(self, subspace_id=None):
        if subspace_id is not None:
            subspace_name = self._functionspace.subspaces.names.get(subspace_id)
            iv_name = self._iv_base_name + '_' + subspace_name
        else:
            iv_name = self._iv_base_name
        return iv_name

    def get_iv(self, subspace_id):
        if hasattr(self, self._get_iv_name(subspace_id)):
            return getattr(self, self._get_iv_name(subspace_id))
        else:
            self.logger.warning("Initial value expression '%s' for subspace %i undefined" % (
                self._get_iv_name(subspace_id), subspace_id))

    def _set_iv(self, iv, subspace_id=None, replace=False):
        iv_name = self._get_iv_name(subspace_id=subspace_id)
        if not hasattr(self, iv_name):
            setattr(self, iv_name, iv)
        else:
            if replace:
                self.logger.warning("Initial value expression '%s' for subspace %i already exists ... replacing" % (
                    iv_name, subspace_id))
                setattr(self, iv_name, iv)
            else:
                self.logger.warning("Initial value expression '%s' for subspace %i already exists ... do nothing" % (
                    iv_name, subspace_id))

    def set_initial_value_expressions(self, ivs={}, replace=False):
        """
        Sets initial value expressions.
        :param ivs: Dictionary with {subspace_id, initial value expression } if subspaces exist,
        or simple initial value expression otherwise
        """
        for subspace_id, iv in ivs.items():
            self._set_iv(iv, subspace_id, replace=replace)

    def create_initial_value_function(self):
        iv_map = self.get_iv_map(return_name=False) # gets ivs
        u_iv = self._functionspace.project_over_space(iv_map)
        return u_iv

    def define_required_params(self, params_name_list=[]):
        param_list = copy.deepcopy(params_name_list)
        if self.time_dependent:
            param_list.extend(['sim_time', 'sim_time_step'])
        self.params_required = list(set(param_list))

    def define_optional_params(self, params_name_list=[]):
        param_list = copy.deepcopy(params_name_list)
        self.params_optional = list(set(param_list))

    def _check_param_arguments(self, kw_args):
        """
        Helper function for
        - verifying whether multiple required parameters in 'param_name_list' are present in a kwargs 'kw_args' dictionary.
        - making these parameter available as attribute of the class instance, and ignoring other parameters that are
          available in kwargs but not needed
        :param kw_args: kwargs dictionary
        :param param_name_list: list of required parameter names
        """
        required_params = self.params_required
        available_params = kw_args.keys()

        required_not_available = set(required_params).difference(set(available_params))
        available_not_required = set(available_params).difference(set(required_params))
        available_and_required = set(required_params).intersection(set(available_params))

        if len(available_and_required) == len(required_params):
            for param in required_params:
                if param in available_params:
                    self.logger.info("    - found '%s'" % param)
                else:
                    self.logger.error("    - required parameter '%s' missing" % param)
            for param in available_not_required:
                self.logger.info("    - parameter '%s' not needed" % param)
            return True
        else:
            for param in required_not_available:
                self.logger.warning("    - parameter '%s' required but not available" % param)
            return False

    def set_parameter(self, param_name, param):
        if type(param) == dict:
            self.logger.info("Parameter '%s' is dictionary -- generate discontinuous scalar" % param_name)
            param_value = self._subdomains.create_discontinuous_scalar_from_parameter_map(param, param_name)
            setattr(self, param_name + '_dict', param)
            setattr(self, param_name, param_value)
        else:
            setattr(self, param_name, param)

    def get_parameter(self, param_name):
        if hasattr(self, param_name):
            param = getattr(self, param_name)
        else:
            self.logger.warning("Parameter '%s' has not been set."%param_name)
            param=None
        return param

    def init_parameters(self, parameter_dict):
        if self._check_param_arguments(parameter_dict):
            for name, value in parameter_dict.items():
                if (name in self.params_required) or (name in self.params_optional):
                    self.set_parameter(name, value)
                else:
                    self.logger.info("Parameter '%s' will be ignored."%name)
        else:
            self.logger.warning("Parameterset incomplete cannot initialize.")

    def time_update_parameters(self, time):
        iv_map = self.get_iv_map()
        if type(iv_map)==dict:
            iv_names = iv_map.values()
        else: # single functionspace & IV
            iv_names = [iv_map]
        for param_list in [self.params_required, self.params_optional, iv_names]:
            for param in param_list:
                self._time_update_expression_attribute(param, time)

    def _time_update_expression_attribute(self, attribute_name, time):
        """
        Function for updating time-dependent expression attribute with time at current time step.
        :param attribute_name: name of attribute
        :param time: current time step
        """
        try:
            if hasattr(self, attribute_name):
                expression = getattr(self, attribute_name)
                expression.t = time
                self.logger.info("      - Updating '%s' at time %.2f " % (attribute_name, time))
        except Exception:
            self.logger.info("Updating %s at time %.2f raised exception" % (attribute_name, time))





class TimeSeriesDataTimePoint():
    """
    This class provides a datastructure for a single observation time point in a time series
    """

    def __init__(self, time, time_step, recording_step):
        self.time = time
        self.recording_step = recording_step
        self.time_step = time_step

    def set_field(self, field):
        self.field = field

    def get_field(self):
        if hasattr(self, 'field'):
            return self.field

    def get_time(self):
        return self.time

    def get_time_step(self):
        return self.time_step

    def get_recording_step(self):
        return self.recording_step


class TimeSeriesData():
    """
    This class provides a datastructure for time series data from observations over a single functionspace.

    """
    def __init__(self, name, functionspace):
        """
        :param name: name of solution time series
        :param functionspace: instance of FunctionSpace helper class
        """
        self.logger = logging.getLogger(__name__)
        self._functionspace = functionspace
        self.name = name
        self.data = {}  # here data is being stored, keys correspond to recording_step

    def exists_recording_step(self, recording_step):
        return recording_step in self.data.keys()

    def add_observation(self, field, time, time_step, recording_step, replace=False):
        # make copy of field
        try:
            field_copy = field.copy(deepcopy=True)
        except:
            field_copy = self._functionspace.project_over_space(field, subspace_id=None, subspace_name=None)
        # create new instance of TimeSeriesDataTimePoint
        observation = TimeSeriesDataTimePoint(time=time, time_step=time_step, recording_step=recording_step)
        observation.set_field(field_copy)
        # Check if data for this recording step already exists
        if self.exists_recording_step(recording_step=recording_step):
            self.logger.warning("Recording step %i already exists" % recording_step)
            if replace:
                self.logger.warning("Replacing existing recording step %i" % recording_step)
                self.data[recording_step] = observation
        else:
            self.data[recording_step] = observation

    def get_observation(self, recording_step):
        if self.exists_recording_step(recording_step):
            return self.data.get(recording_step)
        else:
            self.logger.warning("No solution available for recording step '%d'" % recording_step)

    def get_all_recording_steps(self):
        return sorted(self.data.keys())

    def get_most_recent_observation(self):
        max_rec_step = max(self.data.keys())
        return self.get_observation(max_rec_step)

    def get_solution_function(self, subspace_name=None, subspace_id=None, recording_step=None):
        if recording_step is None:
            self.logger.info("Trying to access observation for latest recording step")
            observation = self.get_most_recent_observation()
        else:
            self.logger.info("Trying to access solution for recording step '%d'" % recording_step)
            observation = self.get_observation(recording_step)
        # get either function or function on subspace
        if observation is not None:
            field_function = observation.get_field()
            result_sub = self._functionspace.split_function(field_function,
                                                            subspace_id=subspace_id, subspace_name=subspace_name)
            # project function over entire function space or subspace
            if config.USE_ADJOINT:
                solution_function = self._functionspace.project_over_space(result_sub,
                                                                           subspace_name=subspace_name,
                                                                           subspace_id=subspace_id,
                                                                           annotate=False)
            else:
                solution_function = self._functionspace.project_over_space(result_sub,
                                                                           subspace_name=subspace_name,
                                                                           subspace_id=subspace_id)
            return solution_function


class TimeSeriesMultiData():

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.time_series_prefix = 'tds_'

    def exists_time_series(self, name):
        return hasattr(self, self.time_series_prefix+name)

    def exists_recording_step(self, name, recording_step):
        ts = self.get_time_series(name)
        return ts.exists_recording_step(recording_step)

    def get_all_time_series(self):
        """
        Returns dictionary {time_series_name : time_series}
        """
        time_series_dict = {}
        for attribute_name in dir(self):
            if attribute_name.startswith(self.time_series_prefix):
                name = attribute_name.replace(self.time_series_prefix, '')
                ts = getattr(self, attribute_name)
                time_series_dict[name] = ts
        return time_series_dict

    def register_time_series(self, name, functionspace, replace=False):
        time_series_data = TimeSeriesData(name=name, functionspace=functionspace)
        attribute_name = self.time_series_prefix+name
        if self.exists_time_series(name):
            self.logger.warning("TimeSeries '%s' already exists" % name)
            if replace:
                self.logger.warning("Replacing existing TimeSeries '%s'" % name)
                setattr(self, attribute_name, time_series_data)
        else:
            setattr(self, attribute_name, time_series_data)

    def get_time_series(self, name):
        attribute_name = self.time_series_prefix + name
        if self.exists_time_series(name):
            return getattr(self, attribute_name)
        else:
            self.logger.warning("TimeSeries '%s' does not exist." % name)

    def get_observation(self, name, recording_step):
        tsd = self.get_time_series(name)
        if tsd is not None:
            return tsd.get_observation(recording_step)

    def add_observation(self, name, field, time, time_step, recording_step, replace=False):
        if not self.exists_time_series(name):
            self.logger.info("Registering new TimeSeries '%s'"%name)
        tsd = self.get_time_series(name)
        if tsd is not None:
            tsd.add_observation(field, time, time_step, recording_step, replace=replace)

    def get_solution_function(self, name, subspace_name=None, subspace_id=None, recording_step=None):
        tsd = self.get_time_series(name)
        if tsd is not None:
            return tsd.get_solution_function(subspace_name, subspace_id, recording_step)

    def get_all_recording_steps(self, name):
        tsd = self.get_time_series(name)
        if tsd is not None:
            return tsd.get_all_recording_steps()

    def _get_mpi_comm(self):
        ts_dict = self.get_all_time_series()
        if len(ts_dict)>0:
            name = list(ts_dict.keys())[0]
            ts = self.get_time_series(name)
            return ts._functionspace._mesh.mpi_comm()

    def save_to_hdf5(self, path_to_file, replace=False):
        if os.path.exists(path_to_file):
            self.logger.warning("File '%s' exists already."%path_to_file)
            if replace:
                self.logger.warning("Overwriting file '%s'." % path_to_file)
            else:
                path_to_file = 'test.h5'
                self.logger.warning("Creating file with different name '%s'." % path_to_file)
        # get mpi_comm from one of the meshes
        mpi_comm = self._get_mpi_comm()
        # open File
        hdf = fenics.HDF5File(mpi_comm, path_to_file, "w")
        # iterate through time series data sets
        for name, ts in self.get_all_time_series().items():
            # iterate through time steps
            for recording_step in self.get_all_recording_steps(name):
                time_step = self.get_observation(name, recording_step).get_time_step()
                function = self.get_solution_function(name, subspace_name=None, recording_step=recording_step)
                if function is not None:
                    hdf.write(function, name, time_step)
        hdf.close()

    def _create_empty_function(self, name, subspace_id=None, subspace_name=None):
        ts = self.get_time_series(name)
        if ts is not None:
            funspace = ts._functionspace.get_functionspace(subspace_id=subspace_id, subspace_name=subspace_name)
            function = fenics.Function(funspace)
            return function

    def load_from_hdf5(self, path_to_file):
        # get mpi_comm from one of the meshes
        mpi_comm = self._get_mpi_comm()
        # open file
        if os.path.exists(path_to_file):
            hdf = fenics.HDF5File(mpi_comm, path_to_file, "r")
            # iterate through registered time series
            for name, ts in self.get_all_time_series().items():
                ts_attribute = hdf.attributes(name)
                n_steps = ts_attribute['count']
                # create new observation for each recording step in hdf file
                for step in range(n_steps):
                    dataset = name+"/vector_%d" % step
                    step_attribute = hdf.attributes(dataset)
                    time_step = step_attribute['timestamp']
                    function = self._create_empty_function(name)
                    #print("before assignment", function.vector().array())
                    hdf.read(function, dataset)
                    #print("after assignment", function.vector().array())
                    self.add_observation(name, function,
                                         time=time_step, time_step=time_step, recording_step=step)
            hdf.close()
        else:
            self.logger.warning("File '%s' does not exist"%path_to_file)



class Results():
    """
      Helper class for management of simulation results.
    """

    def __init__(self, functionspace, subdomains=None, output_dir=config.output_dir_simulation_tmp):
        """
        Init routine.
        :param functionspace: Instance of FunctionSpace.
        """
        self.logger = logging.getLogger(__name__)
        self._functionspace = functionspace
        self.current_time_step = 0
        self.set_save_output_dir(output_dir)
        self.ts_name = 'solution'
        self.data = TimeSeriesMultiData()
        self.data.register_time_series(self.ts_name, functionspace=functionspace)
        if subdomains is not None:
            self._subdomains = subdomains

    def set_save_output_dir(self, output_dir):
        self.output_dir = output_dir
        fu.ensure_dir_exists(self.output_dir)

    def add_to_results(self, current_sim_time, current_time_step, recording_step, field, replace=False):
        self.data.add_observation(name=self.ts_name, time=current_sim_time, time_step=current_time_step,
                                  recording_step=recording_step, field=field, replace=replace)

    def exists_recording_step(self, recording_step):
        return self.data.exists_recording_step(name=self.ts_name, recording_step=recording_step)

    def get_result(self, recording_step):
        return self.data.get_observation(name=self.ts_name, recording_step=recording_step)

    def get_solution_function(self, subspace_name=None, subspace_id=None, recording_step=None):
        return self.data.get_solution_function(name=self.ts_name, subspace_name=subspace_name, subspace_id=subspace_id,
                                                 recording_step=recording_step)

    def save_function(self, function, function_name, function_save_name, time, subspace_id=None, method='xdmf'):
        # copy function & rename
        if hasattr(function, 'function_space'):
            function_local = function.copy(deepcopy=True)
        elif type(function)==fenics.MeshFunctionSizet: # label function
            function_local = function
        else:
            function_local = self._functionspace.project_over_space(function, subspace_id=subspace_id)
        function_local.rename(function_name, 'label')

        if method == 'xdmf':
            if hasattr(self, 'output_xdmf_file'):
                self.output_xdmf_file.write_checkpoint(function_local, function_name, time)
            else:
                path_to_file = os.path.join(self.output_dir, function_save_name + '.xdmf')
                fu.ensure_dir_exists(path_to_file)
                self.output_xdmf_file = fenics.XDMFFile(self._functionspace._mesh.mpi_comm(), path_to_file)
                self.output_xdmf_file.write_checkpoint(function_local, function_name, time)
        elif method == 'vtk':
            path_to_file = os.path.join(self.output_dir, function_name, function_save_name + '.pvd')
            fu.ensure_dir_exists(path_to_file)
            out_file = fenics.File(path_to_file)
            out_file << (function_local, float(time))
        else:
            self.logger.warning("Save method '%s' is not defined" % method)

    def get_function_save_name(self, function_name, recording_step, method='vtk'):
        if method == 'xdmf':
            function_save_name = "solution_xdmf"
        elif method == 'vtk':
            function_save_name = "%s_%05d" % (function_name, recording_step)
        return function_save_name

    def save_solution(self, recording_step, time, function=None, method='xdmf'):
        if method is not None:
            if function is None:
                function = self.get_solution_function(subspace_name=None, subspace_id=None, recording_step=recording_step)
            if self._functionspace.has_subspaces:
                for subspace_id in self._functionspace.subspaces.get_subspace_ids():
                    subspace_function = self._functionspace.split_function(function=function, subspace_id=subspace_id)
                    subspace_name = self._functionspace.subspaces.get_subspace_name(subspace_id)
                    function_save_name = self.get_function_save_name(subspace_name, recording_step, method=method)
                    self.save_function(subspace_function, subspace_name, function_save_name, time, subspace_id, method=method)
            else: # no subspaces
                function_space_name = self._functionspace.name
                function_save_name = self.get_function_save_name(function_space_name, recording_step, method=method)
                self.save_function(function, function_space_name, function_save_name, time, method=method)

    def save_label_function(self, recording_step, time, method='xdmf'):
        function_name = 'label_map'
        function_save_name = self.get_function_save_name(function_name, recording_step, method)
        self.save_function(self._subdomains.subdomains, function_name, function_save_name, time, method=method)

    def save_solution_start(self, method='xdmf', clear_all=False):
        """
        be careful with 'clear_all' option!
        # check FE filter for paraview
        # https://github.com/michalhabera/xdmf-fe-filter
        """
        if method is not None:
            if os.path.exists(self.output_dir) and clear_all:
                try:
                    shutil.rmtree(self.output_dir)
                except:
                    pass
            else:
                fu.ensure_dir_exists(self.output_dir)
            if method == 'xdmf':
                file_name = "solution"
                path_to_out_file = os.path.join(self.output_dir, file_name + '.xdmf')
                path_to_h_file = os.path.join(self.output_dir, file_name + '.h5')
                try:
                    if os.path.isfile(path_to_out_file):
                        os.remove(path_to_out_file)
                    if os.path.isfile(path_to_h_file):
                        os.remove(path_to_h_file)
                except:
                    pass
                self.output_xdmf_file = fenics.XDMFFile(self._functionspace._mesh.mpi_comm(), path_to_out_file)
                self.output_xdmf_file.write(self._functionspace._mesh)
            if hasattr(self, '_subdomains') and method=='vtk':
                self.save_label_function(0, 0, method=method)

    def save_solution_hdf5(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'solution_timeseries.h5')
            fu.ensure_dir_exists(save_path)
        self.data.save_to_hdf5(save_path, replace=True)

    def save_solution_end(self, method='xdmf'):
        if method is not None:
            if method == 'xdmf':
                self.output_xdmf_file.close()

    def get_recording_steps(self):
        return self.data.get_all_recording_steps(self.ts_name)


class Plotting():
    """
      Helper class for plotting from simulation results.
    """

    def __init__(self, results, output_dir=config.output_dir_plot_tmp):
        """
        Init routine.
        :param results: Instance of Results.
        """
        self.logger = logging.getLogger(__name__)
        self._results = results
        self.set_plot_output_dir(output_dir)

    def set_plot_output_dir(self, output_dir):
        self.output_dir = output_dir
        fu.ensure_dir_exists(self.output_dir)

    def plot(self, recording_step=None, file_name=None, subspace_name=None, subspace_id=None,
             output_dir='', **kwargs):
        """
        Creates plot of function on specified functionspace and recording step.
        Additional keywords are processed as in :py:meth:`visualisation.plotting.show_img_seg_f()`.
        :param recording_step: recording step to be plotted; default is None
        :param file_name: filename for saving; default is None
        :param subspace_name: name of subspace to be plotted; default is None
        :param subspace_id: id of subspace to be plotted; default is None
        :param kwargs: additional key-value arguments as in :py:meth:`visualisation.plotting.show_img_seg_f()`
        """
        function = self._results.get_solution_function(subspace_name=subspace_name, subspace_id=subspace_id,
                                                       recording_step=recording_step)
        if file_name is None:
            file_name = "plot_%04d.png"%recording_step

        plott.show_img_seg_f(function=function, path=os.path.join(output_dir, file_name), **kwargs)

    def plot_concentration(self, recording_step, file_name=None, **kwargs):
        if file_name is None:
            file_name = "concentration_%04d.png"%recording_step
        plot_params = {"label" : "concentration",
                       "title" : "concentration @ step %04d"%recording_step}
        kwargs.update(plot_params)
        self.plot(recording_step=recording_step, file_name=file_name, subspace_name="concentration",
                  output_dir=os.path.join(self.output_dir, 'concentration'), **kwargs)

    def plot_displacement(self, recording_step, file_name=None, **kwargs):
        if file_name is None:
            file_name = "displacement_%04d.png"%recording_step
        plot_params = {"label" : "displacement",
                       "title" : "displacement @ step %04d"%recording_step}
        kwargs.update(plot_params)
        self.plot(recording_step=recording_step, file_name=file_name, subspace_name="displacement",
                  output_dir=os.path.join(self.output_dir, 'displacement'), **kwargs)

    def plot_all(self, recording_step):
        if self._results._functionspace.has_subspaces:
            for subspace_name in self._results._functionspace.subspaces.get_subspace_names():
                plot_name = subspace_name+"_%04d.png"%recording_step
                plot_params = {"label": subspace_name,
                               "title": "%s @ step %04d" % (subspace_name, recording_step)}
                self.plot(recording_step=recording_step, file_name=plot_name, subspace_name=subspace_name,
                          output_dir=os.path.join(self.output_dir, subspace_name), **plot_params)



class PostProcess(ABC):

    def __init__(self, results, params, output_dir=config.output_dir_simulation_tmp, plot_params={}):
        """
        Init routine.
        :param results: Instance of Results.
        """
        self.logger = logging.getLogger(__name__)
        self._results = results
        self._params = params
        self._functionspace = self._results._functionspace
        self._subdomains = self._results._subdomains
        self._mesh = self._functionspace._mesh
        self._projection_parameters = self._functionspace._projection_parameters
        self.set_output_dir(output_dir)
        self.dim = self._mesh.geometry().dim()
        self.plot_params = {   "showmesh": False,
                               "contour": False,
                               "exclude_min_max": False,
                               "colormap":'viridis',
                               "n_cmap_levels" : 20,
                               "dpi" : 300,
                               "alpha" : 1,
                               "alpha_f" : 1,
                               "shading" : "gouraud"}
        self.update_plot_params(plot_params)

    def update_plot_params(self, plot_params={}):
        self.plot_params.update(plot_params)

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        fu.ensure_dir_exists(self.output_dir)

    def get_output_dir(self):
        if hasattr(self, 'output_dir'):
            return self.output_dir
        else:
            self.logger.warning("No output directory has been defined. Specify 'output_dir'")

    def get_solution_displacement(self, recording_step=None):
        return self._results.get_solution_function(subspace_name='displacement', recording_step=recording_step)

    def get_solution_concentration(self, recording_step=None):
        return self._results.get_solution_function(subspace_name='concentration', recording_step=recording_step)

    def get_strain_tensor(self, recording_step=None):
        VT = fenics.TensorFunctionSpace(self._mesh, "Lagrange", 1)
        displacement = self.get_solution_displacement(recording_step=recording_step)
        strain_tensor = mle.compute_strain(displacement)
        strain_tensor_fct = fenics.project(strain_tensor, VT, **self._projection_parameters)
        strain_tensor_fct.rename("strain_tensor", "")
        return strain_tensor_fct

    @abstractmethod
    def get_stress_tensor(self, recording_step=None):
        pass

    @abstractmethod
    def get_logistic_growth(self, recording_step=None):
        pass

    @abstractmethod
    def get_mech_expansion(self, recording_step=None):
        pass

    def get_pressure(self, recording_step=None):
        stress = self.get_stress_tensor(recording_step=recording_step)
        pressure      = mle.compute_pressure_from_stress_tensor(stress)
        F = fenics.FunctionSpace(self._mesh, "Lagrange", 1)
        pressure_fct = fenics.project(pressure, F, **self._projection_parameters)
        pressure_fct.rename("pressure", '')
        return pressure_fct

    def get_van_mises_stress(self, recording_step=None):
        stress = self.get_stress_tensor(recording_step=recording_step)
        van_mises_stress = mle.compute_van_mises_stress(stress, self.dim)
        F = fenics.FunctionSpace(self._mesh, "Lagrange", 1)
        van_mises_stress_fct = fenics.project(van_mises_stress, F, **self._projection_parameters)
        van_mises_stress_fct.rename("pressure", '')
        return van_mises_stress_fct

    def compute_force(self, recording_step=None, subdomain_id=None):
        n   = fenics.FacetNormal(self._mesh)
        stress_tensor = self.get_stress_tensor(recording_step=recording_step)
        traction = fenics.dot(stress_tensor, n)
        dss = self._results._subdomains.ds
        if (subdomain_id is not None):
            dss = dss(subdomain_id)
        force    = [fenics.assemble(traction[i] * dss) for i in range(traction.ufl_shape[0])]
        return force

    def get_displacement_norm(self, recording_step=None):
        displacement = self.get_solution_displacement(recording_step=recording_step)
        disp_norm = fenics.inner(displacement,displacement)**0.5
        F = fenics.FunctionSpace(self._mesh, "Lagrange", 1)
        disp_norm_fct = fenics.project(disp_norm, F, **self._projection_parameters)
        disp_norm_fct.rename("displacement_norm", '')
        return disp_norm_fct

    def plot_function(self, function, recording_step, name, file_name=None, units=None, output_dir=None,
                      show_labels=False, **kwargs):
        if output_dir is None:
            output_dir = self.get_output_dir()
        if file_name is None:
            file_name = "%s_%04d.png" % (name.replace(" ", "_"), recording_step)
        if units is not None:
            label_name = "%s [%s]"%(name, units)
        else:
            label_name = "%s"%(name)
        plot_params = self.plot_params.copy()
        plot_params_local = {"label": label_name}
        plot_params.update(plot_params_local)
        title = "%s @ step %04d" %(name, recording_step)
        if 'title' in plot_params.keys():
            if not plot_params.get('title') is None:
                plot_params.update({'title' : title})
        else:
            plot_params.update({'title': title})
        plot_params.update(kwargs)
        save_path = os.path.join(output_dir, file_name)
        if not show_labels:
            plott.show_img_seg_f(function=function, path=save_path, **plot_params)
        else:
            labels = self.get_label_function()
            plot_obj_label = {'object': labels,
                              'cbar_label': None,
                              'exclude_below': None,
                              'exclude_above': None,
                              'exclude_min_max': False,
                              'exclude_around': None,
                              'cmap': 'Greys_r',
                              'n_cmap_levels': None,
                              'range_f': None,
                              'showmesh': False,
                              'shading': "gouraud",
                              'alpha': 1,
                              'norm': None,
                              'norm_ref': None,
                              'color': None
                              }
            plott.show_img_seg_f(function=function, path=save_path,
                                 add_plot_object_pre=plot_obj_label,
                                 **plot_params)

    def plot_concentration(self, recording_step, **kwargs):
        conc = self.get_solution_concentration(recording_step=recording_step)
        plot_params = { "range_f" : [0.000, 1.0] }
        plot_params.update(kwargs)
        self.plot_function(conc, recording_step=recording_step, name="concentration",
                           file_name=None, units=None, output_dir=os.path.join(self.get_output_dir(), 'concentration'), **plot_params)

    def plot_displacement(self, recording_step, **kwargs):
        disp = self.get_solution_displacement(recording_step=recording_step)
        plot_params = {"range_f": [0.000, None]}
        plot_params.update(kwargs)
        self.plot_function(disp, recording_step=recording_step, name="displacement",
                           file_name=None, units="mm", output_dir=os.path.join(self.get_output_dir(), 'displacement'), **kwargs)

    def plot_pressure(self, recording_step, **kwargs):
        pressure = self.get_pressure(recording_step=recording_step)
        self.plot_function(pressure, recording_step=recording_step, name="pressure",
                           file_name=None, units="Pa", output_dir=os.path.join(self.get_output_dir(), 'pressure'), **kwargs)

    def plot_displacement_norm(self, recording_step, **kwargs):
        disp_norm = self.get_displacement_norm(recording_step=recording_step)
        self.plot_function(disp_norm, recording_step=recording_step, name="displacement norm",
                           file_name=None, units="mm", output_dir=os.path.join(self.get_output_dir(), 'displacement_norm'), **kwargs)

    def plot_van_mises_stress(self, recording_step, **kwargs):
        van_mises = self.get_van_mises_stress(recording_step=recording_step)
        self.plot_function(van_mises, recording_step=recording_step, name="van mises stress",
                           file_name=None, units=None,
                           output_dir=os.path.join(self.get_output_dir(), 'van_mises_stress'), **kwargs)

    def get_label_function(self):
        if hasattr(self._subdomains, 'label_function'):
            labelfunction = self._subdomains.label_function
        else:
            if self.dim==2:
                labelfunction = vh.convert_meshfunction_to_function(self._mesh, self._subdomains.subdomains)
        return labelfunction


    def plot_label_function(self, recording_step, **kwargs):
        plot_params = {"colormap": 'Set1'}
        plot_params.update(kwargs)
        labelfunction = self.get_label_function()
        self.plot_function(labelfunction, recording_step=recording_step, name="label function",
                           file_name=None, units=None,
                           output_dir=os.path.join(self.get_output_dir(), 'label_function'), **plot_params)

    def _update_mesh_displacements(self, displacement):
        """
        Applies displacement function to mesh.
        .. warning:: This changes the current mesh! Multiple updates result in additive mesh deformations!
        """
        fenics.ALE.move(self._mesh, displacement)
        self._mesh.bounding_box_tree().build(self._mesh)

    def update_mesh_displacement(self, recording_step=None, reverse=False):
        """
        Update mesh with simulated displacement from specified time-point.
        .. warning:: This changes the current mesh! Multiple updates result in additive mesh deformations!
        """
        displacement = self.get_solution_displacement(recording_step)
        if reverse:
            neg_disp = fenics.project(-1*displacement, displacement.function_space(),  **self._projection_parameters)
            self._update_mesh_displacements(neg_disp)
        else:
            self._update_mesh_displacements(displacement)



class PostProcessTumorGrowth(PostProcess):

    def get_stress_tensor(self, recording_step=None):
        VT = fenics.TensorFunctionSpace(self._mesh, "Lagrange", 1)
        displacement = self.get_solution_displacement(recording_step=recording_step)
        mu = mle.compute_mu(self._params.E, self._params.poisson)
        lmbda = mle.compute_lambda(self._params.E, self._params.poisson)
        stress_tensor = mle.compute_stress(displacement, mu=mu, lmbda=lmbda)
        stress_tensor_fct = fenics.project(stress_tensor, VT, **self._projection_parameters)
        stress_tensor_fct.rename("stress_tensor", "")
        return stress_tensor_fct

    def get_logistic_growth(self, recording_step=None):
        concentration = self.get_solution_concentration(recording_step=recording_step)
        log_growth = mrd.compute_growth_logistic(concentration, self._params.proliferation, 1.0)
        F = fenics.FunctionSpace(self._mesh, "Lagrange", 1)
        log_growth_fct = fenics.project(log_growth, F, **self._projection_parameters)
        log_growth_fct.rename("log_growth", '')
        return log_growth_fct

    def get_mech_expansion(self, recording_step=None):
        VT = fenics.TensorFunctionSpace(self._mesh, "Lagrange", 1)
        concentration = self.get_solution_concentration(recording_step=recording_step)

        mech_exp = simulation.helpers.math_linear_elasticity.compute_growth_induced_strain(concentration, self._params.coupling, self.dim)
        mech_exp_fct = fenics.project(mech_exp, VT, **self._projection_parameters)
        mech_exp_fct.rename("mech_expansion", '')
        return mech_exp_fct

    def get_total_jacobian(self, recording_step=None):
        V = fenics.FunctionSpace(self._mesh, "Lagrange", 1)
        displacement = self.get_solution_displacement(recording_step=recording_step)
        jac = mle.compute_total_jacobian(displacement)
        jac_fct = fenics.project(jac, V, **self._projection_parameters)
        jac_fct.rename("total_jacobian", '')
        return jac_fct

    def get_growth_induced_jacobian(self, recording_step=None):
        V = fenics.FunctionSpace(self._mesh, "Lagrange", 1)
        strain_growth = self.get_mech_expansion(recording_step)
        jac_growth = mle.compute_growth_induced_jacobian(strain_growth, self.dim)
        jac_growth_fct = fenics.project(jac_growth, V, **self._projection_parameters)
        jac_growth_fct.rename("growth_induced_jacobian", '')
        return jac_growth_fct

    def get_concentration_deformed_configuration(self, recording_step=None):
        concentration = self.get_solution_concentration(recording_step=recording_step)
        displacement = self.get_solution_displacement(recording_step=recording_step)
        V = fenics.FunctionSpace(self._mesh, "Lagrange", 1)
        conc_def = mle.compute_concentration_deformed(concentration, displacement, self._params.coupling, self.dim)
        conc_def_fct = fenics.project(conc_def, V, **self._projection_parameters)
        conc_def_fct.rename("concentration_deformed_config", '')
        return conc_def_fct

    def plot_concentration_deformed_configuration(self, recording_step, **kwargs):
        conc_def = self.get_concentration_deformed_configuration(recording_step=recording_step)
        plot_params = {"range_f": [0.000, 1.0]}
        plot_params.update(kwargs)
        self.plot_function(conc_def, recording_step=recording_step, name="concentration deformed configuration",
                           file_name=None, units=None,
                           output_dir=os.path.join(self.get_output_dir(), 'concentration_deformed'), **plot_params)

    def plot_log_growth(self, recording_step, **kwargs):
        log_growth = self.get_logistic_growth(recording_step=recording_step)
        plot_params = {  # "exclude_around" :(0, 0.0001),
                        "colormap": 'RdBu_r',
                        "cmap_ref": 0.0}
        plot_params.update(kwargs)
        self.plot_function(log_growth, recording_step=recording_step, name="logistic growth term",
                           file_name=None, units=None, output_dir=os.path.join(self.get_output_dir(), 'logistic_growth_term'), **plot_params)

    def plot_total_jacobian(self, recording_step, **kwargs):
        jac = self.get_total_jacobian(recording_step=recording_step)
        plot_params = {  # "exclude_around" :(0, 0.0001),
                        "range_f": [0.8, 1.2],
                        "colormap": 'RdBu_r',
                        "cmap_ref": 1.0}
        plot_params.update(kwargs)
        self.plot_function(jac, recording_step=recording_step, name="total jacobian",
                           file_name=None, units=None,
                           output_dir=os.path.join(self.get_output_dir(), 'total_jacobian'), **plot_params)

    def plot_growth_induced_jacobian(self, recording_step, **kwargs):
        jac = self.get_growth_induced_jacobian(recording_step=recording_step)
        plot_params = {  # "exclude_around" :(0, 0.0001),
                        "range_f": [0.8, 1.2],
                        "colormap": 'RdBu_r',
                        "cmap_ref": 1.0}
        plot_params.update(kwargs)
        self.plot_function(jac, recording_step=recording_step, name="growth induced jacobian",
                           file_name=None, units=None,
                           output_dir=os.path.join(self.get_output_dir(), 'growth_induced_jacobian'), **plot_params)

    def plot_all(self, deformed=False, selection=slice(None), output_dir=None, **kwargs):
        """
        :param deformed: boolean flag for mesh deformation
        :param selection: slice object, e.g. slice(10,-1,5)
        :return:
        """
        if output_dir is not None:
            self.set_output_dir(output_dir)
        if deformed:
            self.set_output_dir(self.get_output_dir() + '_deformed')
        else:
            self.plot_label_function(recording_step=0)
        if type(selection) == slice:
            steps=self._results.get_recording_steps()[selection]
        elif type(selection) == list:
            steps = selection
        else:
            print("cannot handle selection '%s'"%selection)
        for recording_step in steps:
            if deformed:
                self.update_mesh_displacement(recording_step)
                self.plot_label_function(recording_step, **kwargs) # if deformed, plot label function in every time step
            self.plot_concentration(recording_step, **kwargs)
            self.plot_displacement(recording_step, **kwargs)
            self.plot_pressure(recording_step, **kwargs)
            self.plot_displacement_norm(recording_step, **kwargs)
            self.plot_log_growth(recording_step, **kwargs)
            self.plot_total_jacobian(recording_step, **kwargs)
            self.plot_growth_induced_jacobian(recording_step, **kwargs)
            self.plot_van_mises_stress(recording_step, **kwargs)
            self.plot_concentration_deformed_configuration(recording_step, **kwargs)
            if deformed:
                self.update_mesh_displacement(recording_step, reverse=True)

    def plot_for_pub(self, deformed=False, selection=slice(None), output_dir=None, **kwargs):
        """
        :param deformed: boolean flag for mesh deformation
        :param selection: slice object, e.g. slice(10,-1,5)
        :return:
        """
        if output_dir is not None:
            self.set_output_dir(output_dir)
        if deformed:
            self.set_output_dir(self.get_output_dir() + '_deformed')
        else:
            self.plot_label_function(recording_step=0)

        # plot without axes, cbar, etc
        plot_params = {'show_axes': False,
                      'show_ticks': False,
                      'show_title': False,
                      'show_cbar': False}
        plot_params.update(kwargs)
        if type(selection) == slice:
            steps=self._results.get_recording_steps()[selection]
        elif type(selection) == list:
            steps = selection
        else:
            print("cannot handle selection '%s'"%selection)
        for recording_step in steps:
            if deformed:
                self.update_mesh_displacement(recording_step)
                self.plot_label_function(recording_step, n_cmap_levels=4, colormap='Greys_r',
                                         **plot_params) # if deformed, plot label function in every time step
            self.plot_concentration(recording_step, exclude_below=0.01, exclude_min_max=True, show_labels=True,
                                    **plot_params)
            self.plot_displacement(recording_step, **plot_params)
            self.plot_pressure(recording_step, **plot_params)
            self.plot_displacement_norm(recording_step, **plot_params)
            self.plot_log_growth(recording_step, **plot_params)
            self.plot_total_jacobian(recording_step, **plot_params)
            self.plot_growth_induced_jacobian(recording_step, **plot_params)
            self.plot_van_mises_stress(recording_step, **plot_params)
            self.plot_concentration_deformed_configuration(recording_step, **plot_params)
            if deformed:
                self.update_mesh_displacement(recording_step, reverse=True)
        # plot colorbars separately
        plot_params_2 = {'show_axes': False,
                          'show_ticks': False,
                          'show_title': False,
                          'show_cbar': True,
                          'dpi': 600,
                          'cbar_size': '20%',
                          'cbar_pad': 0.2,
                          'cbar_fontsize': 15}
        plot_params_2.update(kwargs)
        step = 1
        self.set_output_dir(os.path.join(self.get_output_dir(), 'cbar'))
        self.plot_label_function(step, n_cmap_levels=4, colormap='Greys_r', **plot_params_2)
        self.plot_concentration(step, **plot_params_2)
        self.plot_displacement(step, **plot_params_2, range_f=[0, 8])
        self.plot_displacement_norm(step, **plot_params_2, range_f=[0, 8])
        self.plot_total_jacobian(step, **plot_params_2)

    def save_all(self, save_method='xdmf', clear_all=False, selection=slice(None), output_dir=None):
        if output_dir is not None:
            self.set_output_dir(output_dir)
        self._results.set_save_output_dir(self.get_output_dir())
        self._results.save_solution_start(method=save_method, clear_all=clear_all)
        for recording_step in self._results.get_recording_steps()[selection]:
            current_sim_time = self._results.get_result(recording_step=recording_step).get_time_step()
            u = self._results.get_solution_function(recording_step=recording_step)
            self._results.save_solution(recording_step, current_sim_time, function=u, method=save_method)
            # try merging those files into single vtu
            if save_method != 'xdmf':
                dio.merge_vtus_timestep(self.get_output_dir(), recording_step, remove=False, reference_file_path=None)
        self._results.save_solution_end(method=save_method)


class PostProcessTumorGrowthBrain(PostProcessTumorGrowth):

    def map_params(self):
        """
        This function maps the parameters defined explicitly in the TumorGrowthBrain class into instances of DiscontinousScalar, so that they can be processed by function defined in PostProcessTumorGrowth.
        :return:
        """
        if not hasattr(self._params, 'E'):
            youngmod = {'outside': 10E6,
                        'CSF': self._params.E_CSF,
                        'WM': self._params.E_WM,
                        'GM': self._params.E_GM,
                        'Ventricles': self._params.E_VENT}
            self._params.set_parameter('E', youngmod)

        if not hasattr(self._params, 'poisson'):
            poisson = {'outside': 0.45,
                        'CSF': self._params.nu_CSF,
                        'WM': self._params.nu_WM,
                        'GM': self._params.nu_GM,
                        'Ventricles': self._params.nu_VENT}
            self._params.set_parameter('poisson', poisson)

        if not hasattr(self._params, 'proliferation'):
            prolif = {'outside': 0.0,
                        'CSF': 0.0,
                        'WM': self._params.rho_WM,
                        'GM': self._params.rho_GM,
                        'Ventricles': 0.0}
            self._params.set_parameter('proliferation', prolif)


class Comparison():

    def __init__(self, sim1, sim2):
        self.sim1 = sim1
        self.sim2 = sim2
        self.difference = Results(sim1.functionspace)
        self._initialise()

    def _initialise(self):
        steps_sim1 = self.sim1.results.get_recording_steps()
        steps_sim2 = self.sim2.results.get_recording_steps()
        intersection = set(steps_sim1).intersection(set(steps_sim2))
        self.shared_recording_steps = list(intersection)
        # for recording_step in list(intersection):
        #     self.compute_difference(recording_step=recording_step)

    def compute_difference(self, recording_step):
        obs = self.sim1.results.get_result(recording_step=recording_step)
        sim_time = obs.get_time()
        time_step = obs.get_time_step()
        u1 = self.sim1.results.get_solution_function(recording_step=recording_step)
        u2 = self.sim2.results.get_solution_function(recording_step=recording_step)
        diff = u1 - u2
        self.difference.add_to_results(current_sim_time=sim_time, current_time_step=time_step,
                                       recording_step=recording_step, field=diff, replace=True)

    def compute_errornorm(self, recording_step):
        u1 = self.sim1.results.get_solution_function(recording_step=recording_step)
        u2 = self.sim2.results.get_solution_function(recording_step=recording_step)
        errnorm = fenics.errornorm(u1, u2)
        return errnorm

    def compute_errornorm_by_subspace(self, recording_step):
        diff_dict = {}
        for subspace_name in list(self.difference._functionspace.subspaces.get_subspace_names()):
            u1_sub = self.sim1.results.get_solution_function(subspace_name=subspace_name, recording_step=recording_step)
            u2_sub = self.sim2.results.get_solution_function(subspace_name=subspace_name, recording_step=recording_step)
            diff_dict[subspace_name] = fenics.errornorm(u1_sub,u2_sub)
        return diff_dict

    def get_difference_by_subspace(self, recording_step):
        diff_dict = {}
        for subspace_name in self.difference._functionspace.subspaces.get_subspace_names():
            diff_dict[subspace_name] = self.difference.get_solution_function(subspace_name=subspace_name, recording_step=recording_step)
        return diff_dict

    def compute_max_difference(self, recording_step):
        if not self.difference.exists_recording_step(recording_step):
            self.compute_difference(recording_step)
        diff = self.difference.get_solution_function(recording_step=recording_step)
        return np.max(diff.vector().array())

    def compare(self, selection=slice(None)):
        df = pd.DataFrame()
        count = 0
        for recording_step in self.shared_recording_steps[selection]:
            df.loc[count, 'recording_step'] = recording_step
            df.loc[count, 'errornorm'] = self.compute_errornorm(recording_step)
            for subspace_name, result in self.compute_errornorm_by_subspace(recording_step).items():
                df.loc[count, 'errornorm_'+subspace_name] = result
            count = count+1
        return df