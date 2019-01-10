from ufl import Identity

from fenics_local import *


# -- youngs modulus, poisson ration -> mu, lambda

def compute_mu(young_modulus, poisson_ratio):
    return young_modulus / (2.0 * (1.0 + poisson_ratio))

def compute_lambda(young_modulus, poisson_ratio):
    return young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))

def compute_strain(displacement):
    return sym(grad(displacement))

def compute_stress(displacement, mu, lmbda):
    return 2.0 * mu * compute_strain(displacement) + \
           lmbda * tr(compute_strain(displacement)) * Identity(len(displacement))

def compute_pressure_from_stress_tensor(stress_tensor):
    pressure = 1.0/3.0*tr(stress_tensor)
    return pressure

def u_norm(u):
    return inner(u,u)**0.5

def compute_total_jacobian(displacement):
    return det(Identity(len(displacement)) + grad(displacement))

def compute_growth_induced_jacobian(growth_induced_strain, dim):
    return det(Identity(dim) + growth_induced_strain)

def compute_growth_induced_strain(conc_field, coupling_constant, dim):
    return conc_field * coupling_constant * Identity(dim)

def compute_deviatoric_stress_tensor(stress_tensor, dim):
    return stress_tensor - (1. / 3.) * tr(stress_tensor) * Identity(dim)

def compute_van_mises_stress(stress_tensor, dim):
    dev_stress = compute_deviatoric_stress_tensor(stress_tensor, dim)
    return sqrt(3. / 2. * inner(dev_stress, dev_stress))

def first_invariant(self, tensor):
    """
    First invariant is trace of tensor.
    """
    return tr(tensor)

def second_invariant(self, tensor):
    """
    Second invariant is 1/2*( trace(T)^2 - trace(T^2) )
    """
    return 0.5*(tr(tensor)*tr(tensor) - tr(tensor*tensor))

def third_invariant(self, tensor):
    """
    Third invariant is determinant
    """
    return det(tensor)

def compute_eigenvalues(self, tensor):
    """
    check example
    https://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/eigenvalue/python/documentation.html
    """
    pass

def compute_concentration_deformed(concentration, displacement, coupling_constant, dim):
    jac_total = compute_total_jacobian(displacement)
    strain_growth = compute_growth_induced_strain(concentration, coupling_constant, dim)
    jac_growth = compute_growth_induced_jacobian(strain_growth, dim)
    return concentration * jac_growth / jac_total