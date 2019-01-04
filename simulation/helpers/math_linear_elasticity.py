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