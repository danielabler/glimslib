from fenics_local import *

# -- logistic growth
def compute_growth_logistic(conc, prolif_rate, conc_max):
    return prolif_rate * conc * ( 1 - conc / conc_max)


# -- coupling concentration to elasticity
def compute_expansion(conc_field, coupling_constant, dim):
    return conc_field * coupling_constant * Identity(dim)
