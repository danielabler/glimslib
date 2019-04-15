# -- logistic growth
def compute_growth_logistic(conc, prolif_rate, conc_max):
    return prolif_rate * conc * ( 1 - conc / conc_max)

