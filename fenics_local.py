"""Local wrapper around FENICS and dolfin-adjoint"""

from fenics import *

import config

if config.USE_ADJOINT:
    from dolfin_adjoint import *

