"""Local wrapper around FENICS and dolfin-adjoint"""

from dolfin import *
from dolfin import __version__

import config

if config.USE_ADJOINT:
    from dolfin_adjoint import *


def is_version(comparison_str):
    # target
    comp = comparison_str[0]
    comp_version = comparison_str[1:]
    comp_version_year, comp_version_major, _ = comp_version.split('.')
    # actual
    version = __version__
    version_year, version_major, _ = version.split('.')

    if comp == '=':
        return (version_year == comp_version_year) and (version_major == comp_version_major)
    elif comp == '>':
        return (int(version_year) > int(comp_version_year))
    elif comp == '<':
        return (int(version_year) < int(comp_version_year))