"""Reference parameters of the SIMPLE descriptor implementation.

This subpackage is the *reference implementation* validated in
docs/SIMPLE/SIMPLE.tex: the working parameters below are pinned to the
values used throughout that document (and by the validation suite in
tests/simple/). A production implementation would expose them as
configuration; here they are module constants so that every numerical
result in the document is reproduced exactly.
"""

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

R_C = 3.0 * BOHR_PER_ANGSTROM  # window radius: 3 Angstrom in bohr
XI_TARGET = 2.0                # dimensionless cutoff target xi*
N_OUT = 8                      # exposed radial channels
RHO_MIN = 1e-10                # smooth density floor
L_MAX = 3                      # angular momentum cutoff
