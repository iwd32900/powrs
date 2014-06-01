"""
Experimental module to expose R's math functions (mostly relating to
statistical distributions) via Python's ctypes library.

After downloading the R source code from CRAN, unpack it and do:

    ./configure --enable-R-static-lib --enable-static --with-readline=no
    cd src/nmath/standalone/
    make
    cp libRmath.dll ~/bioinf/pygr/

"""
import sys
from os import path
from ctypes import *

# Try to load the shared library from this same directory as this file,
# or anywhere on our PYTHONPATH.  Library name depends on system.
ext = dict(cygwin=".dll", win32=".dll", darwin=".dylib").get(sys.platform, ".so")
for base in [path.dirname(__file__)] + sys.path:
    if path.lexists(path.abspath(path.join(base, "libRmath"+ext))):
        _lib = CDLL(path.abspath(path.join(base, "libRmath"+ext)))
        break
else:
    raise ValueError("Can't seem to find libRmath!")
# Save the path information for use with scipy.weave.inline.
# E.g. weave.inline('return_val = pbinom(100, 20000, 100./20000., 0, 1);', **rmath.weave_inline_kwargs)
weave_inline_kwargs = dict(
    support_code="""
    extern "C" {
        /* These lines copied from .c or .h files in R/src/nmath */
        double phyper(double x, double NR, double NB, double n, int lower_tail, int log_p);
        double pbinom(double x, double n, double p, int lower_tail, int log_p);
        double dbinom(double x, double n, double p, int give_log);
        double qbinom(double p, double n, double pr, int lower_tail, int log_p);
        double rpois(double mu);
    }
    """,
    library_dirs=[base],
    runtime_library_dirs=[base],
    # On OS X, one must set DYLD_LIBRARY_PATH (*before* launching Python)
    # or copy the *.dylib(s) to a standard location like /usr/local/lib.
    # `runtime_library_dirs` is not respected.  I think this is a distutils issue,
    # which simply uses -L instead of -R on OS X.  Thus linking works, but runtime fails.
    # Setting DYLD_LIBRARY_PATH using os.environ does NOT seem to work, surprisingly.
    # I also couldn't make -rpath do anything useful when passed to weave.inline's linker.
    libraries=["Rmath"]
    )
del base, ext # so we don't pollute the namespace

_lib.phyper.restype = c_double
_lib.phyper.argtypes = [c_double, c_double, c_double, c_double, c_int, c_int]
def phyper(white_drawn, white_total, black_total, drawn_total, prob_le=True, log_p=False):
    """Returns the probability of drawing white_drawn balls (or fewer) in drawn_total draws
    from an urn with white_total and black_total balls, without replacement.
    If prob_le=False, the probability of drawing more than white_drawn balls.
    """
    return _lib.phyper(white_drawn, white_total, black_total, drawn_total, prob_le, log_p)

_lib.pbinom.restype = c_double
_lib.pbinom.argtypes = [c_double, c_double, c_double, c_int, c_int]
def pbinom(k, samp, prob, prob_le=True, log_p=False):
    """
    Cumulative Distribution Function for the binomial distribution, used to model sampling with replacement.
    If k << samp, it's also a good approximation for the hypergeometric distribution, used to model sampling without replacement.

    k - number of "successes" (rounded to int)
    samp - total number of tries (rounded to int)
    prob - probability of success on any given try
    """
    # Without the calls to round(), can return NaN
    return _lib.pbinom(round(k), round(samp), prob, prob_le, log_p)

_lib.dbinom.restype = c_double
_lib.dbinom.argtypes = [c_double, c_double, c_double, c_int]
def dbinom(k, samp, prob, log_p=False):
    """
    Probability Density Function for the binomial distribution, used to model sampling with replacement.
    If k << samp, it's also a good approximation for the hypergeometric distribution, used to model sampling without replacement.

    k - number of "successes" (rounded to int)
    samp - total number of tries (rounded to int)
    prob - probability of success on any given try
    """
    # Without the calls to round(), can return NaN
    return _lib.dbinom(round(k), round(samp), prob, log_p)

_lib.qbinom.restype = c_double
_lib.qbinom.argtypes = [c_double, c_double, c_double, c_int, c_int]
def qbinom(q, samp, prob, prob_le=True, log_p=False):
    """
    Percentile Point Function for the binomial distribution, used to model sampling with replacement.
    With prob_le=False, becomes the Inverse Survival Function.

    q - quantile, between 0 and 1
    samp - total number of tries (rounded to int)
    prob - probability of success on any given try
    """
    # Without the calls to round(), can return NaN
    return _lib.qbinom(q, round(samp), prob, prob_le, log_p)

_lib.rpois.restype = c_double
_lib.rpois.argtypes = [c_double]
def rpois(mu):
    """Random deviates from the Poisson distribution.  Each call returns one random integer."""
    # Declared type is double, but in practice it's always an integer.
    return int(_lib.rpois(mu))
