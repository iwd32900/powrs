"""
ctypes wrapper around libdivsufsort, a library for suffix arrays

Get libdivsufsort 2.0.1 or later from http://code.google.com/p/libdivsufsort/ and
    mkdir build
    cd build
    ccmake ..  # accept default settings
    make
    cp libs/libdivsufsort.so ~/bioinf/pygr/

If this doesn't pan out, there's another interesting collection of suffix array implementations in
Java at http://labs.carrotsearch.com/jsuffixarrays.html
"""
import sys
from os import path
from ctypes import *
import numpy as np

# Try to load the shared library from this same directory as this file,
# or anywhere on our PYTHONPATH.  Library name depends on system.
ext = dict(cygwin=".dll", win32=".dll", darwin=".dylib").get(sys.platform, ".so")
for base in [path.dirname(__file__)] + sys.path:
    if path.lexists(path.abspath(path.join(base, "libdivsufsort"+ext))):
        _lib = CDLL(path.abspath(path.join(base, "libdivsufsort"+ext)))
        break
else:
    raise ValueError("Can't seem to find libdivsufsort!")
del base, ext # so we don't pollute the namespace

# Need a *int32 data type for libdivsufsort
c_int32_p = POINTER(c_int32)

# Type annotations make ctypes ever so slightly *slower*, oddly enough.
# Posts on the web seem to concur, though.

class DivSufSort(object):
    def __init__(self, corpus):
        """
        corpus      an 8-bit string of bytes to search
        """
        assert isinstance(corpus, str), "Input text must be 8-bit string (not unicode)"
        self.corpus = corpus
        self.suffix_array = np.empty(len(corpus), dtype=np.int32)
        err = _lib.divsufsort(self.corpus, self.suffix_array.ctypes.data_as(c_int32_p), len(corpus))
        if err: raise ValueError("Unable to make suffix array (%i)" % err)
    def find_positions(self, pattern):
        """
        Python method is thread safe (read-only to shared data).
        Thread safety of the C method is unknown.
        """
        assert isinstance(pattern, str), "Input text must be 8-bit string (not unicode)"
        sa_start = c_int32(0) # ordinary Python ints aren't mutable
        num_hits = _lib.sa_search(self.corpus, len(self.corpus),
            pattern, len(pattern),
            self.suffix_array.ctypes.data_as(c_int32_p), len(self.suffix_array),
            byref(sa_start)) # have to cast to pointer / take address of
        # Matches to `pattern` occupy a continuous range in the sorted suffix array.
        # Thus, the search function just returns a start position and number of matches.
        sa_start = sa_start.value # convert back to Python int
        if num_hits < 0: raise ValueError("Error in searching (%i)" % num_hits)
        return self.suffix_array[sa_start:sa_start+num_hits].copy()
    def count_positions(self, pattern):
        assert isinstance(pattern, str), "Input text must be 8-bit string (not unicode)"
        sa_start = c_int32(0) # ordinary Python ints aren't mutable
        num_hits = _lib.sa_search(self.corpus, len(self.corpus),
            pattern, len(pattern),
            self.suffix_array.ctypes.data_as(c_int32_p), len(self.suffix_array),
            byref(sa_start)) # have to cast to pointer / take address of
        # Matches to `pattern` occupy a continuous range in the sorted suffix array.
        # Thus, the search function just returns a start position and number of matches.
        if num_hits < 0: raise ValueError("Error in searching (%i)" % num_hits)
        return num_hits

