"""
A variety of helpful functions for general programming tasks.
"""

import gzip, sys

Inf = 1e400 # overflows a double-precision float

def gzopen(filename, mode='rb', compresslevel=1):
    '''Uses gzip to open files that end in .gz, and opens other files normally.
    A single dash (-) can be used to represent (uncompressed) stdin or stdout, depending on the mode.
    We default to minimal compression, maximum speed because additional space savings are minimal but processor load is not.
    E.g. on a 25 Mb FASTA file of gene-length segments, level 1 is only 17% larger than level 9,
    but level 1 can be done on the fly (disk bandwidth limited) while level 9 makes the process several fold slower.
    Python defaults to 9;  command-line gzip defaults to 6.
    According to the bzip2 man page, compresslevel doesn't alter the speed bzip2, just the memory requirement.
    '''
    if filename.lower().endswith(".gz"):
        return gzip.open(filename, mode, compresslevel=1)
    elif filename.lower().endswith(".bz2"):
        import bz2 # not available on our Amazon image?
        return bz2.BZ2File(filename, mode)
    elif filename == '-':
        if 'r' in mode: return sys.stdin
        else: return sys.stdout
    else:
        return open(filename, mode)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
