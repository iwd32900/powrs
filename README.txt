POWRS - the POsition-sensitive WoRd Set motif finder
====================================================

Developed by Ian W. Davis and GrassRoots Biotechnology

Terms of use are in LICENSE.txt.
Installation instructions are in INSTALL.txt.
Instructions for use can be obtained by running

    python powrs.py --help

from this directory once dependencies are installed.



Sample data for the CREB transcription factor in the paper has been
included. To reproduce that analysis, do

    cd sample_data
    python ../powrs.py --window-width 50 --length 2000 --improve-limit 800 CREB.in.fa.gz CREB.out.fa.gz > CREB.powrs.txt

The final results of the algorithm will be at the end of this (large)
text file, below the final double line (=============). See the --help
text for instructions on interpretting the output.

DEVELOPMENT HISTORY
===================

* Version 1.2       21 Mar 2013
  Updated release that adds binning to correct for sequence composition bias
  (--bins, in the style of Amadeus) and searching with a fixed-size gap
  in the middle of the motif (--midgap).  The scoring function changed subtly
  to enable binning, but rank orderings of motifs should change very little.

* Version 1.1       24 May 2012
  Updated release that adds multi-processor support (--parallel) and (experimental!)
  support for clustering similar motifs in a post-processing step.
  The main algorithm is unchanged, and pre-clustering results should be identical.

* Version 1.0       7 Nov 2011
  Initial release of the code.  This is the code referenced by the paper,
  and should be used to reproduce any analysis from that work.