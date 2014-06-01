#!/usr/bin/env python
import itertools, multiprocessing, os, random, sys, time
from optparse import OptionParser
import numpy as np

# Save people from having to set PYTHONPATH
import os
sys.path.insert(0, os.path.dirname(__file__))

from pygr import util, dnaseq, divsufsort, powrs

# The following functions allow us to parallelize the search with multiprocessing:
def set_globals(*args):
    # This is ugly, but prevents us pickling this (static) data over and over again for multiprocessing
    global data
    data, = args

def calc_seed_kmers((motif, ii, nseeds)):
    kmer = "".join(motif)
    retval = powrs.Motif(kmer, False, data)
    if ii % 100 == 0 and nseeds > 0:
        pct = float(ii) / nseeds
        bar = "=" * int(50*pct)
        space = " " * (50 - len(bar))
        print "[%s%s]    %4.1f%%" % (bar, space, 100*pct)
        #print "[%s%s]    %4.1f%%    %s    %.2f" % (bar, space, 100*pct, kmer, retval.score) # DEBUG
    return retval

def calc_improved_motif(best_m):
    # Returns (new_motif, is_finished)
    alt_ms = [powrs.Motif(k, best_m.both_strands, data, parent=best_m) for k in best_m.other_nbrs]
    alt_ms.append(powrs.Motif(best_m.center, not best_m.both_strands, data, parent=best_m))
    alt_ms.sort(reverse=True)
    if alt_ms[0].score <= best_m.score:
        return (best_m, True) # can't improve this further
    else:
        return (alt_ms[0], False) # we made an improvement

#@util.profile("wedmi.prof")
def main(argv):
    '''usage: %prog [options] IN_GROUP.fa[.gz] OUT_GROUP.fa[.gz]

POWRS (POsition-sensitive WoRd Set) motif identification algorithm
[Formerly known as WEDMI (word edit-distance motif identifier).]

It simultaneously finds motifs and regions that distinguish
in-group sequences from out-group sequences.
Motifs are modeled as a central k-mer of fixed length,
and some or all of the k-mers one mutation away from it,
on one or both strands.

Input FASTA files should look like this:
    >ATxGxxxx   3
    ACTGACTG...

The "identifier" field should be used for a gene name,
and the "name" field should indicate how many copies of that gene are in the file.
Multiple copies generally occur when there are multiple gene models (transcripts)
for a single gene, and all of them get included rather than one picked at random.

Output is to stdout and includes 8 fields:
1.  Score of the motif, -log_10(p-value)
2.  Number of in-group genes matching the motif in the specified window
3.  Seed word (k-mer)
4.  Reverse complement of the seed, if motif occurs on both strands, otherwise dashes
5.  Window start (bp from left edge, modified by -5 and -l)
6.  Window end (bp from left edge, modified by -5 and -l)
7.  Full motif pattern (alternate bases in lower case)
8.  powrs.Motif rank

All selected seeds are optimized together, one cycle at a time.
Output is shown for all seeds during these intermediate cycles,
with cycles separated by a double line (=====================).

For the final output, any motif whose seed is part of a higher-ranking motif
is not printed;  thus, some ranks will be "skipped" in last cycle of output.
'''
    parser = OptionParser(usage=main.__doc__, version="%prog 1.0")
    #parser.add_option("-l", "--long-name",
    #                  action="store|store_true|store_false|store_const|append|count|callback",
    #                  dest="var_name",
    #                  type="int|string|float|...",
    #                  default=True,
    #                  metavar="FILE",
    #                  help="munge FILE (default %default)"
    #                  )
    parser.add_option("-k", "--seed-size", type=int, default=8,
        help="Initial seed k-mer size (default %default)")
    parser.add_option("-g", "--midgap", type=int, default=0,
        help="Search with (%default) bp gap in the middle of k-mer")
    parser.add_option("-w", "--window-width", type=int, default=25,
        help="Granularity for optimizing motif region (default %default bp)")
    parser.add_option("-b", "--bins", type=int, default=1,
        help="Number of bins to use in correcting sequence composition bias (%default)")
    parser.add_option("-p", "--permute-evidence", default=False, action="store_true",
        help="Randomly permute the group assignments, for estimating the null distribution of scores.")
    parser.add_option("-A", "--min-genes", type=int, default=50,
        help="Minimum number of sequences that a valid motif will match (default %default)")
    parser.add_option("-B", "--max-genes-frac", type=float, default=0.20,
        help="Maximum fraction of sequences that a valid motif will match (default %default)")
    parser.add_option("-5", "--align-5p", action="store_true", default=False,
        help="Align on 5' edge of sequences instead of 3' edge")
    parser.add_option('-l', '--length', type=int, default=0,
        help='Maximum promoter length -- only used for formatting output (default %default)')
    parser.add_option('-i', '--improve-limit', type=int, default=1000,
        help='After trying to improve all motifs, dicard all but N of them before starting a new cycle (default %default)')
    parser.add_option('-I', '--improve-score-limit', type=float, default=0.1,
        help="Don't bother trying to improve motifs that score below this (default %default)")
    parser.add_option('-c', '--cluster-limit', type=int, default=200,
        help='Only try to cluster the top N motifs (default %default)')
    parser.add_option('-C', '--cluster-score-limit', type=float, default=6,
        help="Don't bother trying to cluster motifs that score below this (default %default)")
    parser.add_option('-P', '--parallel', action='store_true', default=False,
        help='Use all available processors in parallel to speed computation')
    parser.add_option('-v', '--verbose', action='store_true', default=False)
    
    parser.add_option('--save', help='For debugging only.')
    parser.add_option('--load', help='For debugging only.')
    
    (options, args) = parser.parse_args(argv)
    
    if len(args) == 2:
        ingrp_file = util.gzopen(args[0], 'rb')
        outgrp_file = util.gzopen(args[1], 'rb')
    else:
        parser.print_help()
        print "Too many/few arguments!"
        return 1
    
    print "Loading sequences ..."
    T = time.time()
    fasta = list(dnaseq.read_fasta(ingrp_file))
    out_fasta = list(dnaseq.read_fasta(outgrp_file))
    fasta += out_fasta
    evidence = np.zeros(len(fasta))
    evidence[:-len(out_fasta)] = 1.
    assert len(evidence) == len(fasta), "%i != %i" % (len(evidence), len(fasta))
    
    if options.permute_evidence:
        fasta_ids = [i for i,n,s in fasta]
        if set(fasta_ids[:-len(out_fasta)]) & set(fasta_ids[-len(out_fasta):]):
            print "In-group and out-group sequence IDs overlap -- reverting to simple shuffle!"
            # In the worst case of complete overlap, all evidence gets set to 0 using the "smart" algo!
            np.random.shuffle(evidence)
            # TODO: this could screw up the weights, if they're not all equal to start with...
        else:
            print "Shuffling evidence by sequence ID ..."
            # Randomly shuffle the evidence based on gene IDs, but so that all gene models
            # for the same gene retain the same evidence.  Thus weights are unaffected.
            ev_map = dict((i,e) for (i,n,s), e in zip(fasta, evidence))
            keys = ev_map.keys()
            random.shuffle(keys)
            ev_map = dict(zip(keys, ev_map.values()))
            for ii, (ident, name, seq) in enumerate(fasta):
                evidence[ii] = float(ev_map[ident])
   
    # Need weights because some genes are represented by multiple gene models.
    evidence_weights = np.array([1./int(n) for i,n,s in fasta])
    # Evidence is pre-multiplied by the weights to save computations:
    evidence *= evidence_weights
    # Pre-calculate constants used in the cERMIT score function:
    G = evidence_weights.sum() # total "number" of genes
    mu = evidence.sum() / G # average evidence for all genes
    print "G = %f    mu = %f" % (G, mu)
    # G and mu "should" be vectors or matrices to account for the fact that some genes
    # are not full length.  However, the number of genes within any window is not constant
    # across the length of the window, particularly if it's long.
    # So for now, I'm just going to ignore this problem.  Results are still reasonable.
    
    # This will be input to numpy.searchsorted() --
    # add one to each length to account for the newlines in the file.
    # Using searchsorted() is expensive, but padding all sequences to
    # the same length with N's can make suffix array creation VERY expensive.
    # Whether we're searching both strands or just one, we only write one sequence to the suffix array.
    # Writing the sequence and its reverse complement makes indexing complicated,
    # so instead we take the reverse complement of the search motif, which is simpler here.
    seq_lens = np.array([len(s)+1 for i,n,s in fasta])
    seq_offsets = seq_lens.cumsum() - 1 # the 0-based index within the file at which each sequence ends (just past last base)
    # As an alternative to binary search, without padding all genes to the same length --
    # maintain a lookup table that maps positions in the file to sequence numbers.
    # This table will require one int per byte in the file, or about 4x as large as the sequence data.
    # To reduce the size, pad all sequences so their total length (plus trailing newline)
    # is evenly divisible by e.g. 32.  Then divide indexes from the suffix array search
    # by 32 before doing the lookup, and thus the lookup table can be 32x smaller,
    # while ensuring that no sequence is padded with more than 31 "N" bases.
    # However, I don't think that searchsorted() is a major bottleneck anymore, and so
    # this scheme hasn't been implemented yet because the performance gains would be small.
    
    # Cluster sequences ala Linhart et al for binned enrichment.
    # This can (partially) correct for differences in base composition between in-group and out-group.
    bins = powrs.SeqBins(fasta, evidence, evidence_weights, n_bins=options.bins)
    
    # Sequences now held in memory instead of being written to a tmp file:
    out_seq = []
    for ident, name, seq in fasta:
        out_seq.append(seq)
        out_seq.append("\n")
    out_seq = "".join(out_seq)
    print time.time() - T, "seconds"
    print "Building suffix array ..."
    T = time.time()
    suf = divsufsort.DivSufSort(out_seq)
    print time.time() - T, "seconds"
    # "global" data needed for calculating scores
    data = dict(
        bins=bins,
        evidence=evidence,
        ev_wts=evidence_weights,
        G=G, mu=mu,
        suf=suf,
        seq_lens=seq_lens, seq_offsets=seq_offsets,
        options=options)
    
    # Set up for multiprocessing
    if options.parallel:
        os.nice(10) # reduce our priority, in case the user forgot to run us with "nice"
        pool = multiprocessing.Pool(None, set_globals, [data])
        map_func = lambda f,i: pool.imap_unordered(f,i)
    else:
        set_globals(data)
        map_func = itertools.imap
    
    if options.load:
        all_motifs = util.gzunpickle(options.load)
    else:
        # Seed our search with small k-mers
        print "Searching for all k-mers ..."
        T = time.time()
        seeds_iter = list(enumerate(itertools.product("ACGT", repeat=options.seed_size)))
        if options.verbose: nseeds = 4.**options.seed_size
        else: nseeds = 0 # don't print progress
        motif_iter = map_func(calc_seed_kmers, [(motif, ii, nseeds) for ii, motif in seeds_iter])
        # filtering by score on the fly reduces memory consumption when --allow-N and --seed-size are large
        all_motifs = [motif for motif in motif_iter if motif.score >= options.improve_score_limit]
        if len(all_motifs) < options.improve_limit:
            print "***  Only %i motifs are candidates for improvement  ***" % len(all_motifs)
            print "***  Lower --improve-score-limit or --improve-limit  ***"
        
        # Refine the best seeds until they can't be further improved
        print "Refining best motifs ..."
        def print_best(prune=False, bar=True):
            used_kmers = set()
            for ii, best_m in enumerate(all_motifs[:options.improve_limit]):
                if prune and best_m.center in used_kmers: continue
                print "%s    #%i" % (best_m, ii+1)
                used_kmers.update(best_m.all_kmers)
            if bar: print "="*80
        
        finished_motifs = set()
        while True:
            all_motifs.sort(reverse=True)
            if options.verbose: print_best()
            improved_motifs = list(map_func(calc_improved_motif, set(all_motifs[:options.improve_limit]) - finished_motifs))
            new_motifs = list(finished_motifs) # we won't adjust them, but they take up slots
            keep_going = False
            for best_m, is_finished in improved_motifs:
                if is_finished: finished_motifs.add(best_m)
                else: keep_going = True
                new_motifs.append(best_m)
            all_motifs = new_motifs
            if not keep_going: break
        
        all_motifs.sort(reverse=True)
        # Unpruned (final) output can be helpful, even if we're not verbose
        if not options.verbose: print_best()
        # Final print-out, eliminating close relatives
        print_best(prune=True)
        
        if options.save:
            util.gzpickle(all_motifs, options.save)
    # end save/load block
    
    # begin clustering
    # Opportunistic clustering -- highest-scoring clusters get first crack at improving themselves.
    all_clust = [powrs.Cluster(m, data) for m in all_motifs if m.score >= options.cluster_score_limit]
    all_clust = all_clust[:options.cluster_limit]
    merge_memo = {} # {(clust1, clust2) : new_clust}
    print "Trying to merge %i motifs ..." % len(all_clust)
    while True:
        for clust1, clust2 in itertools.combinations(all_clust, 2):
            lost_motifs = []
            if (clust1, clust2) in merge_memo:
                new_clust = merge_memo[clust1, clust2]
            elif (clust2, clust1) in merge_memo:
                assert False, "I don't think this can ever happen"
                new_clust = merge_memo[clust2, clust1]
            else:
                new_clust = merge_memo[clust1, clust2] = clust1.try_merge(clust2)
                # Single stranded motifs are dropped when we take the revcomp,
                # so they will be "lost" to the clustering process at this stage,
                # unless we capture them and return them to the pool.
                if new_clust is None and clust2.revcomp is not None:
                    new_clust = merge_memo[clust1, clust2] = clust1.try_merge(clust2.revcomp)
                    lost_motifs = clust2.revcomp.lost_motifs
                if new_clust is None and clust1.revcomp is not None:
                    new_clust = merge_memo[clust1, clust2] = clust1.revcomp.try_merge(clust2)
                    lost_motifs = clust1.revcomp.lost_motifs
            if new_clust is None: continue
            all_clust.remove(clust1)
            all_clust.remove(clust2)
            all_clust.append(new_clust)
            # If we dropped single stranded motifs from the pool due to a revcomp,
            # return them to the pool now.
            for m in lost_motifs:
                all_clust.append(powrs.Cluster(m, data))
            all_clust.sort(reverse=True)
            if options.verbose: print "Merged %s and %s" % (clust1, clust2)
            break
        else:
            break # nothing was merged, quit trying
    
    used_seeds = set()
    used_kmers = set()
    cluster_num = 1
    for clust in all_clust:
        if len(clust) == 1:
            motif = clust.motifs[0] # the only one
            # Suppress singleton rev. comps. of motifs that were clustered:
            if motif.center in used_seeds: continue
            # These are mostly uninteresting -- could have been merged, but failed to improve the score.
            # (Or had a non-overlapping range, but that's pretty unlikely.)
            if motif.center in used_kmers: continue
            print "-"*80
            print motif
        else:
            max_off = max(clust.offsets)
            spacer = " " * (2*(options.seed_size + max_off) + 3)
            fname = "cluster_%03i_%i_%i.seq" % (cluster_num, clust.start_user, clust.end_user)
            print "-"*80
            print "%8.2f    [%6.1f]    %s    %6i    %6i    (%i motifs in %s)" % (
                clust.score, clust.evidence, spacer, clust.start_user, clust.end_user, len(clust), fname)
            for motif, offset in zip(clust.motifs, clust.offsets):
                print motif.as_str(offset, max_off)
                used_seeds.add(motif.center)
                if motif.both_strands: used_seeds.add(dnaseq.reverse_complement(motif.center))
            f = open(fname, "wb")
            for seq in powrs.extract_positive_seqs(clust, evidence, suf, seq_lens, seq_offsets, options.align_5p, options.midgap):
                f.write(seq)
                f.write("\n")
            f.close()
            cluster_num += 1
            used_kmers.update(clust.all_kmers)
    print
    print "Try:    for f in cluster_*.seq; do seqlogo.sh $f ${f/.seq/.pdf}; done"
    
    if np.isneginf(all_motifs[0].score):
        print
        print "***  Try adjusting -A and -B to eliminate -inf scores  ***"
    
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

