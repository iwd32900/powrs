"""
Motif discovery, incorporating position sensitivity and motif degeneracy.
The basic model is similar to the original BIRT:  for a given (degenerate) motif,
find the window of positions that maximize the significance by the binomial test.

Here, the degeneracy is modeled as one canonical k-mer, and some or all of the 3k+9
"neighboring" k-mers that differ from it by just one base.
The intuition is that each substitution carries an increasing cost;  being one mutation
from consensus isn't as bad as being 3 or 4 mutations away from the consensus sequence.

This machinery is derived from my original "Vermit" implementation of the cERMIT algorithm,
which focused on log-odds "evidence" for each gene, with the in-group/out-group version
as a special case.

However, in this code I'm using binomial distribution p-values, which come only from in-group/
out-group count data (evidence = 0 or 1).  So I kept the machinery, but the nominclature is
not what I would have done if I started with this approach!
"""
import copy
import numpy as np
from scipy import weave

from pygr import util, dnaseq, rmath

def kmer_matches(kmer, suf, seq_offsets, midgap):
    """
    locs    0-based start position of matches to `kmer` within the concatenated sequences.
    matches 0-based index of corresponding sequence for each entry in `locs`.
    midgap  Number of bases between the first and second half of `kmer`
    """
    # Caching speeds the single-processor algorithm by ~7%,
    # but eliminating it frees up a LOT of memory to allow multithreading.
    if midgap == 0:
        locs = suf.find_positions(kmer) # offsets from beginning of file
        matches = np.searchsorted(seq_offsets, locs) # sequence index for each match
    else:
        half = len(kmer) // 2
        locs_left = suf.find_positions(kmer[:half])
        locs_right = suf.find_positions(kmer[half:]) - (half+midgap)
        locs = np.intersect1d(locs_left, locs_right, assume_unique=True)
        matches = np.searchsorted(seq_offsets, locs)
        matches_right = np.searchsorted(seq_offsets, locs + (half+midgap))
        same_seq = (matches == matches_right) # only valid when two halves of the match are in the same sequence!
        locs = locs[same_seq]
        matches = matches[same_seq]
    return (locs, matches)

def kmers_matches(kmers, suf, seq_offsets, midgap):
    # Gather together all matches/locations for all kmers.
    # Sort them, first by sequence ID (implicitly), then by position within the sequence.
    matches = [] # [np.array], will be joined all at once
    locs = [] # [np.array], will be joined all at once
    for kmer in kmers:
        ls, ms = kmer_matches(kmer, suf, seq_offsets, midgap)
        matches.append(ms)
        locs.append(ls)
    matches = np.concatenate(matches)
    locs = np.concatenate(locs)
    # There cannot be any duplicates in locs (assuming all kmers are unique, which they should be)
    # because there is exactly one distinct kmer at each location in the original sequences.
    o = np.argsort(locs)
    return locs[o], matches[o]

def filelocs_to_seqlocs(locs, matches, seq_lens, seq_offsets, align_5p):
    """
    Converts file-based kmer locations to sequence-based kmer locations.
    `locs` is altered *IN PLACE*, so there is no return value.
    """
    max_seq_len = seq_lens.max() - 1 # b/c they're really seq. len. + 1 for the newline
    # seq_offsets[matches] - locs == distance from right edge, <= 0
    locs -= seq_offsets[matches]
    if align_5p:
        # sequences are left-justified and 0 is the leftmost base:
        locs += seq_lens[matches]
        locs -= 1 # b/c they're really seq. len. + 1 for the newline
    else:
        # sequences are right-justified and 0 is the leftmost base:
        #locs = max_seq_len - (seq_offsets[matches] - locs)
        locs += max_seq_len
    #if not (locs.max() < max_seq_len):
    #    import pdb
    #    pdb.set_trace()
    assert locs.max() < max_seq_len
    assert locs.min() >= 0
    return max_seq_len

class SeqBins(object):
    def __init__(self, fasta, evidence, ev_wts, n_bins=1):
        self.n_bins = n_bins
        if self.n_bins <= 1:
            self.bin_ids = np.array([0]*len(evidence))
        else:
            self.bin_ids = self._cluster(fasta, n_bins)
        # The genes are divided into n bins. Let Bi and Ti be the BG and target set genes, respectively,
        # in the ith bin, and denote by bi the subset of genes from Bi whose sequence contains a hit. The
        # goal of this score is to account for cases where the fraction of targets is uneven across bins.
        # Suppose that targets within each bin are selected uniformly. Then, in bin i the probability that
        # a selected gene will contain a hit (i.e., belong to bi) is |bi|/|Bi|. Since the fraction of
        # targets in bin i is |Ti|/|T|, it follows that the probability that a selected gene will contain
        # a hit is p_m = sum_over_i[ (Ti/T) * (bi/Bi) ]
        Bi = np.array([ev_wts[self.bin_ids == cid].sum() for cid in xrange(self.n_bins)]) # total genes in each bin
        Ti = np.array([evidence[self.bin_ids == cid].sum() for cid in xrange(self.n_bins)]) # in-group genes in each bin
        T = evidence.sum() # total in-group genes across all bins
        Bi[Bi == 0] == 1 # to avoid divide-by-zero errors when Ti == Bi == 0 (no genes in bin)
        self.multiplier = (Ti / T / Bi)[None,None,:] # to be multiplied by bi, which is called wt_table in improve_window
    def _get_features(self, seq):
        features = []
        features.append(len(seq))
        bases = util.make_bag(seq)
        L = float(max(1, len(seq)))
        features.extend([bases[b]/L for b in 'ACGT'])
        return features
    def _cluster(self, fasta, n_bins):
        print "Clustering sequences by composition and length..."
        # Compute numeric features for each sequence.
        features = [self._get_features(s) for i,n,s in fasta]
        # Normalize distances by dividing by std. deviation.
        features = np.array(features, np.float)
        stddev = np.apply_along_axis(np.std, 0, features)
        print "S.d. of features:", stddev
        features /= stddev
        # For now, we just run the clustering once, even though it may be sub-optimal
        clusters, cost = util.kmeans(features, n_bins, maxiter=10)
        return np.asarray(clusters)

def improve_window(kmers, bins, evidence, ev_wts, G, mu, suf, seq_lens, seq_offsets, options, mask=None):
    """
    Evaluate the score for a set of k-mers, restricted to various possible sequence windows.
    Choose the window giving the highest Z score for these k-mers.

    kmers       Iterable of 1+ k-mers to find the optimal cERMIT-score window for
    bins        a SeqBins object defining how sequences are grouped into bins/clusters
    evidence    NumPy array of log-odds or log-enrichment scores for each gene in the pool (floats)
    ev_wts      NumPy array of weights on (0,1] for each piece of evidence, 1/(# gene models)
    G           total "number" of genes, ev_wts.sum() [to avoid recomputing every time]
    mu          average evidence, evidence.sum()/G [to avoid recomputing every time]
    suf         the suffix array searcher object
    seq_lens    length of each sequence in characters, including the newline separator
    seq_offsets position in the concatenated sequence file at which each sequence ends
    options     command line options object
    mask        Numpy array with length >= max location index; counts the number of times each bp is "used" in a motif.
                This is optional, for masking out kmers used in one motif so another motif can't claim them.
                If present, it will be updated to mask out the evidence used for these kmers.
    
    Raw evidence numbers should have been pre-multiplied by the provided weights (for efficiency).
    Scoring follows M.A.Newton et al, Annals of Applied Statistics 2007, 1:85-106
    rather than Georgiev et al. because the latter appears to have some typos in the formulae.
    Higher scores are better.
    """
    # Hard to find a good word for the SeqBins, where sequences have been grouped/clustered/binned
    # by sequence composition (percent A/C/G/T).  In this function "bin" is already used for
    # the sequence position windows (typically of ~25bp length).
    # Elsewhere in POWRS, there is some post-process "clustering" that already uses that name.
    # But within this function, I'll call the SeqBins "clusters" to distinguish them from
    # the positional bins.
    cluster_ids = bins.bin_ids # for access by Weave
    n_clust = bins.n_bins
    # Gather together all matches/locations for all kmers.
    # Sort them, first by sequence ID (implicitly), then by position within the sequence.
    # The algorithm below expects that locations occur in sorted order.
    #T = time.time()
    locs, matches = kmers_matches(kmers, suf, seq_offsets, options.midgap)
    # "Masking" to reduce the score of alternative motifs that closely match a previously evaluated motif.
    if mask is not None:
        motif_len = max(len(kmer) for kmer in kmers) # should all be the same length, but just in case...
        assert len(mask) >= locs[-1] + motif_len # locs are sorted, so -1 is max element
        loc_mask = np.zeros(locs.shape, dtype=np.int) # each pos. varies from 0 (totally free) to motif_len (totally masked)
        mask_limit = options.mask_limit
        weave.inline(r"""
            int n = Nlocs[0]; // see Numpy book; == locs.shape in Python
            int klen = motif_len;
            int maskmax = mask_limit;
            int ii, jj, kk, loc;
            for(ii = 0; ii < n; ii++) { // for each location...
                loc = locs[ii];
                for(jj = 0; jj < klen; jj++) {
                    if(mask[loc+jj] >= maskmax) loc_mask[ii]++; // if it's already maxed out, we can't use it (maybe)
                    else mask[loc+jj]++; // otherwise, note that we've used one count worth
                }
            }
        """, ['locs','mask','motif_len','loc_mask','mask_limit'])
        loc_mask = (loc_mask <= motif_len//2) # mask out locations that are more than half used by someone else
        locs = locs[loc_mask]
        matches = matches[loc_mask]
    # locs/matches could be empty, possibly due to masking
    if len(locs) == 0: return 0, 0, -util.Inf, 0
    # This sort of complicated indexing creates copies (unlike simple slices):
    # We need a copy (of locs, at least) because we're going to modify it in place, and it's cached. [not true anymore]
    assert locs.base == matches.base == None
    # convert file-based offsets into sequence-based offsets
    max_seq_len = filelocs_to_seqlocs(locs, matches, seq_lens, seq_offsets, options.align_5p) # locs altered in place!
    #print "Finding %i matches: %f" % (len(locs), time.time()-T); T = time.time()
    # Tally kmer locations in fixed-width bins, then build a table
    # of start,end windows showing the total (weighted) evidence
    # of genes with 1+ copies of the kmer located within that window.
    W = options.window_width
    max_W = max(1, max_seq_len // W) # in case W > max_seq_len, so we don't get a 0x0 table
    ev_table = np.zeros((max_W,max_W), dtype=np.float) # sum evidence for selected genes
    wt_table = np.zeros((max_W,max_W,n_clust), dtype=np.float) # "number" of selected genes, a.k.a. "m"
    locs //= W # convert base-pair locations into binned locations
    # now we'll subtract out evidence for genes that don't have a kmer in that window:
        ## This loop is by far the slowest part of this algorithm, so...
        #last_loc, last_seqid = None, None
        #for locbin, seqid in zip(locs, matches):
        #    if seqid != last_seqid: # first kmer of a new sequence
        #        ev_table += evidence[seqid]
        #        wt_table += ev_wts[seqid]
        #        if last_seqid is not None: # last kmer of the old sequence
        #            ev_table[last_loc:max_W,last_loc:max_W] -= evidence[last_seqid]
        #            wt_table[last_loc:max_W,last_loc:max_W] -= ev_wts[last_seqid]
        #        ev_table[0:locbin,0:locbin] -= evidence[seqid]
        #        wt_table[0:locbin,0:locbin] -= ev_wts[seqid]
        #        last_seqid = seqid
        #    else: # non-first kmer
        #        ev_table[last_loc:locbin,last_loc:locbin] -= evidence[seqid]
        #        wt_table[last_loc:locbin,last_loc:locbin] -= ev_wts[seqid]
        #    # We add one so that only bins *between* kmers will be penalized,
        #    # not bins *containing* kmers. In NumPy if start > end, it's just an empty array.
        #    last_loc = locbin + 1
        #if last_seqid is not None: # last kmer of the last sequence
        #    ev_table[last_loc:max_W,last_loc:max_W] -= evidence[last_seqid]
        #    wt_table[last_loc:max_W,last_loc:max_W] -= ev_wts[last_seqid]
    ## After re-writing in C++, the time to actually find the k-mers dominates!
    weave.inline(r"""
        int maxw = max_W; // need this to avoid int/float ambiguity error
        int nclust = n_clust;
        int last_loc = -1, last_seqid = -1, last_clust = -1;
        int n = Nlocs[0]; // see Numpy book; == locs.shape in Python
        int locbin, seqid, clust;
        float evid, evwt, last_evid, last_evwt;
        int ii, jj, kk;
        for(ii = 0; ii < n; ii++) {
            locbin = locs[ii];
            seqid = matches[ii];
            clust = cluster_ids[seqid];
            evid = evidence[seqid];
            evwt = ev_wts[seqid];
            if(seqid != last_seqid) {
                // Only need to do the diagonal and above...
                // Finish up last sequence:
                if(last_seqid != -1) {
                    for(jj = last_loc; jj < maxw; jj++) { for(kk = jj; kk < maxw; kk++) {
                        ev_table[jj*maxw + kk] -= last_evid;
                        wt_table[(jj*maxw + kk)*nclust + last_clust] -= last_evwt;
                    }}
                }
                // Now start this new sequence:
                // Add in evidence everywhere (happens ONCE per SEQUENCE)
                for(jj = 0; jj < maxw; jj++) { for(kk = jj; kk < maxw; kk++) {
                    ev_table[jj*maxw + kk] += evid;
                    wt_table[(jj*maxw + kk)*nclust + clust] += evwt;
                }}
                // Remove evidence between left edge and this occurrence
                for(jj = 0; jj < locbin; jj++) { for(kk = jj; kk < locbin; kk++) {
                    ev_table[jj*maxw + kk] -= evid;
                    wt_table[(jj*maxw + kk)*nclust + clust] -= evwt;
                }}
                last_seqid = seqid;
            } else {
                // Continuation of previous sequence;  evidence has already been added.
                // Just remove evidence between last occurrence and this occurrence.
                for(jj = last_loc; jj < locbin; jj++) { for(kk = jj; kk < locbin; kk++) {
                    ev_table[jj*maxw + kk] -= evid;
                    wt_table[(jj*maxw + kk)*nclust + clust] -= evwt;
                }}
            }
            last_loc = locbin + 1;
            last_clust = clust;
            last_evid = evid;
            last_evwt = evwt;
        }
        if(last_seqid != -1) {
            clust = cluster_ids[last_seqid];
            evid = evidence[last_seqid];
            evwt = ev_wts[last_seqid];
            for(jj = last_loc; jj < maxw; jj++) { for(kk = jj; kk < maxw; kk++) {
                ev_table[jj*maxw + kk] -= evid;
                wt_table[(jj*maxw + kk)*nclust + clust] -= evwt;
            }}
        }
    """, ['locs', 'matches', 'cluster_ids', 'evidence', 'ev_wts', 'max_W', 'n_clust', 'ev_table', 'wt_table'])
    #print "For loop over %i matches: %f" % (len(locs), time.time()-T); T = time.time()
    # Instead of the cERMIT Z-score, we can compute the binomial p-value (using in-group/out-group evidence only):
    # Trying to use non-integer valued floats in pbinom() produces NaNs
    score_table = np.around(ev_table)
    prob_table = (wt_table * bins.multiplier).sum(axis=2) # probability weighted by bin membership
    wt_table = wt_table.sum(axis=2) # condense clusters, for normal use by later code
    wt_table = np.around(wt_table)
    # This code used to compute pbinom(Ak, Nk, A/N)
    # In Amadeus and historically (tally_all), we use pbinom(Ak, A, Nk/N) instead.
    # It would be easy to use either one here, really, without changing much (see commented-out lines below).
    # P-values are quite similar, although pbinom(Ak, A, Nk/N) seems to be a little stronger in each case.
    # I'm not sure which one is better justified statistically, but top hits are nearly identical empirically.
    # We use the Amadeus version now so that we can use their binned enrichment correction.
    weave.inline(r"""
        int maxw = max_W; // need this to avoid int/float ambiguity error
        int ii, jj, kk;
        double prob = mu;
        double N = G;
        double A = (int)(prob * N);
        for(jj = 0; jj < maxw; jj++) { for(kk = jj; kk < maxw; kk++) {
            ii = jj*maxw + kk; // linear index into score_table / wt_table
            // pbinom(Ak, Nk, A/N):
            //score_table[ii] = -0.4342945 * pbinom(score_table[ii]-1, wt_table[ii], prob, 0, 1); // -log_10(P): bigger is better
            // pbinom(Ak, A, Nk/N):
            score_table[ii] = -0.4342945 * pbinom(score_table[ii]-1, A, prob_table[ii], 0, 1); // -log_10(P): bigger is better
        }}
    """, ['mu', 'max_W', 'score_table', 'prob_table', 'G'],
    **rmath.weave_inline_kwargs)
    # Disqualify bins with too few / too many hits:
    min_genes = options.min_genes
    max_genes = options.max_genes_frac * G
    score_table[wt_table < min_genes] = -util.Inf
    score_table[wt_table > max_genes] = -util.Inf
    # Almost there. Find the highest scoring, longest, rightmost window in which start <= end.
    # (Start and end are both inclusive when we read them out from the evidence table).
    olderr = np.seterr(all='ignore') # I think triu() is implemented with a multiplication...
    score_table = np.triu(score_table, k=0) # zero out everything below the diagonal (ensures start <= end) (except NaNs!)
    np.seterr(**olderr)
    max_Z = np.nanmax(score_table)
    # This was caused originally by non-integer weightings on genes -- leaving it here just in case.
    if np.isnan(max_Z):
        print "Failed with %i locs for %i kmers with -- %s" % (len(locs), len(kmers), kmers)
        return 0, 0, -util.Inf, 0
    peaks = [(end-start, end, start) for start, end in zip(*np.nonzero(score_table == max_Z))]
    peaks.sort(reverse=True)
    first_bin, last_bin = peaks[0][2], peaks[0][1]
    #score_table[~np.isfinite(score_table)] = 0 # to make plotting nicer
    #print "Post processing: %f" % (time.time()-T); T = time.time()
    # Return [inclusive, exclusive) base pair indices starting from 0 == left edge
    return first_bin*W, (last_bin+1)*W, max_Z, ev_table[first_bin,last_bin]

class Motif(object):
    def __init__(self, kmer, both_strands, data, parent=None):
        """
        `data` is a dictionary containing the following keys:
            evidence    NumPy array of 1's and 0's marking in-group and out-group, respectively (floats)
            ev_wts      NumPy array of weights on (0,1] for each piece of evidence, 1/(# gene models)
            G           total "number" of genes, ev_wts.sum() [to avoid recomputing every time]
            mu          average evidence, evidence.sum()/G [to avoid recomputing every time]
            suf         the suffix array searcher object
            seq_lens    length of each sequence in characters, including the newline separator
            seq_offsets position in the concatenated sequence file at which each sequence ends
            options     command line options object
        """
        if parent:
            self.center = parent.center # the central kmer
            self.friends = parent.friends | set([kmer]) # other included neighboring kmers
            self.other_nbrs = parent.other_nbrs - set([kmer]) # neighboring kmers not yet included
        else:
            self.center = kmer
            self.friends = set()
            self.other_nbrs = generate_neighbors(kmer)
        self.both_strands = both_strands
        # Don't want to save whole data object or pickled Motifs will be huge...
        self.options_length = data['options'].length
        self._pattern = None
        # Scoring
        self.all_kmers = set([self.center]) | self.friends
        if self.both_strands: self.all_kmers |= set(dnaseq.reverse_complement(m) for m in self.all_kmers)
        self.start, self.end, self.score, self.evidence = improve_window(kmers=self.all_kmers, **data)
        # Using center_score means that the preferred strand version will sort to the top at the end!
        if parent: self.center_score = parent.center_score
        else: self.center_score = self.score
        # When sorting, prefer both-strands version when scores are equal (e.g. palindromic k-mers)
        self.sortkey = (self.score, self.both_strands, self.center_score, self.end, self.start, self.pattern)
    @property
    def start_user(self): return self.start - self.options_length
    @property
    def end_user(self): return self.end - self.options_length
    def __hash__(self): # Motifs are uniquely identified by their string pattern, but may get different scores due to re-scoring
        return hash(self.sortkey)
    def __cmp__(self, other):
        return cmp(self.sortkey, other.sortkey)
    def __str__(self):
        return self.as_str()
    def as_str(self, shift=0, max_shift=0):
        s1 = " "*shift
        s2 = " "*(max_shift - shift)
        if self.both_strands: rc_seed = dnaseq.reverse_complement(self.center)
        else: rc_seed = "-" * len(self.center)
        return "%8.2f    [%6.1f]    %s%s%s / %s%s%s    %6i    %6i    %50s" % (
            self.score, self.evidence,
            s1, self.center, s2, s2, rc_seed, s1, self.start_user, self.end_user, self.pattern)
    @property
    def revcomp(self):
        """Returns the reverse complement of this motif (or None if not on both strands)."""
        if not self.both_strands: return None
        elif self.center == dnaseq.reverse_complement(self.center): return self # palindrome
        elif not hasattr(self, "_revcomp"):
            rc = self._revcomp = copy.copy(self)
            rc.center = dnaseq.reverse_complement(self.center)
            rc.friends = set(dnaseq.reverse_complement(f) for f in self.friends)
            rc.other_nbrs = set(dnaseq.reverse_complement(o) for o in self.other_nbrs)
            rc._pattern = None
            rc.sortkey = (rc.score, rc.both_strands, rc.center_score, rc.end, rc.start, rc.pattern)
        return self._revcomp
    @property
    def pattern(self):
        if self._pattern is None:
            self._pattern = ""
            for pos, base in enumerate(self.center):
                alts = set(f[pos] for f in self.friends)
                alts.discard(base)
                if not alts:
                    self._pattern += base
                else:
                    self._pattern += "[" + base + "".join(sorted(a.lower() for a in alts)) + "]"
        return self._pattern

def generate_neighbors(kmer):
    nbrs = set()
    for pos, base in enumerate(kmer):
        for alt in "ACGT":
            if alt != base:
                assert base != "N"
                nbrs.add(kmer[:pos] + alt + kmer[pos+1:])
    return nbrs

class Cluster(object):
    """A group of similar Motif objects."""
    def __init__(self, motif, data):
        self.motifs = [motif]
        self.lost_motifs = [] # used only in reverse complement operations...
        self.offsets = [0]
        self.all_kmers = motif.all_kmers
        self.start = motif.start
        self.end = motif.end
        self.score = motif.score
        self.evidence = motif.evidence
        # Need to keep `data` for merge scoring and revcomp scoring,
        # but it could make pickles holding Clusters quite large...
        self.data = data
    @property
    def start_user(self): return self.start - self.data['options'].length
    @property
    def end_user(self): return self.end - self.data['options'].length
    def try_merge(self, other):
        """Returns either a new Cluster or None."""
        if not overlaps(self, other): return None
        kmers_linked = False
        # First try exact alignment
        for m1, o1 in zip(self.motifs, self.offsets):
            if kmers_linked: break
            for m2, o2 in zip(other.motifs, other.offsets):
                # `or` is too permissive; `and` is better
                if m1.center in m2.friends and m2.center in m1.friends:
                    kmers_linked = True
                    offset = (o1 - o2)
                    break
        # Second try shifted left and shifted right
        # Most conservative version:  seeds overlap by N-1 bases
        # Overlapping seeds with variants turns out to be too permissive.
        for m1, o1 in zip(self.motifs, self.offsets):
            if kmers_linked: break
            for m2, o2 in zip(other.motifs, other.offsets):
                if m1.center[1:] == m2.center[:-1]:
                    kmers_linked = True
                    offset = (o1 - o2) + 1
                    break
                elif m2.center[1:] == m1.center[:-1]:
                    kmers_linked = True
                    offset = (o1 - o2) - 1
                    break
        if not kmers_linked: return None
        new = copy.copy(self)
        new.motifs = self.motifs + other.motifs
        new.offsets = self.offsets + [o+offset for o in other.offsets]
        offset = min(new.offsets)
        new.offsets = [o-offset for o in new.offsets] # set min offset to 0
        new.all_kmers = self.all_kmers | other.all_kmers
        new.start, new.end, new.score, new.evidence = improve_window(kmers=new.all_kmers, **new.data)
        if new.score > max(self.score, other.score) and overlaps(self, new) and overlaps(other, new):
            return new
        else: return None
    @property
    def revcomp(self):
        if not hasattr(self, "_revcomp"):
            # Motifs that are single stranded will return None.
            # If all our motifs are single stranded, we'll return None also.
            rc = self._revcomp = copy.copy(self)
            rc.motifs = []
            rc.lost_motifs = [] # single stranded motifs that can't be used in the revcomp
            rc.offsets = []
            rc.all_kmers = set()
            for motif, offset in zip(self.motifs, self.offsets):
                if motif.revcomp:
                    rc.motifs.append(motif.revcomp)
                    rc.offsets.append(offset) # will be reversed shortly
                    rc.all_kmers.update(motif.revcomp.all_kmers)
                else:
                    rc.lost_motifs.append(motif)
            if rc.motifs:
                max_off = max(rc.offsets)
                rc.offsets = [max_off - o for o in rc.offsets] # have to reverse direction of offsets
                rc.start, rc.end, rc.score, rc.evidence = improve_window(kmers=rc.all_kmers, **rc.data)
                rc._revcomp = self
            else:
                self._revcomp = None
        return self._revcomp
    def __cmp__(self, other):
        return cmp(self.score, other.score)
    def __len__(self):
        return len(self.motifs)
    def __str__(self):
        return "<%.2f, %i to %i, %i motifs: %s ...>" % (self.score, self.start_user, self.end_user, len(self.motifs), self.motifs[0].center)

def overlaps(mc1, mc2):
    """Compare two motifs and/or clusters to see if their location ranges overlap."""
    return (mc1.start < mc2.end) and (mc1.end > mc2.start)

def extract_positive_seqs(cluster, evidence, suf, seq_lens, seq_offsets, align_5p, midgap):
    """
    For the "positive" (in-group) sequences, extract all the unique regions
    that contain 1+ of the kmers in cluster.
    We do this to avoid the multiple-counting problem of tallying kmers individually.
    """
    regions = set() # set([(start,end,revcomp)]) within suf.corpus (the concatenated sequences)
    max_offset = max(cluster.offsets)
    kmer_len = len(cluster.motifs[0].center) + midgap # k-mer length, including "gap" residues in the middle
    region_len = max_offset + kmer_len
    for motif, offset in zip(cluster.motifs, cluster.offsets):
        # First, the positive strand:
        fwd_kmers = set([motif.center]) | motif.friends
        locs, matches = kmers_matches(fwd_kmers, suf, seq_offsets, midgap)
        seq_locs = locs.copy()
        max_seq_len = filelocs_to_seqlocs(seq_locs, matches, seq_lens, seq_offsets, align_5p)
        # matches within the positive set of sequences and the allowed region
        keep = (evidence[matches] > 0) & (cluster.start <= seq_locs) & (seq_locs < cluster.end)
        starts = locs[keep] - offset
        regions.update((start, start+region_len, False) for start in starts)
        # Then, the reverse strand
        rev_kmers = motif.all_kmers - fwd_kmers # avoid double-counting palindromes
        if rev_kmers:
            locs, matches = kmers_matches(rev_kmers, suf, seq_offsets, midgap)
            seq_locs = locs.copy()
            max_seq_len = filelocs_to_seqlocs(seq_locs, matches, seq_lens, seq_offsets, align_5p)
            # matches within the positive set of sequences and the allowed region
            keep = (evidence[matches] > 0) & (cluster.start <= seq_locs) & (seq_locs < cluster.end)
            starts = locs[keep] - (max_offset - offset)
            regions.update((start, start+region_len, True) for start in starts)
    seqs = []
    for start, end, revcomp in regions:
        subseq = suf.corpus[start:end]
        # Because of the padding at the start and/or end due to max_offset,
        # subseq may inadvertently span two different genomic sequences!
        nl = subseq.find("\n")
        if nl >= 0:
            if nl < kmer_len: subseq = subseq[nl+1:].rjust(region_len, "N") # left side shorter than k
            elif nl >= max_offset: subseq = subseq[:nl].ljust(region_len, "N") # right side shorter than k
            else: continue # could have been either side, too much trouble to figure out, skip it
        if revcomp: seqs.append(dnaseq.reverse_complement(subseq))
        else: seqs.append(subseq)
    return seqs

