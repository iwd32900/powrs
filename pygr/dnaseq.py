r"""Basic functions for working with nucleic acid sequences and motifs.
"""

import string

def read_fasta(infile):
    '''Returns an iterator over (id, name, seq) for each entry, where name may be null.'''
    ident, name, seqs = None, None, []
    for line in infile:
        line = line.strip()
        if line == "": continue
        elif line.startswith(">"):
            seq = "".join(seqs)
            if len(seq) > 0: yield (ident, name, seq)
            fields = line[1:].split(None, 1)
            ident = fields[0]
            if len(fields) > 1: name = fields[1]
            seqs = []
        else:
            seqs.append(line)
    seq = "".join(seqs)
    if len(seq) > 0: yield (ident, name, seq)

def reverse_complement(seq):
    # By including punctuation, motif patterns can be reversed too!
    if not isinstance(seq, str): seq = str(seq) # in case of Unicode arguments!
    table = string.maketrans('ACBDGHKMNSRUTWVYacbdghkmnsrutwvy[]()/', 'TGVHCDMKNSYAAWBRTGVHCDMKNSYAAWBR][)(/')
    return seq.translate(table)[::-1]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
