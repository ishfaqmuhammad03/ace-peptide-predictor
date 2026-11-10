"""
Build the full 879-descriptor vector the ExtraTrees model expects, from sequence alone.

Three blocks, assembled in the exact order stored in model/feature_columns.json:
  base__   3    molecular weight, isoelectric point, length
  own__  103    BLOSUM62 profiles, AAindex / physicochemical scales, ACC autocorrelations
                (computed by peptide_features_103.py, the validated deployment featuriser)
  new__  452    Sandberg z-scales + 9 physicochemical scales, terminal positional encoding,
                reduced-alphabet composition, auto-cross-covariance   (pepfeat.py)
  ext__  321    CTD composition/transition/distribution, extended positional encoding,
                terminal tripeptide indicators, quasi-sequence-order  (pepfeat2.py)
"""
import numpy as np
import pandas as pd

import pepfeat
import pepfeat2
from peptide_features_103 import AA, compute_feature_table

# ---- residue molecular weights (monoisotopic-free average, water added once) ----
_MW = {'A': 71.1, 'C': 103.1, 'D': 115.1, 'E': 129.1, 'F': 147.2, 'G': 57.1, 'H': 137.1,
       'I': 113.2, 'K': 128.2, 'L': 113.2, 'M': 131.2, 'N': 114.1, 'P': 97.1, 'Q': 128.1,
       'R': 156.2, 'S': 87.1, 'T': 101.1, 'V': 99.1, 'W': 186.2, 'Y': 163.2}

# ---- isoelectric point: bisection on Henderson-Hasselbalch net charge ----
_PKA_N, _PKA_C = 9.69, 2.34
_PKA = {'D': 3.65, 'E': 4.25, 'C': 8.18, 'Y': 10.07, 'H': 6.00, 'K': 10.53, 'R': 12.48}
_NEG, _POS = set('DECY'), set('HKR')


def net_charge(seq, pH):
    q = 1 / (1 + 10 ** (pH - _PKA_N)) - 1 / (1 + 10 ** (_PKA_C - pH))
    for a in seq:
        if a in _POS:
            q += 1 / (1 + 10 ** (pH - _PKA[a]))
        elif a in _NEG:
            q -= 1 / (1 + 10 ** (_PKA[a] - pH))
    return q


def isoelectric_point(seq, lo=0.0, hi=14.0, iters=80):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def molecular_weight(seq):
    return sum(_MW.get(a, 0.0) for a in seq) + 18.0


def compute_feature_matrix(seqs, feature_columns):
    """seqs: list of clean uppercase sequences. Returns a DataFrame with columns in model order."""
    seqs = [s.strip().upper() for s in seqs]

    own_cols = [c[len('own__'):] for c in feature_columns if c.startswith('own__')]
    own = compute_feature_table(seqs, own_cols)
    own.columns = ['own__' + c for c in own.columns]

    new = pepfeat.build(seqs)
    new.columns = ['new__' + c for c in new.columns]
    new.index = own.index

    ext = pepfeat2.build2(seqs)
    ext.columns = ['ext__' + c for c in ext.columns]
    ext.index = own.index

    # base__pi: reuse the Biopython isoelectric point already computed in the own__ block.
    # It reproduces the training 'pi' column to within 0.05 pH units on average, whereas a
    # plain Henderson-Hasselbalch bisection is off by ~0.7, so this keeps training and
    # inference consistent. isoelectric_point() above is kept as a dependency-free fallback.
    if 'own__Phys_isoelectric_point' in own.columns:
        pi_vals = own['own__Phys_isoelectric_point'].astype(float).tolist()
    else:
        pi_vals = [isoelectric_point(s) for s in seqs]
    base = pd.DataFrame({'base__molwt': [molecular_weight(s) for s in seqs],
                         'base__pi': pi_vals,
                         'base__length': [float(len(s)) for s in seqs]}, index=own.index)

    X = pd.concat([base, own, new, ext], axis=1)
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise KeyError(f"{len(missing)} feature(s) could not be computed, e.g. {missing[:5]}")
    X = X[list(feature_columns)].astype(float)
    return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
