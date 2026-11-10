"""Physicochemical, composition and autocorrelation descriptors computed
directly from a peptide sequence, used by the prediction server."""
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = "ACDEFGHIKLMNPQRSTVWY"
_B62 = substitution_matrices.load("BLOSUM62")

SCALES = {
    "AAindex_KYTJ820101": dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
                               L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2),
    "AAindex_GRAR740102": dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0, H=10.4, I=5.2,
                               L=4.9, K=11.3, M=5.7, F=5.2, P=8.0, S=9.2, T=8.6, W=5.4, Y=6.2, V=5.9),
    "Phys_hopp_woods": dict(A=-0.5, R=3.0, N=0.2, D=3.0, C=-1.0, Q=0.2, E=3.0, G=0.0, H=-0.5, I=-1.8,
                            L=-1.8, K=3.0, M=-1.3, F=-2.5, P=0.0, S=0.3, T=-0.4, W=-3.4, Y=-2.3, V=-1.5),
    "Phys_janin": dict(A=0.28, R=-1.14, N=-0.55, D=-0.52, C=0.97, Q=-0.69, E=-1.01, G=0.43, H=-0.31, I=0.60,
                       L=0.60, K=-1.62, M=0.43, F=0.46, P=-0.42, S=-0.19, T=-0.32, W=0.29, Y=-0.15, V=0.60),
    "Phys_flexibility": dict(A=0.984, C=0.906, D=1.068, E=1.094, F=0.915, G=1.031, H=0.950, I=0.927, K=1.102,
                             L=0.935, M=0.952, N=1.048, P=1.049, Q=1.037, R=1.008, S=1.046, T=0.997,
                             V=0.931, W=0.904, Y=0.929),
    "Phys_eisenberg": dict(A=-0.5652857436, C=-1.8022155672, D=0.8097336327, E=1.2857018785, F=-0.9178548145, G=-0.8620313783, H=0.5071118469, I=-1.1881577688, K=1.5795094375, L=-1.1881577689, M=-0.8620313784, N=0.8479286154, P=0.6716440800, Q=1.0007085462, R=1.3738441462, S=0.3161369337, T=0.5188641494, V=-1.1793435421, W=-0.5858522727, Y=0.2397469681),
}


def _z(sc):
    v = np.array([sc[a] for a in AA])
    mu, sd = v.mean(), v.std()           # population std over the 20 amino acids
    return {a: (sc[a] - mu) / sd for a in AA}


ZSCALES = {k: _z(v) for k, v in SCALES.items()}


def _acc(seq, a, b, lag):
    """(Cross-)covariance of z-scored scales a,b at a given lag; 0 if sequence too short."""
    L = len(seq)
    if L <= lag:
        return 0.0
    za = np.array([ZSCALES[a][r] for r in seq])
    zb = np.array([ZSCALES[b][r] for r in seq])
    return float(((za[:L - lag] - za.mean()) * (zb[lag:] - zb.mean())).sum() / (L - lag))


def compute_features(seq, feature_columns):
    seq = seq.strip().upper()
    f = {"SEQ_length": len(seq)}
    for a in AA:
        v = [_B62[r][a] for r in seq]
        f[f"BLOSUM62_mean_{a}"], f[f"BLOSUM62_N_{a}"], f[f"BLOSUM62_C_{a}"] = np.mean(v), v[0], v[-1]
    for name, sc in SCALES.items():
        v = np.array([sc[r] for r in seq])
        f.update({f"{name}_mean": v.mean(), f"{name}_std": v.std(), f"{name}_min": v.min(),
                  f"{name}_max": v.max(), f"{name}_N": v[0], f"{name}_C": v[-1]})
    pa = ProteinAnalysis(seq)
    f["Phys_isoelectric_point"] = pa.isoelectric_point()
    f["Phys_charge_pH7"] = pa.charge_at_pH(7)
    f["Phys_instability_index"] = pa.instability_index()
    out = {}
    for c in feature_columns:                       # ACC features parsed from the column name
        if c in f:
            out[c] = f[c]
        elif c.startswith("ACC_auto_"):
            n, lag = c[len("ACC_auto_"):].rsplit("_lag", 1)
            out[c] = _acc(seq, n, n, int(lag))
        elif c.startswith("ACC_cross_"):
            ab, lag = c[len("ACC_cross_"):].rsplit("_lag", 1)
            a, b = ab.split("_to_")
            out[c] = _acc(seq, a, b, int(lag))
        else:
            raise KeyError(f"Don't know how to compute feature: {c}")
    return out


def compute_feature_table(seqs, feature_columns):
    return pd.DataFrame([compute_features(s, feature_columns) for s in seqs])[list(feature_columns)]
