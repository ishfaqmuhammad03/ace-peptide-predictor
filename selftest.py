"""
Self-test for the deployed model. Run after installing requirements:
    python selftest.py

It checks that the featuriser, the model and the applicability-domain reference all load and
reproduce the reference predictions. Reference values are stored at full precision.

A difference below 1e-6 means your machine reproduces the published model bit-for-bit.
Anything up to 1e-4 is still a pass (harmless floating-point variation between platforms or
Biopython builds). A genuine version mismatch shows differences of 0.01 or more.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parent
TOL = 1e-4          # pass threshold
EXACT = 1e-6        # bit-for-bit threshold

REF = {  # sequence: (probability, inside applicability domain)
    "VVYPW":      (0.6855188127401898, True),
    "LFP":        (0.7050217054082291, True),
    "TSKY":       (0.6829869208153148, True),
    "FSAL":       (0.6602681480989256, True),
    "VVYPWTQ":    (0.6575196391090044, True),
    "IA":         (0.0735973512518676, True),
    "AL":         (0.0653508207979272, True),
    "VLDAFSDGLK": (0.3408174145613496, False),
}


def main():
    from features import compute_feature_matrix
    feats = json.loads((BASE / "model" / "feature_columns.json").read_text())
    model = joblib.load(BASE / "model" / "extratrees_model.joblib")
    ad = joblib.load(BASE / "model" / "ad_reference.joblib")

    seqs = list(REF)
    X = compute_feature_matrix(seqs, feats)
    if list(X.columns) != feats:
        print("FAILED: feature order does not match the model."); sys.exit(1)
    Xv = X.to_numpy(float)
    if not np.isfinite(Xv).all():
        print("FAILED: feature calculation produced non-finite values."); sys.exit(1)

    prob = model.predict_proba(Xv)[:, 1]
    dist = ad["nn"].kneighbors(ad["scaler"].transform(Xv))[0][:, :5].mean(1)

    worst, failed = 0.0, 0
    print(f"{'sequence':<12}{'expected':>12}{'got':>12}{'diff':>12}   domain")
    for s, p, d in zip(seqs, prob, dist):
        ep, ed = REF[s]
        inside = bool(d <= ad["threshold"])
        diff = abs(p - ep)
        worst = max(worst, diff)
        bad = diff > TOL or inside != ed
        failed += bad
        print(f"{s:<12}{ep:>12.6f}{p:>12.6f}{diff:>12.2e}   "
              f"{'inside' if inside else 'outside'}{'   <-- MISMATCH' if bad else ''}")

    print(f"\nfeatures: {Xv.shape[1]}   largest probability difference: {worst:.2e}")
    if failed:
        print(f"\nFAILED: {failed} sequence(s) differ by more than {TOL:g}.")
        print("Most likely cause is a scikit-learn version mismatch. Run:")
        print("    pip install scikit-learn==1.9.1")
        sys.exit(1)
    if worst < EXACT:
        print("PASS - bit-for-bit identical to the published model.")
    else:
        print(f"PASS - within {TOL:g} of the published model "
              f"(small floating-point variation, nothing to fix).")


if __name__ == "__main__":
    main()
