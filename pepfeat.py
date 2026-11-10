"""Dense, length-robust descriptor set for short bioactive peptides."""
import numpy as np, pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")
AAI = {a: i for i, a in enumerate(AA)}

# Sandberg z-scales (z1 lipophilicity, z2 steric bulk, z3 polarity, z4/z5 electronic)
Z = {
 'A':(0.24,-2.32,0.60,-0.14,1.30),  'C':(0.84,-1.67,3.71,0.18,-2.65),
 'D':(3.98,0.93,1.93,-2.46,0.75),   'E':(3.11,0.26,-0.11,-3.04,-0.25),
 'F':(-4.22,1.94,1.06,0.54,-0.62),  'G':(2.05,-4.06,0.36,-0.82,-0.38),
 'H':(2.47,1.95,0.26,3.90,0.09),    'I':(-3.89,-1.73,-1.71,-0.84,0.26),
 'K':(2.29,0.89,-2.49,1.49,0.31),   'L':(-4.28,-1.30,-1.49,-0.72,0.84),
 'M':(-2.85,-0.22,0.47,1.94,-0.98), 'N':(3.05,1.62,1.04,-1.15,1.61),
 'P':(-1.66,0.27,1.84,0.70,2.00),   'Q':(1.75,0.50,-1.44,-1.34,0.66),
 'R':(3.52,2.50,-3.50,1.99,-0.17),  'S':(2.39,-1.07,1.15,-1.39,0.67),
 'T':(0.75,-2.18,-1.12,-1.46,-0.40),'V':(-2.59,-2.64,-1.54,-0.85,-0.02),
 'W':(-4.36,3.94,0.59,3.44,-1.59),  'Y':(-2.54,2.44,0.43,0.04,-1.47)}

# physicochemical scales
KD = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,
      'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,
      'W':-0.9,'Y':-1.3}
VOL = {'A':88.6,'C':108.5,'D':111.1,'E':138.4,'F':189.9,'G':60.1,'H':153.2,'I':166.7,
       'K':168.6,'L':166.7,'M':162.9,'N':114.1,'P':112.7,'Q':143.8,'R':173.4,'S':89.0,
       'T':116.1,'V':140.0,'W':227.8,'Y':193.6}
POL = {'A':8.1,'C':5.5,'D':13.0,'E':12.3,'F':5.2,'G':9.0,'H':10.4,'I':5.2,'K':11.3,
       'L':4.9,'M':5.7,'N':11.6,'P':8.0,'Q':10.5,'R':10.5,'S':9.2,'T':8.6,'V':5.9,
       'W':5.4,'Y':6.2}
CHG = {a:0.0 for a in AA}; CHG.update({'D':-1,'E':-1,'K':1,'R':1,'H':0.1})
ARO = {a:0.0 for a in AA}; ARO.update({'F':1,'W':1,'Y':1,'H':0.5})
FLEX = {'A':0.36,'C':0.35,'D':0.51,'E':0.50,'F':0.31,'G':0.54,'H':0.32,'I':0.46,
        'K':0.47,'L':0.37,'M':0.30,'N':0.46,'P':0.51,'Q':0.49,'R':0.53,'S':0.51,
        'T':0.44,'V':0.39,'W':0.31,'Y':0.42}
HELIX = {'A':1.42,'C':0.70,'D':1.01,'E':1.51,'F':1.13,'G':0.57,'H':1.00,'I':1.08,
         'K':1.16,'L':1.21,'M':1.45,'N':0.67,'P':0.57,'Q':1.11,'R':0.98,'S':0.77,
         'T':0.83,'V':1.06,'W':1.08,'Y':0.69}
SHEET = {'A':0.83,'C':1.19,'D':0.54,'E':0.37,'F':1.38,'G':0.75,'H':0.87,'I':1.60,
         'K':0.74,'L':1.30,'M':1.05,'N':0.89,'P':0.55,'Q':1.10,'R':0.93,'S':0.75,
         'T':1.19,'V':1.70,'W':1.37,'Y':1.47}
HBD = {a:0.0 for a in AA}; HBD.update({'K':1,'R':2,'W':1,'N':1,'Q':1,'S':1,'T':1,'Y':1,'H':1,'C':1})
MW = {'A':71.1,'C':103.1,'D':115.1,'E':129.1,'F':147.2,'G':57.1,'H':137.1,'I':113.2,
      'K':128.2,'L':113.2,'M':131.2,'N':114.1,'P':97.1,'Q':128.1,'R':156.2,'S':87.1,
      'T':101.1,'V':99.1,'W':186.2,'Y':163.2}

SCALES = {'z1':{a:Z[a][0] for a in AA}, 'z2':{a:Z[a][1] for a in AA},
          'z3':{a:Z[a][2] for a in AA}, 'z4':{a:Z[a][3] for a in AA},
          'z5':{a:Z[a][4] for a in AA}, 'kd':KD, 'vol':VOL, 'pol':POL,
          'chg':CHG, 'aro':ARO, 'flex':FLEX, 'helix':HELIX, 'sheet':SHEET, 'hbd':HBD}
SNAMES = list(SCALES)

# reduced alphabet (7 physicochemical groups) for dense n-gram features
GROUP = {}
for g, mem in {'aliph':'AVLIM','arom':'FWY','pos':'KRH','neg':'DE','polar':'STNQ',
               'pro':'P','tiny':'GC'}.items():
    for a in mem: GROUP[a] = g
GNAMES = ['aliph','arom','pos','neg','polar','pro','tiny']

NTERM_POS, CTERM_POS = 4, 4   # positions encoded from each end


def _vals(seq, sc):
    return np.array([SCALES[sc].get(a, 0.0) for a in seq], float)


def featurize(seq):
    s = ''.join(c for c in seq.upper() if c in AAI)
    L = len(s)
    f = {}
    f['length'] = L
    f['log_length'] = np.log(L)
    f['mw'] = sum(MW.get(a, 0) for a in s) + 18.0
    # ---- global aggregates of every scale (dense for any L>=1) ----
    for sc in SNAMES:
        v = _vals(s, sc)
        f[f'{sc}_mean'] = v.mean(); f[f'{sc}_sum'] = v.sum()
        f[f'{sc}_max'] = v.max();   f[f'{sc}_min'] = v.min()
        f[f'{sc}_std'] = v.std()
        f[f'{sc}_range'] = v.max() - v.min()
        # weighted toward C-terminus (ACE binding is C-terminal driven)
        w = np.exp(np.arange(L) - (L - 1))      # 1 at C-term, decaying to N-term
        f[f'{sc}_cwt'] = float((v * w).sum() / w.sum())
        wn = np.exp(-np.arange(L))
        f[f'{sc}_nwt'] = float((v * wn).sum() / wn.sum())
    # ---- positional encoding anchored at BOTH termini ----
    for p in range(CTERM_POS):
        a = s[-(p + 1)] if L >= p + 1 else None
        for sc in SNAMES:
            f[f'C{p+1}_{sc}'] = SCALES[sc].get(a, 0.0) if a else 0.0
        f[f'C{p+1}_present'] = 1.0 if a else 0.0
    for p in range(NTERM_POS):
        a = s[p] if L >= p + 1 else None
        for sc in SNAMES:
            f[f'N{p+1}_{sc}'] = SCALES[sc].get(a, 0.0) if a else 0.0
        f[f'N{p+1}_present'] = 1.0 if a else 0.0
    # ---- one-hot of the two terminal residues (the strongest local signal) ----
    for a in AA:
        f[f'Cterm_is_{a}'] = 1.0 if s[-1] == a else 0.0
        f[f'Nterm_is_{a}'] = 1.0 if s[0] == a else 0.0
        f[f'C2_is_{a}'] = 1.0 if L >= 2 and s[-2] == a else 0.0
    # ---- composition (dense) ----
    for a in AA:
        f[f'AAC_{a}'] = s.count(a) / L
    for g in GNAMES:
        f[f'GRP_{g}'] = sum(1 for a in s if GROUP.get(a) == g) / L
    # reduced-alphabet terminal dipeptide (7x7 = 49, far denser than 400 DPC)
    if L >= 2:
        gc = f"{GROUP.get(s[-2],'x')}_{GROUP.get(s[-1],'x')}"
    else:
        gc = 'none'
    for g1 in GNAMES:
        for g2 in GNAMES:
            f[f'CTDI_{g1}_{g2}'] = 1.0 if gc == f'{g1}_{g2}' else 0.0
    # reduced-alphabet dipeptide composition over whole sequence (49, dense)
    gseq = [GROUP.get(a, 'x') for a in s]
    tot = max(L - 1, 1)
    cnt = {}
    for i in range(L - 1):
        cnt[(gseq[i], gseq[i + 1])] = cnt.get((gseq[i], gseq[i + 1]), 0) + 1
    for g1 in GNAMES:
        for g2 in GNAMES:
            f[f'GDPC_{g1}_{g2}'] = cnt.get((g1, g2), 0) / tot
    # ---- auto-cross-covariance of z-scales, lags 1-3 (dense, 0 + mask if too short) ----
    for lag in (1, 2, 3):
        ok = L > lag
        for sc in ['z1', 'z2', 'z3', 'kd', 'vol']:
            v = _vals(s, sc)
            f[f'ACC_{sc}_l{lag}'] = float(np.mean(v[:-lag] * v[lag:])) if ok else 0.0
        f[f'ACC_mask_l{lag}'] = 1.0 if ok else 0.0
    # ---- net charge / pI-like, hydrophobic moment proxy ----
    f['net_charge'] = sum(CHG.get(a, 0) for a in s)
    f['charge_density'] = f['net_charge'] / L
    f['frac_hydrophobic'] = sum(1 for a in s if a in 'AVLIMFWC') / L
    f['frac_aromatic'] = sum(1 for a in s if a in 'FWY') / L
    f['frac_proline'] = s.count('P') / L
    f['has_PP'] = 1.0 if 'PP' in s else 0.0
    f['cterm_PP'] = 1.0 if s.endswith('P') else 0.0
    f['cterm_aromatic'] = 1.0 if s[-1] in 'FWY' else 0.0
    f['cterm_hydrophobic'] = 1.0 if s[-1] in 'AVLIMFWCP' else 0.0
    f['nterm_branched'] = 1.0 if s[0] in 'VLI' else 0.0
    # length-regime indicators (model can switch behaviour)
    f['is_di'] = 1.0 if L == 2 else 0.0
    f['is_tri'] = 1.0 if L == 3 else 0.0
    f['is_short'] = 1.0 if L <= 5 else 0.0
    f['is_long'] = 1.0 if L >= 8 else 0.0
    return f


def build(seqs):
    rows = [featurize(s) for s in seqs]
    return pd.DataFrame(rows, index=list(seqs))


# ---------- BLOSUM62 local-alignment sequence kernel ----------
B62_TXT = """A R N D C Q E G H I L K M F P S T W Y V
A 4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
R -1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
N -2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
D -2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
C 0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
E -1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
G 0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
H -2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
L -1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
K -1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
M -1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
F -2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
S 1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
T 0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
V 0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""


def _b62():
    lines = B62_TXT.strip().split('\n')
    order = lines[0].split()
    M = np.zeros((20, 20))
    for ln in lines[1:]:
        p = ln.split(); a = p[0]
        for j, v in enumerate(p[1:]):
            M[AAI[a], AAI[order[j]]] = float(v)
    return M

B62 = _b62()


def _sw(a, b, gap_open=11.0, gap_ext=1.0):
    """Smith-Waterman local alignment score with affine gaps."""
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1)); E = np.zeros((n + 1, m + 1)); F = np.zeros((n + 1, m + 1))
    best = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            E[i, j] = max(E[i, j - 1] - gap_ext, H[i, j - 1] - gap_open)
            F[i, j] = max(F[i - 1, j] - gap_ext, H[i - 1, j] - gap_open)
            H[i, j] = max(0.0, H[i - 1, j - 1] + B62[ai, b[j - 1]], E[i, j], F[i, j])
            if H[i, j] > best: best = H[i, j]
    return best


def sw_kernel(seqs_a, seqs_b=None):
    """Normalised Smith-Waterman BLOSUM62 kernel K(a,b)=sw(a,b)/sqrt(sw(a,a)sw(b,b))."""
    A = [np.array([AAI[c] for c in s.upper() if c in AAI], dtype=np.int64) for s in seqs_a]
    sym = seqs_b is None
    B = A if sym else [np.array([AAI[c] for c in s.upper() if c in AAI], dtype=np.int64) for s in seqs_b]
    sa = np.array([_sw(x, x) for x in A]); sb = sa if sym else np.array([_sw(x, x) for x in B])
    K = np.zeros((len(A), len(B)))
    for i, x in enumerate(A):
        jr = range(i, len(B)) if sym else range(len(B))
        for j in jr:
            v = _sw(x, B[j]) / np.sqrt(sa[i] * sb[j])
            K[i, j] = v
            if sym: K[j, i] = v
    return K
