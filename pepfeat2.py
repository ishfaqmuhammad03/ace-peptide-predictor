"""Additional descriptor families: CTD, extended positional, quasi-sequence-order."""
import numpy as np, pandas as pd
import pepfeat as P
AA=P.AA

# three coarse property alphabets for CTD (standard Dubchak groupings)
CTD_GROUPS={
 'hydrophob':['RKEDQN','GASTPHY','CLVIMFW'],
 'vdw':      ['GASTPD','NVEQIL','MHKFRYW'],
 'polarity': ['LIFWCMVY','PATGS','HQRKNED'],
 'polariz':  ['GASDT','CPNVEQIL','KMHFRYW'],
 'charge':   ['KR','ANCQGHILMFPSTWYV','DE'],
 'ss':       ['EALMQKRH','VIYCWFT','GNPSD'],
 'sa':       ['ALFCGIVW','RKQEND','MPSTHY'],
}
def _gidx(a,groups):
    for i,g in enumerate(groups):
        if a in g: return i
    return 0

def ctd(seq):
    f={}
    for name,groups in CTD_GROUPS.items():
        idx=[_gidx(a,groups) for a in seq]; L=len(idx)
        # Composition
        for i in range(3): f[f'CTDC_{name}_{i}']=idx.count(i)/L
        # Transition
        tot=max(L-1,1)
        for i in range(3):
            for j in range(i+1,3):
                n=sum(1 for k in range(L-1) if {idx[k],idx[k+1]}=={i,j})
                f[f'CTDT_{name}_{i}{j}']=n/tot
        # Distribution: relative position of 1st/25%/50%/75%/100% occurrence
        for i in range(3):
            pos=[k+1 for k,v in enumerate(idx) if v==i]
            if not pos:
                for q in (0,25,50,75,100): f[f'CTDD_{name}_{i}_{q}']=0.0
            else:
                n=len(pos)
                for q in (0,25,50,75,100):
                    r=max(int(np.ceil(n*q/100.0)),1)
                    f[f'CTDD_{name}_{i}_{q}']=pos[r-1]/L
    return f

PAD=8   # positions encoded from each terminus in the extended set
ZS=['z1','z2','z3','z4','z5','kd','vol','pol']
def extended_positional(seq):
    f={};L=len(seq)
    for p in range(PAD):
        aN=seq[p] if L>p else None
        aC=seq[-(p+1)] if L>=p+1 else None
        for sc in ZS:
            f[f'XN{p+1}_{sc}']=P.SCALES[sc].get(aN,0.0) if aN else 0.0
            f[f'XC{p+1}_{sc}']=P.SCALES[sc].get(aC,0.0) if aC else 0.0
        f[f'XN{p+1}_ok']=1.0 if aN else 0.0
        f[f'XC{p+1}_ok']=1.0 if aC else 0.0
    # terminal tripeptide on a 3-group alphabet (27 dense indicators)
    G3={}
    for g,mem in {'h':'AVLIMFWCY','p':'STNQGP','c':'KRHDE'}.items():
        for a in mem: G3[a]=g
    t=''.join(G3.get(a,'p') for a in seq[-3:]) if L>=3 else 'none'
    for a in 'hpc':
        for b in 'hpc':
            for c in 'hpc':
                f[f'CT3_{a}{b}{c}']=1.0 if t==a+b+c else 0.0
    return f

def quasi_order(seq,maxlag=4):
    f={};L=len(seq)
    for sc in ['kd','vol','pol']:
        v=np.array([P.SCALES[sc].get(a,0.0) for a in seq])
        for lag in range(1,maxlag+1):
            if L>lag:
                f[f'QSO_{sc}_l{lag}']=float(np.mean((v[:-lag]-v[lag:])**2))
            else:
                f[f'QSO_{sc}_l{lag}']=0.0
    return f

def featurize2(seq):
    s=''.join(c for c in seq.upper() if c in P.AAI)
    f={}
    f.update(ctd(s)); f.update(extended_positional(s)); f.update(quasi_order(s))
    return f

def build2(seqs):
    return pd.DataFrame([featurize2(s) for s in seqs],index=list(seqs))
