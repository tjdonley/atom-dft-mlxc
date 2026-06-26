import os, numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from scipy.spatial import cKDTree
z=np.load("cache/refs/holes/repr_cache.npz"); d=z['d']; e_sf=z['e_sf']; Z=z['Z']; NOUT=int(z['NOUT']); LMAX=int(z['LMAX'])
N=len(Z)
def block(l): return d[:, l*NOUT:(l+1)*NOUT]
def invset(n):
    cn1=block(0)[:,1:]
    return {'cn+s2(cur)':np.column_stack([cn1,np.sum(block(1)**2,1)]),
      '+l1vec':np.column_stack([cn1,block(1)]),
      '+l<=2':np.column_stack([cn1,block(1),block(2)]),
      '+l<=3':np.column_stack([cn1,block(1),block(2),block(3)]),
      'powspec_l6':np.column_stack([cn1]+[np.sum(block(l)**2,1) for l in range(1,LMAX+1)]),
      'V_full':np.column_stack([cn1]+[block(l) for l in range(1,LMAX+1)])}[n]
def std01(X): mu=X.mean(0); sd=X.std(0); k=sd>1e-9; return (X[:,k]-mu[k])/sd[k]
print("CROSS-ATOM TRANSFER QUALITY: among each point's nearest CROSS-atom neighbor, the closest 25%%")
print("(best-matched), mean |Delta e_sf| & mean sqrt-d. Lower Delta e among close matches = better transfer.")
print("Also: matched-distance ratio = cross|Delta e| / in-domain|Delta e| at the same sqrt-d (>1 = info lost).")
print("%-12s %5s %14s %12s %16s"%("invariants","dim","close-cross|de|","close sqrt-d","cross/in ratio"))
for n in ['cn+s2(cur)','+l1vec','+l<=2','+l<=3','powspec_l6','V_full']:
    Rz=std01(invset(n)); dim=Rz.shape[1]; tree=cKDTree(Rz)
    # nearest cross-atom neighbor per point
    dist,idx=tree.query(Rz,k=40)
    ncx_d=np.full(N,np.inf); ncx_de=np.zeros(N)
    for i in range(N):
        for m in range(1,40):
            if Z[idx[i,m]]!=Z[i]: ncx_d[i]=dist[i,m]; ncx_de[i]=abs(e_sf[idx[i,m]]-e_sf[i]); break
    ok=np.isfinite(ncx_d); thr=np.quantile(ncx_d[ok],0.25); close=ok&(ncx_d<=thr)
    # matched in-domain |de| at the close-cross sqrt-d range: nearest same-atom with sqrt-d ~ thr
    # use all-neighbor pairs binned: compare cross vs in at the close sqrt-d band [0, thr]
    rd=[]; de=[]; cr=[]
    for i in range(N):
        for m in range(1,40):
            rd.append(dist[i,m]); de.append(abs(e_sf[idx[i,m]]-e_sf[i])); cr.append(Z[idx[i,m]]!=Z[i])
    rd=np.array(rd); de=np.array(de); cr=np.array(cr); band=rd<=thr
    de_in=de[band&~cr].mean() if (band&~cr).sum() else np.nan; de_cx=de[band&cr].mean() if (band&cr).sum() else np.nan
    ratio=de_cx/de_in if de_in>0 else np.nan
    print("%-12s %5d %14.4f %12.3f %16.2f"%(n,dim,ncx_de[close].mean(),ncx_d[close].mean(),ratio))
