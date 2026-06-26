import os, numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from scipy.spatial import cKDTree
z=np.load("cache/refs/holes/repr_cache.npz"); d=z['d']; e_sf=z['e_sf']; Z=z['Z']; NOUT=int(z['NOUT']); LMAX=int(z['LMAX'])
N=len(Z)
def block(l): return d[:, l*NOUT:(l+1)*NOUT]
def invset(n):
    cn1=block(0)[:,1:]
    return {'cn+s2(cur)':np.column_stack([cn1,np.sum(block(1)**2,1)]),
      'powspec_l6':np.column_stack([cn1]+[np.sum(block(l)**2,1) for l in range(1,LMAX+1)]),
      'V_full':np.column_stack([cn1]+[block(l) for l in range(1,LMAX+1)])}[n]
def std01(X): mu=X.mean(0); sd=X.std(0); k=sd>1e-9; return (X[:,k]-mu[k])/sd[k]
def binned(name,K=20):
    Rz=std01(invset(name)); tree=cKDTree(Rz); dist,idx=tree.query(Rz,k=K+1)
    rootd=[]; de=[]; cross=[]
    for i in range(N):
        for m in range(1,K+1):
            j=idx[i,m]; rootd.append(dist[i,m]); de.append(abs(e_sf[j]-e_sf[i])); cross.append(Z[j]!=Z[i])
    rootd=np.array(rootd); de=np.array(de); cross=np.array(cross)
    # shared sqrt-d bins
    edges=np.quantile(rootd,np.linspace(0,1,9))
    print("\n%s : mean |Delta e_sf| (scale-free) vs sqrt-d, in-domain vs cross-atom"%name)
    print("%10s %10s %12s %12s %8s %8s"%("sqrt-d_lo","sqrt-d_hi","in-domain","cross-atom","n_in","n_cx"))
    for b in range(len(edges)-1):
        m=(rootd>=edges[b])&(rootd<edges[b+1]); mi=m&~cross; mc=m&cross
        din=de[mi].mean() if mi.sum() else np.nan; dcx=de[mc].mean() if mc.sum() else np.nan
        print("%10.3f %10.3f %12.4f %12.4f %8d %8d"%(edges[b],edges[b+1],din,dcx,mi.sum(),mc.sum()))
for name in ['cn+s2(cur)','V_full']: binned(name)
