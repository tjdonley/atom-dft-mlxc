import os, numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
z=np.load("cache/refs/holes/repr_cache.npz"); d=z['d']; e_sf=z['e_sf']; c=z['c']; ew=z['ew']; rho=z['rho']; Z=z['Z']; NOUT=int(z['NOUT']); LMAX=int(z['LMAX'])
N=len(Z); WE=np.abs(ew*rho*c); natom=len(np.unique(Z)); uZ=np.unique(Z)
def block(l): return d[:, l*NOUT:(l+1)*NOUT]
def invset(n):
    cn1=block(0)[:,1:]
    return {'cn+s2(cur)':np.column_stack([cn1,np.sum(block(1)**2,1)]),
      'powspec_l6':np.column_stack([cn1]+[np.sum(block(l)**2,1) for l in range(1,LMAX+1)]),
      'V_full':np.column_stack([cn1]+[block(l) for l in range(1,LMAX+1)])}[n]
def std01(X): mu=X.mean(0); sd=X.std(0); k=sd>1e-9; return (X[:,k]-mu[k])/sd[k]
def extrap(R,mode,P=8):
    Rz=std01(R); sd=np.zeros((N,P)); de=np.zeros((N,P))
    for i in range(N):
        if mode=='in': pool=np.where(Z==Z[i])[0]
        else: pool=np.where(Z!=Z[i])[0]
        D2=np.sum((Rz[pool]-Rz[i])**2,1)
        if mode=='in': D2[np.where(pool==i)[0]]=np.inf
        o=np.argsort(D2)[:P]; sd[i]=np.sqrt(D2[o]); de[i]=np.abs(e_sf[pool[o]]-e_sf[i])
    # per-rank: mean sqrt(delta) and dimensionalized energy spread
    rootd=sd.mean(0); emha=np.array([1e3*np.sum(WE*de[:,M]/np.sqrt(2))/natom for M in range(P)])
    a=np.polyfit(rootd,emha,1); return rootd,emha,a[1]   # intercept = floor (mHa)
print("Energy spread (mHa/atom) vs sqrt(feature dist); |de|~sqrt(delta) so intercept(sqrt d->0) = floor.")
print("Known: in-domain MODEL floor ~1 mHa.")
for name in ['cn+s2(cur)','powspec_l6','V_full']:
    rd,e,f=extrap(invset(name),'in'); print("IN  %-12s M1..8: "%name+" ".join("%.0f"%x for x in e)+"  -> floor %.1f mHa"%f,flush=True)
for name in ['cn+s2(cur)','powspec_l6','V_full']:
    rd,e,f=extrap(invset(name),'cross'); print("CRX %-12s M1..8: "%name+" ".join("%.0f"%x for x in e)+"  -> floor %.1f mHa (nearest-cross sqrt-d %.2f)"%(f,rd[0]),flush=True)
