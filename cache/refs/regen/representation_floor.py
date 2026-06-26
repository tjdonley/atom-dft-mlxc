import os, numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
SCR="/private/tmp/claude-501/-Users-ajm-Library-CloudStorage-Dropbox-GaTech-Andrew-Medford-amedford6-admin-admin-coding-SIMPLE-hole-functional/12298e99-707e-4af4-aa27-716af07a7159/scratchpad"
z=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","holes","repr_cache.npz")); d=z['d']; sig=z['sig']; e_sf=z['e_sf']; Cmom=z['Cmom']; c=z['c']; ew=z['ew']; rho=z['rho']; Z=z['Z']; NOUT=int(z['NOUT']); LMAX=int(z['LMAX'])
N=len(Z); WE=np.abs(ew*rho*c); natom=len(np.unique(Z))
Cn=Cmom/np.linalg.norm(Cmom); sig_perp=sig-(sig@Cn)[:,None]*Cn[None,:]
def block(l): return d[:, l*NOUT:(l+1)*NOUT]
def invset(n):
    cn1=block(0)[:,1:]
    return {'l0_only':cn1,'+l1norm(cur)':np.column_stack([cn1,np.sum(block(1)**2,1)]),
      '+l1vec':np.column_stack([cn1,block(1)]),'+l<=2':np.column_stack([cn1,block(1),block(2)]),
      '+l<=3':np.column_stack([cn1,block(1),block(2),block(3)]),
      'powspec':np.column_stack([cn1]+[np.sum(block(l)**2,1) for l in range(1,LMAX+1)]),
      'V_full':np.column_stack([cn1]+[block(l) for l in range(1,LMAX+1)])}[n]
SETS=['l0_only','+l1norm(cur)','+l1vec','+l<=2','+l<=3','powspec','V_full']
def std01(X): mu=X.mean(0); sd=X.std(0); k=sd>1e-9; return (X[:,k]-mu[k])/sd[k]
def knn_floor(R,K=5):
    Rz=std01(R); cross_e=np.zeros(N); cross_p=np.zeros(N); same_e=np.zeros(N)
    for i in range(N):
        dd=np.sum((Rz-Rz[i])**2,1)
        oth=np.where(Z!=Z[i])[0]; sam=np.where(Z==Z[i])[0]
        oc=oth[np.argsort(dd[oth])[:K]]; sc=sam[np.argsort(dd[sam])[1:K+1]]
        cross_e[i]=np.std(np.append(e_sf[oc],e_sf[i])); same_e[i]=np.std(np.append(e_sf[sc],e_sf[i]))
        cross_p[i]=np.mean(np.linalg.norm(sig_perp[oc]-sig_perp[i],axis=1))
    # dimensional energy floor (mHa/atom): sum WE*local_std over points / natom
    return 1e3*np.sum(WE*same_e)/natom, 1e3*np.sum(WE*cross_e)/natom, np.median(cross_p)
print("FIT-FREE k-NN conditional floors (K=5). same-atom = in-domain control; cross-atom = transfer.")
print("Energy floor in mHa/atom (irreducible Var(e_sf|invariants) dimensionalized). PBE~50, rSCAN~16.")
print("%-15s %14s %16s %16s"%("invariants","E same-atom","E CROSS-atom","shape-null cross"))
for n in SETS:
    se,ce,cp=knn_floor(invset(n)); print("%-15s %14.0f %16.0f %16.4f"%(n,se,ce,cp),flush=True)
