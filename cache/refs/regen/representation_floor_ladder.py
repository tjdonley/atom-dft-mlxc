import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from scipy.spatial import cKDTree
from atom.descriptors.simple.invariants import clebsch_gordan, triangle_triples
CACHE=sys.argv[1] if len(sys.argv)>1 else "cache/refs/holes/repr_cache.npz"
SPLIT=sys.argv[2] if len(sys.argv)>2 else "all"   # all | closed
z=np.load(CACHE); d=z['d']; e_sf=z['e_sf']; Z=z['Z']; NOUT=int(z['NOUT']); LMAX=int(z['LMAX'])
if SPLIT=="closed" and 'closed' in z.files:
    m=z['closed'].astype(bool); d,e_sf,Z=d[m],e_sf[m],Z[m]
N=len(Z); block=lambda l: d[:, l*NOUT:(l+1)*NOUT]
def powspec(lmax): return np.column_stack([block(0)[:,1:]]+[np.sum(block(l)**2,1) for l in range(1,lmax+1)])
def bispec(lmax):
    cols=[]
    for (l1,l2,l3) in triangle_triples(lmax):
        cg=clebsch_gordan(l1,0,l2,0,l3,0)
        if abs(cg)<1e-12: continue
        cols.append(cg*block(l1)*block(l2)*block(l3))   # per-n diagonal axial bispectrum
    return np.column_stack(cols)
def vfull(lmax): return np.column_stack([block(0)[:,1:]]+[block(l) for l in range(1,lmax+1)])
SUBSETS={
  'cn+s2(cur)':np.column_stack([block(0)[:,1:],np.sum(block(1)**2,1)]),
  'cn+s2+l2pow':np.column_stack([block(0)[:,1:],np.sum(block(1)**2,1),np.sum(block(2)**2,1)]),
  'powspec l<=2':powspec(2),'powspec l<=4':powspec(4),'powspec l<=6':powspec(6),
  'pow+bisp l<=2':np.column_stack([powspec(2),bispec(2)]),
  'pow+bisp l<=4':np.column_stack([powspec(4),bispec(4)]),
  'Vfull signed l<=4':vfull(4),'Vfull signed l<=6':vfull(6),
}
def std01(X): mu=X.mean(0); sd=X.std(0); k=sd>1e-9; return (X[:,k]-mu[k])/sd[k]
def floor(R,K=40):
    Rz=std01(R); dim=Rz.shape[1]; tree=cKDTree(Rz); dist,idx=tree.query(Rz,k=K+1)
    ncx_d=np.full(N,np.inf); ncx_de=np.zeros(N); rd=[]; de=[]; cr=[]
    for i in range(N):
        got=False
        for m in range(1,K+1):
            j=idx[i,m]; rd.append(dist[i,m]); de.append(abs(e_sf[j]-e_sf[i])); cr.append(Z[j]!=Z[i])
            if (not got) and Z[j]!=Z[i]: ncx_d[i]=dist[i,m]; ncx_de[i]=abs(e_sf[j]-e_sf[i]); got=True
    ok=np.isfinite(ncx_d); thr=np.quantile(ncx_d[ok],0.25); close=ok&(ncx_d<=thr)
    rd=np.array(rd); de=np.array(de); cr=np.array(cr); band=rd<=thr
    di=de[band&~cr].mean() if (band&~cr).sum() else np.nan; dc=de[band&cr].mean() if (band&cr).sum() else np.nan
    return dim, ncx_de[close].mean(), ncx_d[close].mean(), (dc/di if di>0 else np.nan)
print("CACHE=%s SPLIT=%s  N=%d atoms=%d"%(os.path.basename(CACHE),SPLIT,N,len(set(Z))))
print("%-20s %5s %16s %12s %12s"%("invariant subset","dim","close-cross|de|","close sqrt-d","cross/in"))
res=[]
for name,R in SUBSETS.items():
    dim,cde,cd,ratio=floor(R); res.append((name,cde)); print("%-20s %5d %16.5f %12.3f %12.2f"%(name,dim,cde,cd,ratio),flush=True)
best=min(res,key=lambda x:x[1]); print("\nLOWEST close-cross |de| floor: %s (%.5f)"%(best[0],best[1]))
