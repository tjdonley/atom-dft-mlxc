import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, _bound
from cache.refs.loader import load_hole_refs_full
from scipy.spatial import cKDTree
SY={2:'He',4:'Be',6:'C',7:'N',8:'O',10:'Ne',12:'Mg',14:'Si',18:'Ar'}
p=load_hole_refs_full(); lk=np.maximum(np.abs(p['leakQ']),np.abs(p['leakE'])); m=(lk<=0.10)&(p['rho']>=0.01)
cn=p['cn'][m]; rt=p['rt'][m]; rho=p['rho'][m]; s=p['s'][m]; Q=p['Q'][m]; Z=p['Z'][m]; r0=p['r0'][m]
rr=np.linspace(1e-3,14,400); F=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=np.gradient(rr)); C=F._Cmom
e_sf=(rt/(-0.5*rho)[:,None])@C
D=np.column_stack([cn[:,1:], _bound(s**2)[0], _bound(Q)[0]]); Dz=(D-D.mean(0))/(D.std(0)+1e-12)
print("For each atom: how well are ITS points covered by OTHER atoms' refs, and the hole spread at the match.")
print("(coverage = median nearest-OTHER-atom feature distance; doppel = median |de_sf| at that NN)")
print("%4s %6s %14s %16s"%("atom","npts","NN-dist(cover)","|de_sf|@NN(doppel)"))
for Zt in (2,4,10,12,18,6,7,8,14):
    sel=Z==Zt; oth=Z!=Zt
    if sel.sum()<5: continue
    tree=cKDTree(Dz[oth]); dist,idx=tree.query(Dz[sel],k=1)
    de=np.abs(e_sf[sel]-e_sf[oth][idx])
    tag='OPEN ' if Zt in (6,7,8,14) else 'closed'
    print("%4s %6d %14.3f %16.4f   %s"%(SY[Zt],sel.sum(),np.median(dist),np.median(de),tag),flush=True)
# also: WITHIN-atom consistency (own NN) for N -- does density determine hole within N?
for Zt in (7,10):
    sel=np.where(Z==Zt)[0]
    tree=cKDTree(Dz[sel]); dist,idx=tree.query(Dz[sel],k=2)
    de=np.abs(e_sf[sel]-e_sf[sel][idx[:,1]])
    print("WITHIN %s: own-NN dist median %.4f, |de_sf| median %.4f"%(SY[Zt],np.median(dist[:,1]),np.median(de)),flush=True)
