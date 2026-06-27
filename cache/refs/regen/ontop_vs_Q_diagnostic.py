import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from cache.refs.loader import load_hole_refs_full
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP
from atom.xc.simple_hole_expansion_explicit import enclosed_charge_switch
f=load_hole_refs_full(); rt=np.asarray(f['rt']); rho=np.asarray(f['rho']); Q=np.asarray(f['Q'])
lk=np.maximum(np.abs(f['leakQ']),np.abs(f['leakE'])); cl=np.asarray(f['closed'])
r=np.linspace(1e-3,14,400); F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r))
R0=F._R0  # on-top projection vector: hole on-top = coeffs @ R0
m=cl&(lk<=0.10)
ontop_dimless=(rt[m]@R0)/(-0.5*rho[m])   # exact reference dimensionless on-top (1=bulk -rho/2)
Qm=Q[m]; Qs=np.maximum(Qm,1e-12); W=enclosed_charge_switch(0.5*Qm)
wmodel=(1.0-W)*1.0 + W*(2.0/Qs)          # the W-blend's dimensionless on-top target
print("Exact reference on-top vs Q (dimensionless, 1 = bulk -rho/2; >1 toward FA/SIC -rho/Qs):")
print("%10s %6s %14s %14s %12s"%("Q-bin","n","exact on-top","W-model","|diff|"))
edges=[1.95,2.5,3,4,5,7,10,14,20]
for i in range(len(edges)-1):
    b=(Qm>=edges[i])&(Qm<edges[i+1])
    if b.sum()<3: continue
    print("%4.1f-%4.1f %6d %8.3f+-%.3f %8.3f+-%.3f %12.3f"%(edges[i],edges[i+1],b.sum(),
        ontop_dimless[b].mean(),ontop_dimless[b].std(),wmodel[b].mean(),wmodel[b].std(),
        np.abs(ontop_dimless[b]-wmodel[b]).mean()))
print("\nspread of exact on-top WITHIN bins (std) measures learnability; |diff| measures W-model bias")
print("overall: exact on-top mean %.3f, W-model mean %.3f, corr(exact,Q)=%.3f, corr(exact,wmodel)=%.3f"%(
    ontop_dimless.mean(),wmodel.mean(),np.corrcoef(ontop_dimless,Qm)[0,1],np.corrcoef(ontop_dimless,wmodel)[0,1]))
