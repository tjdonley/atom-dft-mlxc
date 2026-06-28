import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, _bound
from cache.refs.loader import load_hole_refs_full
SY={1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',36:'Kr'}
def sy(z): return SY.get(int(z),str(int(z)))
p=load_hole_refs_full(); lk=np.maximum(np.abs(p['leakQ']),np.abs(p['leakE']))
m=(lk<=0.10)&(p['rho']>=0.01)&(p['Z']<=36)
cn=p['cn'][m]; rt=p['rt'][m]; rho=p['rho'][m]; s=p['s'][m]; Q=p['Q'][m]; Z=p['Z'][m]; r0=p['r0'][m]
rr=np.linspace(1e-3,14,400); F=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=np.gradient(rr)); C=F._Cmom
sig=rt/(-0.5*rho)[:,None]; e_sf=sig@C
# DIMENSIONAL local exchange-energy density proxy: eps ~ -rho^(4/3) * (sigma.C)/(sigma_LDA.C) (relative)
s2=_bound(s**2)[0]; Qb=_bound(Q)[0]
D=np.column_stack([cn[:,1:], s2, Qb]); Dz=(D-D.mean(0))/(D.std(0)+1e-12)
from scipy.spatial import cKDTree
tree=cKDTree(Dz); dist,idx=tree.query(Dz,k=15)
cands=[]
for i in range(len(Dz)):
    for jj in range(1,15):
        j=idx[i,jj]
        if Z[i]>=Z[j]: continue                      # unordered, cross-atom
        if Z[i]==Z[j] or dist[i,jj]>0.06: continue   # very tight density match
        cands.append((dist[i,jj],abs(e_sf[i]-e_sf[j]),i,j))
cands.sort(key=lambda t:-t[1]/(t[0]+1e-6))
print("SAME LOCAL DENSITY, DIFFERENT EXACT HOLE (Z<=36, feature-dist<0.06 z-scored). e_sf=sigma.Cmom (energy-relevant).")
seen=set()
for d,de,i,j in cands:
    k=tuple(sorted((sy(Z[i]),sy(Z[j]))))
    if k in seen: continue
    seen.add(k)
    print("\n*** %s(r0=%.2f) vs %s(r0=%.2f) : density match dist=%.4f ***"%(sy(Z[i]),r0[i],sy(Z[j]),r0[j],d))
    print("   rho %.4f/%.4f  s %.3f/%.3f  Q %.2f/%.2f  ||cn shape diff||=%.4f"%(rho[i],rho[j],s[i],s[j],Q[i],Q[j],np.linalg.norm(cn[i,1:]-cn[j,1:])))
    print("   exact hole e_sf = %.4f vs %.4f  (%.0f%% different)   ||hole shape sigma diff||=%.4f"%(e_sf[i],e_sf[j],100*abs(e_sf[i]-e_sf[j])/abs(0.5*(e_sf[i]+e_sf[j])),np.linalg.norm(sig[i]-sig[j])))
    if len(seen)>=5: break
