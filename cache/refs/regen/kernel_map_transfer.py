"""Sparsity vs cross-atom-transfer diagnosis for the kernel feature->hole map (baseline feats).
kernel-only (L0) energy error vs Ehf: own-N refs (sparsity) and leave-one-atom-out (transfer).
Finding (2026-06-26): per-atom ~40 refs -> few mHa; LEAVE-ATOM-OUT is catastrophic (160-1885 mHa) ->
the features (l=0 monopole vec + l=1 NORM s^2) are NOT universal across atoms. The fix the richer
invariants (higher-l power spectrum, CG bispectrum) target. See reports/hole_expansion/kernel_map_control.txt."""

import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
MU,KLO,RIDGE=0.2195,0.804,1e-9
f=load_hole_refs_full(); Zf=np.asarray(f['atom_Z']); off=np.asarray(f['atom_offset']); npt=np.asarray(f['atom_npts']); cloA=np.asarray(f['closed'])
def inv_ell(nl0): return np.concatenate([np.full(nl0,1/0.7),[1/0.5]])
def kmat(A,B,ie):
    Aw=A*ie[None,:]; Bw=B*ie[None,:]; d2=np.sum(Aw*Aw,1)[:,None]+np.sum(Bw*Bw,1)[None,:]-2*Aw@Bw.T
    return np.exp(-0.5*np.maximum(d2,0))
def atom_data(Z):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(sat_gradient=True,fp_mu=MU,fp_kappa_lo=KLO))
    g=F._grad_op@rho; R_ad,_=F._R_ad(rho); kF=(3*np.pi**2*rho)**(1/3)
    cp=np.array([op@rho for op in F._ops]); c_ad=F._c_ad(cp,R_ad); cn=c_ad/np.where(np.abs(c_ad[:,:1])>1e-30,c_ad[:,:1],1e-30)
    s2b,_=_bound((g/(2*kF*rho))**2); X=np.column_stack([cn[:,1:],s2b])
    ai=int(np.where(Zf==Z)[0][0]); s,e=off[ai],off[ai]+npt[ai]; r0=np.sort(np.asarray(f['r0'])[s:e]); rt=np.asarray(f['rt'])[s:e][np.argsort(np.asarray(f['r0'])[s:e])]
    sel=np.array([int(np.argmin(np.abs(r-x))) for x in r0])
    sig_node=rt/(-0.5*rho[sel])[:,None]
    return dict(F=F,r=r,rho=rho,ew=F.energy_weights,Ehf=Ehf,X=X,R_ad=R_ad,Xn=X[sel],sign=sig_node)
def L0_energy(test,Xn,sign):
    F=test['F']; ie=inv_ell(Xn.shape[1]-1)
    coef=np.linalg.solve(kmat(Xn,Xn,ie)+RIDGE*np.eye(len(Xn)),sign)
    sig=kmat(test['X'],Xn,ie)@coef                      # kernel-only (base=0)
    eps=2*np.pi*test['R_ad']**2*((-0.5*test['rho'])[:,None]*sig@F._Cmom)
    return 1e3*(float(np.sum(test['ew']*test['rho']*eps))-test['Ehf'])
CLOSED=[2,4,10,12,18,20,30,36,48,54]; D={Z:atom_data(Z) for Z in [2,4,10,12,18]}
print("kernel-only (L0) energy error vs Ehf (mHa), baseline feats [cn[1:],s2]. PBE: He23 Be44 Ne78 Mg105 Ar62")
print("%4s %12s %12s %12s %16s"%("atom","own-150","own-40","own-15","LeaveAtomOut"))
allc={Z:atom_data(Z) for Z in CLOSED}
for Z,sym in [(2,'He'),(4,'Be'),(10,'Ne'),(12,'Mg'),(18,'Ar')]:
    t=D[Z]; n=len(t['Xn'])
    sub=lambda k: np.linspace(0,n-1,min(k,n)).astype(int)
    e150=L0_energy(t,t['Xn'],t['sign']); e40=L0_energy(t,t['Xn'][sub(40)],t['sign'][sub(40)]); e15=L0_energy(t,t['Xn'][sub(15)],t['sign'][sub(15)])
    # leave-one-atom-out: nodes = all OTHER closed atoms' ref points
    Xo=np.vstack([allc[z]['Xn'] for z in CLOSED if z!=Z]); so=np.vstack([allc[z]['sign'] for z in CLOSED if z!=Z])
    elo=L0_energy(t,Xo,so)
    print("%4s %12.0f %12.0f %12.0f %16.0f"%(sym,e150,e40,e15,elo),flush=True)
