"""Richer rotational-invariant features vs cross-atom transfer (leave-one-atom-out).
Computes consistent higher-l axial-multipole coefficients d_{nl} (window-projected, transferred to the
stored adaptive Rad, monopole-normalized -- VERIFIED: l=0 reproduces stored cn[1:] to 2e-3, corr 1.0).
Regularized KRR (standardized feats, median-heuristic length scale, ridge scan) predicts a held-out
atom from the other 9 closed-shell atoms. FINDING (2026-06-26): replacing the scalar reduced-gradient
s^2 with the full l=1 radial VECTOR improves leave-atom-out transfer ~6x (39750->6761 mHa, same dim
count -- the info, not dims). l<=2/3 HURT in this 10-atom KRR (overfitting -> inconclusive). Absolute
levels are huge because leave-one-WHOLE-atom-out from 10 atoms is extreme extrapolation. See
reports/hole_expansion/richer_invariant_transfer.txt."""

import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP
from atom.descriptors.simple.pipeline import window_basis, transfer_matrix
from atom.descriptors.simple.bessel import radial_gauss_grid
from cache.refs.loader import load_hf, load_hole_refs_full
NIN,NOUT,RC,LMAX=20,10,6.0,3
quad=radial_gauss_grid(RC,512); qn=np.asarray(quad.nodes); qw=np.asarray(quad.weights)
u,wu=leggauss(64); BAS={l:window_basis(l,NIN).evaluate(l,qn) for l in range(LMAX+1)}
ANG={l:np.sqrt(4*np.pi/(2*l+1)) for l in range(LMAX+1)}; PL={l:eval_legendre(l,u) for l in range(LMAX+1)}
Cmom=SIMPLE_HOLE_KERNEL_FP(r_quad=np.linspace(1e-3,14,400),quadrature_weights=np.gradient(np.linspace(1e-3,14,400)))._Cmom
f=load_hole_refs_full(); Zf=np.asarray(f['atom_Z']); off=np.asarray(f['atom_offset']); npt=np.asarray(f['atom_npts'])
CLOSED=[2,4,10,12,18,20,30,36,48,54]; NPP=50
def rho_fn(r,rho): rs=np.sort(r); rr=rho[np.argsort(r)]; return lambda x: np.interp(np.asarray(x),rs,rr,left=rr[0],right=0.0)
def coeffs(rf,r0,Rad):
    dist=np.sqrt(np.maximum(r0**2+qn[:,None]**2-2*qn[:,None]*r0*u[None,:],0.0)); rv=rf(dist)
    out={}
    for l in range(LMAX+1):
        prof=(2*l+1)/2.0*(rv*(wu*PL[l])[None,:]).sum(1)
        cwin=ANG[l]*BAS[l]@(qw*qn**2*prof)
        out[l]=transfer_matrix(l,Rad,NOUT,NIN)@cwin
    n0=out[0][0] if abs(out[0][0])>1e-30 else 1e-30
    return {l:out[l]/n0 for l in range(LMAX+1)}
def atom_pack(Z):
    hf=load_hf(Z); r=np.asarray(hf['r']); rho=np.maximum(np.asarray(hf['rho']),1e-12); w=np.asarray(hf['w']); Ehf=float(hf['Ehf'])
    o=np.argsort(r); r,rho,w=r[o],rho[o],w[o]; ew=4*np.pi*r**2*w; rf=rho_fn(r,rho)
    ai=int(np.where(Zf==Z)[0][0]); s,e=off[ai],off[ai]+npt[ai]
    r0=np.asarray(f['r0'])[s:e]; oo=np.argsort(r0); sub=np.linspace(0,len(oo)-1,NPP).astype(int); oo=oo[sub]
    r0=r0[oo]; rt=np.asarray(f['rt'])[s:e][oo]; rho0=np.asarray(f['rho'])[s:e][oo]; Rad0=np.asarray(f['Rad'])[s:e][oo]
    cn_st=np.asarray(f['cn'])[s:e][oo]; s_st=np.asarray(f['s'])[s:e][oo]
    D=[coeffs(rf,float(x),float(R)) for x,R in zip(r0,Rad0)]
    d={l:np.array([D[k][l] for k in range(len(r0))]) for l in range(LMAX+1)}   # (npts,NOUT) per l, monopole-normalized
    return dict(d=d,cn_st=cn_st,s_st=s_st,sig=rt/(-0.5*rho0)[:,None],rho0=rho0,Rad0=Rad0,r0=r0,r=r,rho=rho,ew=ew,Ehf=Ehf)
print("computing features (10 atoms x %d pts)..."%NPP,flush=True)
packs={Z:atom_pack(Z) for Z in CLOSED}
# consistency: my l=0 (d[0][:,1:]) vs stored cn[1:]
alld0=np.vstack([packs[z]['d'][0][:,1:] for z in CLOSED]); allcn=np.vstack([packs[z]['cn_st'][:,1:] for z in CLOSED])
print("CONSISTENCY l=0 vs stored cn[1:]: max|diff|=%.2e  corr=%.4f"%(np.max(np.abs(alld0-allcn)),np.corrcoef(alld0.ravel(),allcn.ravel())[0,1]),flush=True)
def featset(p,kind,lmax):
    cols=[p['d'][0][:,1:]]                       # l=0 signed vector (cn[1:]), always
    if kind=='baseline': cols.append((p['s_st']**2)[:,None])      # + scalar s^2 (current functional)
    else:
        for l in range(1,lmax+1):
            cols.append(p['d'][l] if kind=='signed' else p['d'][l]**2)   # signed vec or power spectrum
    return np.hstack(cols)
def loao(kind,lmax,lam):
    tot=[]
    for test in CLOSED:
        Xn=np.vstack([featset(packs[z],kind,lmax) for z in CLOSED if z!=test]); y=np.vstack([packs[z]['sig'] for z in CLOSED if z!=test])
        mu=Xn.mean(0); sd=Xn.std(0); keep=sd>1e-8; mu,sd=mu[keep],sd[keep]
        Zn=(Xn[:,keep]-mu)/sd
        from scipy.spatial.distance import pdist
        ell=np.median(pdist(Zn[::3])) if len(Zn)>50 else np.median(pdist(Zn)); ell=max(ell,1e-6)
        K=np.exp(-0.5*(np.sum(Zn*Zn,1)[:,None]+np.sum(Zn*Zn,1)[None,:]-2*Zn@Zn.T)/ell**2)
        coef=np.linalg.solve(K+lam*np.eye(len(Zn)),y)
        t=packs[test]; Zt=(featset(t,kind,lmax)[:,keep]-mu)/sd
        Kt=np.exp(-0.5*(np.sum(Zt*Zt,1)[:,None]+np.sum(Zn*Zn,1)[None,:]-2*Zt@Zn.T)/ell**2)
        sig=Kt@coef; eps=2*np.pi*t['Rad0']**2*((-0.5*t['rho0'])[:,None]*sig@Cmom)
        tot.append(abs(1e3*(float(np.sum(t['ew']*t['rho']*np.interp(t['r'],t['r0'],eps)))-t['Ehf'])))
    return np.mean(tot)
print("\nLEAVE-ONE-ATOM-OUT transfer MAE (mHa), regularized KRR. best over lambda in {1e-2,1e-1,1}. PBE~50, rSCAN~16")
print("%-26s %8s"%("feature set","LOAO MAE"))
for kind,lmax,name in [('baseline',0,'cn[1:] + s^2 (current)'),('signed',1,'+ l=1 vector'),('signed',2,'+ l<=2 vectors'),('signed',3,'+ l<=3 vectors'),('pow',3,'power spectrum l<=3')]:
    best=min(loao(kind,lmax,lam) for lam in (1e-2,1e-1,1.0))
    print("%-26s %8.0f"%(name,best),flush=True)
