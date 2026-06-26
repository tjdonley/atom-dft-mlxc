import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP
from atom.descriptors.simple.pipeline import window_basis, transfer_matrix
from atom.descriptors.simple.bessel import radial_gauss_grid
from cache.refs.loader import load_hf, load_hole_refs_full
NIN,NOUT,RC,LMAX,NPP=20,10,6.0,6,150
quad=radial_gauss_grid(RC,512); qn=np.asarray(quad.nodes); qw=np.asarray(quad.weights)
u,wu=leggauss(64); BAS={l:window_basis(l,NIN).evaluate(l,qn) for l in range(LMAX+1)}
ANG={l:np.sqrt(4*np.pi/(2*l+1)) for l in range(LMAX+1)}; PLv={l:eval_legendre(l,u) for l in range(LMAX+1)}
Cmom=SIMPLE_HOLE_KERNEL_FP(r_quad=np.linspace(1e-3,14,400),quadrature_weights=np.gradient(np.linspace(1e-3,14,400)))._Cmom
f=load_hole_refs_full(); Zf=np.asarray(f['atom_Z']); off=np.asarray(f['atom_offset']); npt=np.asarray(f['atom_npts']); cloflat=np.asarray(f['closed'])
ATOMS=[int(zz) for zz in Zf]   # ALL atoms
def rho_fn(r,rho): rs=np.sort(r); rr=rho[np.argsort(r)]; return lambda x: np.interp(np.asarray(x),rs,rr,left=rr[0],right=0.0)
def coeffs(rf,r0,Rad):
    dist=np.sqrt(np.maximum(r0**2+qn[:,None]**2-2*qn[:,None]*r0*u[None,:],0.0)); rv=rf(dist); out=[]
    for l in range(LMAX+1):
        prof=(2*l+1)/2.0*(rv*(wu*PLv[l])[None,:]).sum(1)
        out.append(transfer_matrix(l,Rad,NOUT,NIN)@(ANG[l]*BAS[l]@(qw*qn**2*prof)))
    n0=out[0][0] if abs(out[0][0])>1e-30 else 1e-30
    return np.concatenate([o/n0 for o in out])
R={k:[] for k in ['Z','d','sig','e_sf','eps_full','Rad','rho','ew','c','closed']}
for ai,Z in enumerate(ATOMS):
    try:
        hf=load_hf(Z)
    except Exception: 
        print("skip Z=%d (no hf)"%Z,flush=True); continue
    r=np.asarray(hf['r']); rho=np.maximum(np.asarray(hf['rho']),1e-12); w=np.asarray(hf['w'])
    o=np.argsort(r); r,rho,w=r[o],rho[o],w[o]; ew=4*np.pi*r**2*w; rf=rho_fn(r,rho)
    s,e=off[ai],off[ai]+npt[ai]
    r0=np.asarray(f['r0'])[s:e]; oo=np.argsort(r0); sub=np.linspace(0,len(oo)-1,min(NPP,len(oo))).astype(int); oo=oo[sub]
    r0=r0[oo]; rt=np.asarray(f['rt'])[s:e][oo]; rho0=np.asarray(f['rho'])[s:e][oo]; Rad0=np.asarray(f['Rad'])[s:e][oo]
    epsf=np.asarray(f['eps_full'])[s:e][oo]; clo=cloflat[s:e][oo]; sig=rt/(-0.5*rho0)[:,None]
    sel=np.array([int(np.argmin(np.abs(r-x))) for x in r0]); ew0=ew[sel]; cc=2*np.pi*Rad0**2*(-0.5*rho0)
    for x,Rd in zip(r0,Rad0): R['d'].append(coeffs(rf,float(x),float(Rd)))
    R['Z']+=[Z]*len(r0); R['sig'].append(sig); R['e_sf'].append(sig@Cmom); R['eps_full'].append(epsf)
    R['Rad'].append(Rad0); R['rho'].append(rho0); R['ew'].append(ew0); R['c'].append(cc); R['closed'].append(clo)
    if ai%10==0: print("...Z=%d (%d/%d atoms)"%(Z,ai+1,len(ATOMS)),flush=True)
D=np.array(R['d']); sig=np.vstack(R['sig']); e_sf=np.concatenate(R['e_sf']); c=np.concatenate(R['c'])
chk=np.abs(e_sf-np.concatenate(R['eps_full'])/c); print("CHECK e_sf vs eps_full/c: max|diff|=%.2e"%chk.max(),flush=True)
np.savez("cache/refs/holes/repr_cache_all.npz",Z=np.array(R['Z']),d=D,sig=sig,e_sf=e_sf,
    eps_full=np.concatenate(R['eps_full']),c=c,Cmom=Cmom,NOUT=NOUT,LMAX=LMAX,
    Rad=np.concatenate(R['Rad']),rho=np.concatenate(R['rho']),ew=np.concatenate(R['ew']),closed=np.concatenate(R['closed']))
print("SAVED repr_cache_all.npz  d=",D.shape," atoms=",len(set(R['Z'])),flush=True)
