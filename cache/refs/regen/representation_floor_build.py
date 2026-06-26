import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP
from atom.descriptors.simple.pipeline import window_basis, transfer_matrix
from atom.descriptors.simple.bessel import radial_gauss_grid
from cache.refs.loader import load_hf, load_hole_refs_full
NIN,NOUT,RC,LMAX,NPP=20,10,6.0,6,80
quad=radial_gauss_grid(RC,512); qn=np.asarray(quad.nodes); qw=np.asarray(quad.weights)
u,wu=leggauss(64); BAS={l:window_basis(l,NIN).evaluate(l,qn) for l in range(LMAX+1)}
ANG={l:np.sqrt(4*np.pi/(2*l+1)) for l in range(LMAX+1)}; PLv={l:eval_legendre(l,u) for l in range(LMAX+1)}
Cmom=SIMPLE_HOLE_KERNEL_FP(r_quad=np.linspace(1e-3,14,400),quadrature_weights=np.gradient(np.linspace(1e-3,14,400)))._Cmom
f=load_hole_refs_full(); Zf=np.asarray(f['atom_Z']); off=np.asarray(f['atom_offset']); npt=np.asarray(f['atom_npts'])
CLOSED=[2,4,10,12,18,20,30,36,48,54]
def rho_fn(r,rho): rs=np.sort(r); rr=rho[np.argsort(r)]; return lambda x: np.interp(np.asarray(x),rs,rr,left=rr[0],right=0.0)
def coeffs(rf,r0,Rad):
    dist=np.sqrt(np.maximum(r0**2+qn[:,None]**2-2*qn[:,None]*r0*u[None,:],0.0)); rv=rf(dist); out=[]
    for l in range(LMAX+1):
        prof=(2*l+1)/2.0*(rv*(wu*PLv[l])[None,:]).sum(1)
        out.append(transfer_matrix(l,Rad,NOUT,NIN)@(ANG[l]*BAS[l]@(qw*qn**2*prof)))
    n0=out[0][0] if abs(out[0][0])>1e-30 else 1e-30
    return np.concatenate([o/n0 for o in out])    # (LMAX+1)*NOUT, monopole-normalized
rows={'Z':[],'d':[],'sig':[],'e_sf':[],'eps_full':[],'Rad':[],'rho':[],'ew':[],'c':[]}
for Z in CLOSED:
    hf=load_hf(Z); r=np.asarray(hf['r']); rho=np.maximum(np.asarray(hf['rho']),1e-12); w=np.asarray(hf['w'])
    o=np.argsort(r); r,rho,w=r[o],rho[o],w[o]; ew=4*np.pi*r**2*w; rf=rho_fn(r,rho)
    ai=int(np.where(Zf==Z)[0][0]); s,e=off[ai],off[ai]+npt[ai]
    r0=np.asarray(f['r0'])[s:e]; oo=np.argsort(r0); sub=np.linspace(0,len(oo)-1,NPP).astype(int); oo=oo[sub]
    r0=r0[oo]; rt=np.asarray(f['rt'])[s:e][oo]; rho0=np.asarray(f['rho'])[s:e][oo]; Rad0=np.asarray(f['Rad'])[s:e][oo]
    epsf=np.asarray(f['eps_full'])[s:e][oo]; sig=rt/(-0.5*rho0)[:,None]
    sel=np.array([int(np.argmin(np.abs(r-x))) for x in r0]); ew0=ew[sel]
    c=2*np.pi*Rad0**2*(-0.5*rho0)
    for k,(x,R) in enumerate(zip(r0,Rad0)):
        rows['d'].append(coeffs(rf,float(x),float(R)))
    rows['Z']+=[Z]*len(r0); rows['sig'].append(sig); rows['e_sf'].append(sig@Cmom)
    rows['eps_full'].append(epsf); rows['Rad'].append(Rad0); rows['rho'].append(rho0); rows['ew'].append(ew0); rows['c'].append(c)
    print("done Z=%d (%d pts)"%(Z,len(r0)),flush=True)
D=np.array(rows['d']); sig=np.vstack(rows['sig']); e_sf=np.concatenate(rows['e_sf']); eps_full=np.concatenate(rows['eps_full'])
c=np.concatenate(rows['c']); 
# verify e_sf == eps_full/c
chk=np.abs(e_sf - eps_full/c); print("CHECK e_sf vs eps_full/c: max|diff|=%.2e median=%.2e"%(chk.max(),np.median(chk)),flush=True)
np.savez(os.path.join(os.path.dirname("%s/"%os.path.dirname(__file__)),"repr_cache.npz") if False else os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","holes","repr_cache.npz"),
    Z=np.array(rows['Z']),d=D,sig=sig,e_sf=e_sf,eps_full=eps_full,c=c,Cmom=Cmom,NOUT=NOUT,LMAX=LMAX,
    Rad=np.concatenate(rows['Rad']),rho=np.concatenate(rows['rho']),ew=np.concatenate(rows['ew']))
print("SAVED repr_cache.npz shape d=",D.shape,flush=True)
