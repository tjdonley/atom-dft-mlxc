import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.hf import CoulombCouplingCalculator as CC
from cache.refs.loader import load_hf, load_hole_refs_full
SY={2:'He',7:'N',8:'O',10:'Ne',12:'Mg',15:'P'}
pool=load_hole_refs_full()
def wigner_resolved(Z):
    hf=load_hf(Z); rs=np.asarray(hf['r_sorted']); g=np.asarray(hf['g_sorted']); occ=np.asarray(hf['occ'],float); lv=np.asarray(hf['l_values'],int)
    r=np.asarray(hf['r']); w=np.asarray(hf['w']); o=np.argsort(r); 
    # weights aligned to r_sorted (r_sorted = sorted r)
    ws=np.interp(rs,r[o],w[o])
    u=rs[:,None]*g                                  # reduced radial wavefn u=rR (hf.py convention)
    maxl=int(lv.max()); dens=np.zeros(len(rs))
    for L in range(0,2*maxl+1):
        wig=np.array([[CC.wigner_3j_000(int(a),int(b),L)**2 for b in lv] for a in lv])
        occm=occ[:,None]*occ[None,:]; rk=CC.radial_kernel(L,rs,ws)
        dens+=-0.25*(2*L+1)*np.einsum('ij,li,ki,kj,lj,kl->l',occm*wig,u,u,u,u,rk,optimize=True)
    return rs,dens,float(hf['Ehf'])
print("Radially-resolved HF exchange energy density WITH Wigner coupling (hf.py machinery), summed:")
print("%4s %12s %14s %16s"%("atom","Ehf","sum(Wigner)","sum(spherical eps_full)"))
for Z in (2,10,12,7,8,15):
    rs,dens,Ehf=wigner_resolved(Z)
    m=pool['Z']==Z; r0=pool['r0'][m]; ef=pool['eps_full'][m]; rho_p=pool['rho'][m]
    hf=load_hf(Z); r=np.asarray(hf['r']); rho=np.maximum(np.asarray(hf['rho']),1e-12); w=np.asarray(hf['w']); oo=np.argsort(r)
    epsx_sph=np.interp(r[oo],np.sort(r0),ef[np.argsort(r0)]); E_sph=float(np.sum(4*np.pi*r[oo]**2*w[oo]*rho[oo]*epsx_sph))
    print("%4s %12.4f %14.4f %16.4f"%(SY[Z],Ehf,dens.sum(),E_sph),flush=True)
