import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
DATA="atom/xc/data"; f=load_hole_refs_full(); Zf=np.asarray(f['Z']); r0f=np.asarray(f['r0'])
# smoke test: instantiate use_l2_power, run reference-free on Ne
hf=load_hf(10); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(use_l2_power=True,fp_l0=0.7,fp_l1=0.5))
cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
eps=F._kernel_eps(cp,rho,g); print("SMOKE: reference-free use_l2_power runs; eps finite=%s, sum=%.4f"%(np.all(np.isfinite(eps)),float(np.sum(F.energy_weights*rho*eps))))
# GEA slope check (use_l2_power, no refs)
s=np.linspace(0,30,6000); sm=(s>1e-6)&(s<0.2); print("GEA slope (use_l2_power, no refs): %.5f (target %.5f)"%(np.polyfit(s[sm]**2,F._fx_gea_axis(s)[sm]-1,1)[0],10/81))
# p2 per atom at ref radii (consistent _l2_power_feat) -> add column to n1098
_pc={}
def p2_atom(Z):
    if Z in _pc: return _pc[Z]
    h=load_hf(Z); oo=np.argsort(np.asarray(h['r'])); rr=np.asarray(h['r'])[oo]; rh=np.maximum(np.asarray(h['rho'])[oo],1e-12); ww=np.asarray(h['w'])[oo]
    G=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=ww,params=P(use_l2_power=True)); R_ad,_=G._R_ad(rh)
    cpp=np.array([op@rh for op in G._ops]); c_ad=G._c_ad(cpp,R_ad); p2=G._l2_power_feat(rh,R_ad,c_ad[:,0])
    _pc[Z]=(rr,p2); return _pc[Z]
z=np.load(DATA+"/kernel_fp_refs_closed_n1098.npz"); idx=z['idx']; Zr=Zf[idx]; r0r=r0f[idx]; p2col=np.zeros(len(idx))
for Z in np.unique(Zr):
    rr,p2=p2_atom(int(Z)); m=Zr==Z; p2col[m]=np.interp(r0r[m],rr,p2)
out={k:z[k] for k in z.files}; out['X']=np.column_stack([z['X'],p2col]); np.savez(DATA+"/kernel_fp_refs_closed_n1098_l2pow.npz",**out)
print("built n1098_l2pow; p2 range [%.3f,%.3f]"%(p2col.min(),p2col.max()),flush=True)
IN=[2,10,18,36,54]
def ex(Z,refs,l2pow,ridge=1e-8):
    h=load_hf(Z); oo=np.argsort(np.asarray(h['r'])); rr=np.asarray(h['r'])[oo]; rh=np.maximum(np.asarray(h['rho'])[oo],1e-12); ww=np.asarray(h['w'])[oo]; Ehf=float(h['Ehf'])
    G=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=ww,params=P(use_l2_power=l2pow,fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.5,fp_ref_ridge=ridge,refs_path=os.path.abspath(os.path.join(DATA,refs))))
    cpp=np.array([op@rh for op in G._ops]); gg=G._grad_op@rh
    return 1e3*(float(np.sum(G.energy_weights*rh*G._kernel_eps(cpp,rh,gg)))-Ehf)
print("\nin-domain err vs Ehf (mHa), DENSE n1098, exact interp. no-l2pow vs +l2pow(power spectrum):")
print("%4s %14s %18s"%("atom","n1098 no-l2pow","n1098 +l2pow"))
for Z in IN: print("%4d %14.0f %18.0f"%(Z,ex(Z,"kernel_fp_refs_closed_n1098.npz",False),ex(Z,"kernel_fp_refs_closed_n1098_l2pow.npz",True)),flush=True)
