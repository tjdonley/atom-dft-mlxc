import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from cache.refs.loader import load_hf, load_hole_refs_full
p=load_hole_refs_full()
SY={2:'He',4:'Be',7:'N',8:'O',10:'Ne',12:'Mg',15:'P',18:'Ar',20:'Ca'}
print("Does the EXACT per-point energy density (eps_full) integrate to the HF total Ehf?")
print("%-4s %12s %12s %10s   %s"%("atom","Ehf","E[eps_full]","err(mHa)","npts(pool/grid)"))
for Z in (2,4,10,12,18,7,8,15,20):
    m=p['Z']==Z; r0=p['r0'][m]; ef=p['eps_full'][m]; rho_p=p['rho'][m]
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    epsx=np.interp(r,np.sort(r0),ef[np.argsort(r0)])           # exact eps_x density on the HF grid
    Erec=float(np.sum(4*np.pi*r**2*w*rho*epsx))
    print("%-4s %12.4f %12.4f %10.0f   %d/%d"%(SY[Z],Ehf,Erec,1e3*(Erec-Ehf),m.sum(),len(r)),flush=True)
