"""Phase C: enrich exact-hole references using the fleet's CACHED orbitals (no re-solving).
For every converged closed-shell atom with cached HF orbitals, build the moment-matched
references (project exact hole onto adaptive n_out=10 frame, then pin charge=-1, on-top=-rho/2,
Coulomb=exact_eps_x_l). Carries over Hg from the existing hole_refs.npz (still solving in fleet).
Writes the enriched hole_refs.npz."""
import os, numpy as np
from numpy.polynomial.legendre import leggauss
np.seterr(all="ignore")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_EXPANSION, SIMPLEHOLEEXPParameters
from atom.xc import orbital_hole as oh
from atom.descriptors.simple.bessel import RadialBesselBasis
from atom.descriptors.simple.derivatives import build_spectral_gradient_operator
from atom.utils.periodic import atomic_number_to_name as nm
SCR = "/private/tmp/claude-501/-Users-ajm-Library-CloudStorage-Dropbox-GaTech-Andrew-Medford-amedford6-admin-admin-coding-SIMPLE-hole-functional/a2aa6175-98ec-455a-9307-c93f60e2052e/scratchpad"
REF = SCR + "/refs"
NOUT = 10
tu, wu = leggauss(120); t = 0.5 * (tu + 1.0); wt = 0.5 * wu
Rb1 = RadialBesselBasis(NOUT - 1, 0, 1.0).evaluate(0, t)
R0u = (np.arange(NOUT) + 1) * np.pi * np.sqrt(2.0)
CLOSED = [2, 4, 10, 12, 18, 20, 30, 36, 38, 48, 54, 56, 80]

# carry over Hg (and any atom not freshly built) from the existing file
prev = dict(np.load(SCR + "/hole_refs.npz")) if os.path.exists(SCR + "/hole_refs.npz") else {}
out = {}
print(f"{'atom':>4} {'E_x(HF)':>10} {'moment-match':>13} {'err(mHa)':>9} {'reloc%':>7} {'npts':>5}")
built = set()
for Z in CLOSED:
    p = f"{REF}/hf_Z{Z:02d}.npz"
    if not os.path.exists(p): continue
    hf = np.load(p, allow_pickle=True)
    if not bool(hf["converged"]) or hf["g_sorted"].size == 0: continue
    name = nm(Z)
    r = np.asarray(hf["r"]); rho = np.maximum(np.asarray(hf["rho"]), 1e-12); w = np.asarray(hf["w"])
    o = np.argsort(r); r, rho, w = r[o], rho[o], w[o]; ew = 4*np.pi*r**2*w
    Ehf = float(hf["Ehf"])
    r_sorted = np.asarray(hf["r_sorted"]); g_sorted = np.asarray(hf["g_sorted"])
    occ = np.asarray(hf["occ"]); lval = np.asarray(hf["l_values"])
    F = SIMPLE_HOLE_EXPANSION(r_quad=r, quadrature_weights=w, params=SIMPLEHOLEEXPParameters())
    C = np.array([op @ rho for op in F._ops]); Rad, _ = F._R_ad(rho)
    c_ad = F._c_ad(C, Rad); cn = c_ad / np.where(np.abs(c_ad[:, :1]) > 1e-30, c_ad[:, :1], 1e-30)
    al = F._G[0]; be = F._H[0]
    d = c_ad / (4*np.pi*Rad[:, None]**1.5); Q = 4*np.pi*Rad**3*(d @ al)
    gop = build_spectral_gradient_operator(F._r_grid); kF = (3*np.pi**2*rho)**(1/3)
    s = np.abs(gop @ rho)/(2*kF*rho)
    sig = rho > 1e-4*rho.max(); idx = np.where(sig)[0]; sel = idx[np.linspace(0, len(idx)-1, 40).astype(int)]
    rt = np.zeros((len(sel), NOUT)); reloc = np.zeros(len(sel))
    for k, i in enumerate(sel):
        u = Rad[i]*t
        nx = oh.exchange_hole(float(r[i]), u, r_sorted, g_sorted, occ, lval, n_mu=64)
        rt0 = Rb1 @ (nx * t**2 * wt)
        eps_full = oh.exact_eps_x_l(float(r[i]), r_sorted, g_sorted, occ, lval, n_u=128, n_mu=64)
        a_row = 4*np.pi*Rad[i]**3 * al; e_row = 2*np.pi*Rad[i]**2 * be
        A3 = np.stack([a_row, R0u, e_row])
        tgt = np.array([-1.0, -0.5*rho[i], eps_full])
        lam = np.linalg.solve(A3 @ A3.T, tgt - A3 @ rt0)
        rt[k] = rt0 + A3.T @ lam; reloc[k] = np.linalg.norm(A3.T@lam)/max(np.linalg.norm(rt0),1e-30)
    eps_sel = 2*np.pi*Rad[sel]**2*(rt @ be)
    eps = np.interp(r, r[sel], eps_sel); Emm = float(np.sum(ew * rho * eps))
    out[name+"_cn"]=cn[sel]; out[name+"_s"]=s[sel]; out[name+"_Q"]=Q[sel]; out[name+"_rt"]=rt
    out[name+"_Rad"]=Rad[sel]; out[name+"_rho"]=rho[sel]; out[name+"_Ehf"]=Ehf
    out[name+"_eps_sel"]=eps_sel; out[name+"_rsel"]=r[sel]
    built.add(name)
    print(f"{name:>4} {Ehf:10.4f} {Emm:13.4f} {1e3*(Emm-Ehf):9.1f} {1e2*reloc.mean():7.1f} {len(sel):5d}", flush=True)
# carry over atoms present before but not rebuilt (e.g. Hg if fleet hasn't reached it)
for k, v in prev.items():
    a = k.rsplit("_", 1)[0]
    if a not in built and a in {"Hg"}:
        out[k] = v
np.savez(SCR + "/hole_refs.npz", **out)
allatoms = sorted({k.rsplit('_',1)[0] for k in out})
print(f"\nCACHED enriched hole_refs.npz: {allatoms}")
