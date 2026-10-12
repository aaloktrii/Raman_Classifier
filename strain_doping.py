"""ω_G - ω_2D correlation (strain / doping vector decomposition) for Raman
maps, split by layer number and substrate, with peak counting.

Per spectrum:
  1. Si calibration: the axis is shifted so the spectrum's own Si peak sits at
     520.7 cm-1 (raman_tools.si_shift).
  2. Substrate: 'RuCl3' when the RuCl3 mode at ~164 cm-1 is taller than
     RU_SNR x noise, otherwise 'hBN'.
  3. Layer number and stacking from graphene_reference (1L / 2L reference).
  4. Peak count in the G (1500-1680) and 2D (2560-2810) ranges, chosen from
     the data (raman_tools.fit_peaks, n='auto').
  5. Correlation points. A spectrum whose G band is split (two G peaks found)
     is treated as two layers doped differently: G and 2D are fitted with two
     Lorentzians each and paired by order, (G-, 2D-) and (G+, 2D+). Otherwise
     one Lorentzian each gives (G, 2D).

Decomposition (Lee et al., Nat. Commun. 3, 1024 (2012)):
    Δω = ε (a_G, s_ε a_G) + n (b_G, s_n b_G)
with origin (ω_G0, ω_2D0), biaxial-strain slope s_ε = 2.2 and hole-doping
slope s_n = 0.70. PARAMS defaults were read off the user's figure (origin and
doping rate are estimates; edit them to the published values).

Run:
    python strain_doping.py FOLDER [FOLDER ...] [--out DIR]
"""
import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import graphene_reference as gref
import raman_peak_fit as rpf
import raman_tools as rt
from raman_peak_fit import styled, subplots

PARAMS = {
    "G0": 1578.3,          # cm-1, origin read off the figure
    "D0": 2683.6,
    "a_G": -69.1,          # cm-1 per % biaxial strain (Lee 2012)
    "s_strain": 2.2,       # Δω2D/ΔωG for strain
    "b_G": -14.1,          # cm-1 per 1e13 cm-2 (negative n = holes); figure grid
    "s_doping": 0.70,      # Δω2D/ΔωG for hole doping
}
RU_SNR = 10.0
PHYS_G, PHYS_2D = (1560.0, 1630.0), (2640.0, 2780.0)   # plausible graphene positions
G_RANGE, D_RANGE = (1500.0, 1680.0), (2560.0, 2810.0)
GROUP_STYLE = {                       # (colour, marker)
    "1L hBN": ("#2a8c4a", "o"), "1L RuCl3": ("#8a5a00", "o"),
    "2L hBN": ("#7b3fa0", "D"),
    "2L RuCl3 G-,2D-": ("#d62728", "o"), "2L RuCl3 G+,2D+": ("#1f3fa8", "s"),
    "2L RuCl3": ("#e8402a", "D"),
}


def decompose(wG, w2D, p=PARAMS):
    """(strain %, doping 1e13 cm-2) for each (ω_G, ω_2D)."""
    A = np.array([[p["a_G"], p["b_G"]],
                  [p["s_strain"] * p["a_G"], p["s_doping"] * p["b_G"]]])
    d = np.vstack([np.asarray(wG, float) - p["G0"], np.asarray(w2D, float) - p["D0"]])
    eps, n = np.linalg.solve(A, d)
    return eps, n


def _point(p, eps, n):
    return (p["G0"] + p["a_G"] * eps + p["b_G"] * n,
            p["D0"] + p["s_strain"] * p["a_G"] * eps + p["s_doping"] * p["b_G"] * n)


def draw_grid(ax, p=PARAMS, eps_ticks=(-0.1, -0.2, -0.3), n_ticks=(-0.5, -1, -1.5, -2),
              xlim=None, ylim=None):
    """Strain axis (n = 0) and doping axis (ε = 0) as solid lines, iso-strain
    lines dashed (parallel to the doping axis), iso-doping lines dotted
    (parallel to the strain axis), as in Lee et al. 2012."""
    big = 5.0
    e_ext, n_ext = (min(eps_ticks) - 0.1, 0.2), (min(n_ticks) - 1.0, 1.0)
    x, y = zip(_point(p, e_ext[0], 0), _point(p, e_ext[1], 0))
    ax.plot(x, y, color="k", lw=0.8, zorder=1)
    x, y = zip(_point(p, 0, n_ext[0]), _point(p, 0, n_ext[1]))
    ax.plot(x, y, color="k", lw=0.8, zorder=1)
    for e in eps_ticks:
        x, y = zip(_point(p, e, 0), _point(p, e, -big))
        ax.plot(x, y, color="#999", lw=0.6, ls=(0, (4, 3)), zorder=0)
        tx, ty = _point(p, e, 0)
        ax.text(tx - 0.6, ty + 0.8, "%g" % e, fontsize=7, ha="right")
    for n in n_ticks:
        x, y = zip(_point(p, -big / 10, n), _point(p, big / 10, n))
        ax.plot(x, y, color="#999", lw=0.6, ls=(0, (1, 2)), zorder=0)
    ax.plot([p["G0"]], [p["D0"]], marker="s", ms=7, mfc="#4caf50", mec="k", mew=0.6,
            zorder=3)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    # doping labels where the iso-doping lines leave the bottom of the frame
    y0 = ax.get_ylim()[0]
    for n in n_ticks:
        # point on the line ε-parameterised: solve ω2D(ε, n) = y0 + small margin
        e = (y0 + 1.5 - p["D0"] - p["s_doping"] * p["b_G"] * n) / (p["s_strain"] * p["a_G"])
        tx, ty = _point(p, e, n)
        if ax.get_xlim()[0] < tx < ax.get_xlim()[1]:
            ax.text(tx, ty, "%g" % n, fontsize=7, ha="center", va="bottom")
    ax.text(0.98, 0.02, "n (10$^{13}$ cm$^{-2}$)", transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom")
    ax.text(0.98, 0.98, "ε (%)", transform=ax.transAxes, fontsize=8, ha="right", va="top")


# ---------------------------------------------------------------------------
# per-spectrum analysis
# ---------------------------------------------------------------------------

def analyse(wn, y, ref):
    y = np.asarray(y, float)
    shift, si = rt.si_shift(wn, y)
    w = wn + shift
    nz = rt.noise_level(w, y)
    s = (w > 150) & (w < 180)
    fl = ((w > 120) & (w < 140)) | ((w > 190) & (w < 210))
    ru_h = float(np.max(y[s]) - np.median(y[fl])) if s.any() and fl.any() else np.nan
    sub = "RuCl3" if np.isfinite(ru_h) and ru_h > RU_SNR * nz else "hBN"
    r, _, _ = rpf.analyse_spectrum(w, y)
    layer, poly, note, z1, z2 = gref.predict_row(r, ref)
    G = rt.analyse_band(w, y, *G_RANGE, n="auto", nmax=3, noise=nz)
    D = rt.analyse_band(w, y, *D_RANGE, n="auto", nmax=4, noise=nz)
    out = {"si_pos": None if si is None or not si.get("ok") else si["comps"][0]["pos"],
           "si_shift": shift, "si_height": None if si is None or not si.get("ok")
           else si["comps"][0]["height"], "noise": nz, "RuCl3_164_height": ru_h,
           "substrate": sub, "layer": layer, "polytype": poly, "note": note,
           "G_npeaks": G.get("n", 0), "2D_npeaks": D.get("n", 0),
           "G_peaks": ";".join("%.1f/%.1f" % (c["pos"], c["fwhm"]) for c in G.get("comps", [])),
           "2D_peaks": ";".join("%.1f/%.1f" % (c["pos"], c["fwhm"]) for c in D.get("comps", [])),
           "points": []}
    if layer not in ("1L", "2L") or not G.get("ok") or not D.get("ok"):
        return out
    split = layer == "2L" and G["n"] >= 2
    if split:
        G2 = rt.analyse_band(w, y, *G_RANGE, n=2, noise=nz,
                             starts=[(1585, 10, 1), (1603, 10, 1)])
        D2 = rt.analyse_band(w, y, *D_RANGE, n=2, noise=nz,
                             starts=[(2695, 30, 1), (2725, 30, 1)])
        if G2.get("ok") and D2.get("ok") and len(G2["comps"]) == 2 and len(D2["comps"]) == 2:
            for k, tag in enumerate(("G-,2D-", "G+,2D+")):
                out["points"].append(("%s %s %s" % (layer, sub, tag),
                                      G2["comps"][k]["pos"], D2["comps"][k]["pos"]))
            return out
    G1 = rt.analyse_band(w, y, *G_RANGE, n=1, noise=nz)
    D1 = rt.analyse_band(w, y, *D_RANGE, n=1, noise=nz)
    if G1.get("ok") and D1.get("ok"):
        out["points"].append(("%s %s" % (layer, sub), G1["comps"][0]["pos"],
                              D1["comps"][0]["pos"]))
    return out


def _job(args):
    path, ref = args
    wn, y = rpf.read_spectrum(path)
    r = analyse(wn, y, ref)
    m = re.search(r"__X_([-+\d.]+)__Y_([-+\d.]+)__", os.path.basename(path))
    r.update(file=os.path.basename(path), x=float(m.group(1)) if m else np.nan,
             y=float(m.group(2)) if m else np.nan)
    return r


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


class single_threaded_workers:
    """Worker processes inherit the environment when they start: one BLAS
    thread each, or a pool of N workers runs N x N threads on N cores and
    the tiny least-squares solves spend their time contending."""

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        for k in _THREAD_VARS:
            os.environ[k] = "1"

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _job_arrays(args):
    wn, y, ref = args
    return analyse(wn, y, ref)


def analyse_many(spectra, ref, workers=None, progress=None):
    """analyse() over [(wn, y), ...] in parallel processes. progress(i, n) is
    called as results arrive (for a GUI progress bar)."""
    out = [None] * len(spectra)
    with single_threaded_workers(), ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job_arrays, (wn, y, ref)): i for i, (wn, y) in enumerate(spectra)}
        from concurrent.futures import as_completed
        for k, f in enumerate(as_completed(futs)):
            out[futs[f]] = f.result()
            if progress and progress(k + 1, len(spectra)) is False:
                ex.shutdown(cancel_futures=True)
                return None
    return out


def analyse_folder(folder, ref, workers=None):
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(rpf.SPECTRUM_EXT))
    with single_threaded_workers(), ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, [(f, ref) for f in files], chunksize=4))


def write_csv(rows, path):
    keys = ["file", "x", "y", "substrate", "layer", "polytype", "G_npeaks", "2D_npeaks",
            "G_peaks", "2D_peaks", "si_pos", "si_shift", "si_height", "RuCl3_164_height",
            "noise", "note"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["point_group", "w_G", "w_2D", "strain_pct", "doping_1e13"])
        for r in rows:
            base = [("%.5g" % r[k]) if isinstance(r.get(k), float) else r.get(k, "")
                    for k in keys]
            if not r["points"]:
                w.writerow(base)
            for g, wg, wd in r["points"]:
                e, n = decompose([wg], [wd])
                w.writerow(base + [g, "%.2f" % wg, "%.2f" % wd, "%.4f" % e[0], "%.3f" % n[0]])


def draw_correlation(ax, rows, p=PARAMS):
    """Scatter the correlation points of `rows` by group over the grid."""
    groups = {}
    for r in rows:
        for g, wg, wd in r["points"]:
            # positions outside the physical windows are failed fits
            if PHYS_G[0] < wg < PHYS_G[1] and PHYS_2D[0] < wd < PHYS_2D[1]:
                groups.setdefault(g, []).append((wg, wd))
    for g in sorted(groups, key=lambda k: list(GROUP_STYLE).index(k) if k in GROUP_STYLE else 99):
        c, m = GROUP_STYLE.get(g, ("#777", "o"))
        a = np.array(groups[g])
        ax.scatter(a[:, 0], a[:, 1], s=9, color=c, marker=m, lw=0, alpha=0.85, zorder=2,
                   label="%s (%d)" % (g, len(a)))
    allp = np.array([pt for v in groups.values() for pt in v]) if groups else np.zeros((0, 2))
    # limits from the bulk of the data (1st-99th percentile) and the origin
    if allp.size:
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    else:
        lo, hi = np.array([p["G0"], p["D0"]]), np.array([p["G0"], p["D0"]])
    xlim = (min(p["G0"], lo[0]) - 2, max(p["G0"] + 26, hi[0]) + 2)
    ylim = (min(p["D0"], lo[1]) - 4, max(p["D0"] + 58, hi[1]) + 4)
    draw_grid(ax, p, xlim=xlim, ylim=ylim)
    ax.set_xlabel("ω$_G$ (cm$^{-1}$)")
    ax.set_ylabel("ω$_{2D}$ (cm$^{-1}$)")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")


@styled
def plot_correlation(rows, path, title="", p=PARAMS):
    fig, ax = subplots(1, 1, figsize=(8.7 / 2.54, 8.0 / 2.54))
    draw_correlation(ax, rows, p)
    if title:
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


def summarise(rows):
    """Peak-count table: {(layer, substrate): Counter of (nG, n2D)}."""
    from collections import Counter
    t = {}
    for r in rows:
        if r["layer"] in ("1L", "2L"):
            t.setdefault((r["layer"], r["substrate"]), Counter())[(r["G_npeaks"], r["2D_npeaks"])] += 1
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", help="folder for the combined plot (default: first folder's output)")
    ap.add_argument("--json", default=gref.DEFAULT_REF)
    a = ap.parse_args()
    ref = gref.load(a.json)
    every = []
    for f in a.folders:
        f = os.path.expanduser(f).rstrip(os.sep)
        rows = analyse_folder(f, ref)
        out = os.path.join(f, "output")
        os.makedirs(out, exist_ok=True)
        write_csv(rows, os.path.join(out, "strain_doping.csv"))
        plot_correlation(rows, os.path.join(out, "strain_doping.pdf"), os.path.basename(f))
        every += rows
        print("%s -> %s" % (os.path.basename(f), out))
        for (lay, sub), cnt in sorted(summarise(rows).items()):
            print("   %s on %-5s n=%3d   (G peaks, 2D peaks): %s" % (
                lay, sub, sum(cnt.values()),
                ", ".join("%s:%d" % (k, v) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]))))
    if len(a.folders) > 1:
        out = os.path.expanduser(a.out) if a.out else os.path.join(
            os.path.expanduser(a.folders[0]).rstrip(os.sep), "output")
        os.makedirs(out, exist_ok=True)
        plot_correlation(every, os.path.join(out, "strain_doping_combined.pdf"), "combined")
        write_csv(every, os.path.join(out, "strain_doping_combined.csv"))
        print("combined ->", out)


if __name__ == "__main__":
    main()
