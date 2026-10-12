"""1L / 2L reference values from labelled Raman datasets, and layer-number /
stacking predictions for new data against them.

Build the reference (fits every spectrum with raman_peak_fit, stores the
statistics and plots):
    python graphene_reference.py build --ref 1L ~/Downloads/ML_hBN_RuCl3_532nm_map1_2__X_-2372 \
                                       --ref 2L ~/Downloads/BL_hBN_RuCl3_532nm_map1_7__X_-2367

Predict a new dataset (folder of spectra; writes <folder>/output/predictions.csv
and prediction.pdf):
    python graphene_reference.py predict ~/Downloads/new_dataset

Reference statistics are kept twice for every dataset:
  all       every spectrum whose G and 2D fits passed
  matching  only the spectra whose fitted 2D width agrees with the dataset's
            label (1L: FWHM < 35 cm-1; 2L: 42-65 cm-1)
A map named after one layer number can contain other thicknesses (the ML and
BL maps here both contain a monolayer strip and a bilayer region), so
predictions use the 'matching' statistics.

Prediction, per spectrum whose fits passed:
  - distance to each reference in units of its standard deviation, over
    2D FWHM and ln I(2D)/I(G); the nearer reference gives 1L or 2L
  - 2D FWHM above the 2L reference (mean + 3 SD): >=3L, with ABA/ABC from
    the Bouhafs et al. (Carbon 2021) 2D-width table
  - 2D FWHM between the 1L and 2L references (beyond 3 SD of both):
    '2L twisted?' - a single, intermediate-width 2D band is typical of
    twisted bilayers, which these references do not contain
  - 1L-like but 2D blue-shifted by more than 3 SD: '1L or twisted 2L'
  - more than MAX_Z SD from both: unassigned
Stacking for 2L inside the reference: 'AB (Bernal)', since the 2L reference
is Bernal bilayer. These are statistical calls against two references from
one flake; check them against optical contrast or AFM.
"""
import argparse
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np

import raman_peak_fit as rpf
from raman_peak_fit import styled, subplots

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.join(HERE, "reference", "graphene_reference.json")

MATCH = {"1L": (0.0, 35.0), "2L": (42.0, 65.0)}       # fitted 2D FWHM, cm-1
FEATURES = ["G_pos", "G_fwhm", "G_height", "2D_pos", "2D_fwhm", "2D_height",
            "I2D_IG", "A2D_AG", "log_I2D_IG", "Gminus_pos", "Gplus_pos"]
MAX_Z = 4.0
#: Bouhafs et al., Carbon 177 (2021) 282, Fig. 2c: 2D FWHM by layer, cm-1.
BOUHAFS = {3: (45, 65), 4: (50, 70), 5: (55, 75), 6: (58, 80), 7: (60, 85),
           8: (63, 88), 9: (65, 90)}
COLORS = {"1L": "#1f5fa8", "2L": "#e8402a", ">=3L": "#222222",
          "2L twisted?": "#f2a900", "1L or twisted 2L": "#7fb2e5",
          "unassigned": "#bbbbbb", "no fit": "#eeeeee"}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def stats(rows):
    out = {}
    for q in FEATURES:
        v = np.array([r.get(q, np.nan) for r in rows], float)
        v = v[np.isfinite(v)]
        out[q] = {"mean": float(v.mean()) if v.size else None,
                  "std": float(v.std(ddof=1)) if v.size > 1 else None,
                  "median": float(np.median(v)) if v.size else None, "n": int(v.size)}
    return out


def _with_log(rows):
    for r in rows:
        x = r.get("I2D_IG", np.nan)
        r["log_I2D_IG"] = float(np.log(x)) if np.isfinite(x) and x > 0 else np.nan
    return rows


def _xy_from_name(name):
    m = re.search(r"__X_([-+\d.]+)__Y_([-+\d.]+)__", name)
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def build(refs, out_json=DEFAULT_REF):
    """refs: list of (label, folder). Fits each folder into
    <reference dir>/<folder name>/ and writes the JSON reference."""
    root = os.path.dirname(out_json)
    os.makedirs(root, exist_ok=True)
    args = SimpleNamespace(ng="auto", n2d=1, g_range=(1500, 1680),
                           d_range=(2560, 2810), out=None)
    data = {"created": time.strftime("%Y-%m-%d %H:%M"), "match_rule": MATCH,
            "datasets": []}
    for label, folder in refs:
        folder = os.path.expanduser(folder).rstrip(os.sep)
        name = os.path.basename(folder)
        args.out = os.path.join(root, "%s_%s" % (label, name.split("_532nm")[0]))
        rows, _ = rpf.run(folder, args)
        good = _with_log([r for r in rows if r.get("ok")])
        lo, hi = MATCH[label]
        match = [r for r in good if lo <= r["2D_fwhm"] <= hi]
        pts = [{"file": r["file"], "2D_fwhm": r["2D_fwhm"], "I2D_IG": r["I2D_IG"],
                "2D_pos": r["2D_pos"], "match": bool(lo <= r["2D_fwhm"] <= hi)}
               for r in good]
        data["datasets"].append({
            "label": label, "name": name, "path": folder, "output": args.out,
            "n_spectra": len(rows), "n_fitted": len(good), "n_matching": len(match),
            "all": stats(good), "matching": stats(match), "points": pts})
        print("  %s %s: %d spectra, %d fitted, %d matching %s"
              % (label, name, len(rows), len(good), len(match), label))
    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=1)
    plot_reference(data, os.path.join(root, "reference_summary.pdf"))
    print("reference ->", out_json)
    return data


def load(path=DEFAULT_REF):
    with open(path) as fh:
        return json.load(fh)


def ref_stats(ref, label, which="matching"):
    for d in ref["datasets"]:
        if d["label"] == label:
            return d[which]
    raise KeyError("no %s reference" % label)


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

def _z(r, s):
    terms = []
    for q, x in (("2D_fwhm", r["2D_fwhm"]), ("log_I2D_IG", r.get("log_I2D_IG"))):
        m, sd = s[q]["mean"], s[q]["std"]
        if m is None or not sd or x is None or not np.isfinite(x):
            continue
        terms.append(((x - m) / sd) ** 2)
    return float(np.sqrt(sum(terms))) if terms else np.inf


def bouhafs(fwhm, tol=10.0):
    """(N, 'ABA'|'ABC') whose tabulated width is nearest, or (None, None)."""
    best = None
    for n, (aba, abc) in BOUHAFS.items():
        for poly, w in (("ABA", aba), ("ABC", abc)):
            d = abs(fwhm - w)
            if d <= tol and (best is None or d < best[0]):
                best = (d, n, poly)
    return (best[1], best[2]) if best else (None, None)


def predict_row(r, ref):
    """(layer, polytype, note, z_1L, z_2L) for one analyse_spectrum row."""
    if not r.get("ok"):
        return "no fit", "", r.get("note", ""), np.nan, np.nan
    _with_log([r])
    s1, s2 = ref_stats(ref, "1L"), ref_stats(ref, "2L")
    z1, z2 = _z(r, s1), _z(r, s2)
    f = r["2D_fwhm"]
    f1 = s1["2D_fwhm"]["mean"] + 3 * s1["2D_fwhm"]["std"]
    f2lo = s2["2D_fwhm"]["mean"] - 3 * s2["2D_fwhm"]["std"]
    f2hi = s2["2D_fwhm"]["mean"] + 3 * s2["2D_fwhm"]["std"]
    if f > f2hi:
        n, poly = bouhafs(f)
        return (">=3L", ("%dL %s" % (n, poly)) if n else "",
                "2D FWHM %.1f above the 2L reference" % f, z1, z2)
    if f1 < f < f2lo:
        return ("2L twisted?", "",
                "2D FWHM %.1f between the 1L and 2L references" % f, z1, z2)
    if min(z1, z2) > MAX_Z:
        return "unassigned", "", "more than %g SD from both references" % MAX_Z, z1, z2
    if z1 <= z2:
        p1 = s1["2D_pos"]
        if p1["std"] and r["2D_pos"] > p1["mean"] + 3 * p1["std"]:
            return ("1L or twisted 2L", "",
                    "1L-like but 2D blue-shifted to %.1f" % r["2D_pos"], z1, z2)
        return "1L", "—", "", z1, z2
    return "2L", "AB (Bernal)", "", z1, z2


def predict_folder(folder, ref, out=None):
    folder = os.path.expanduser(folder).rstrip(os.sep)
    out = out or os.path.join(folder, "output")
    args = SimpleNamespace(ng="auto", n2d=1, g_range=(1500, 1680),
                           d_range=(2560, 2810), out=out)
    rows, _ = rpf.run(folder, args)
    return write_predictions(rows, ref, out, os.path.basename(folder))


def write_predictions(rows, ref, out, title):
    os.makedirs(out, exist_ok=True)
    preds = []
    for r in rows:
        layer, poly, note, z1, z2 = predict_row(r, ref)
        x, y = r.get("x", np.nan), r.get("y", np.nan)
        if not np.isfinite(x):
            x, y = _xy_from_name(r.get("file", ""))
        preds.append({"file": r.get("file", ""), "x": x, "y": y, "layer": layer,
                      "polytype": poly, "z_1L": z1, "z_2L": z2,
                      "2D_fwhm": r.get("2D_fwhm", np.nan), "I2D_IG": r.get("I2D_IG", np.nan),
                      "2D_pos": r.get("2D_pos", np.nan), "G_pos": r.get("G_pos", np.nan),
                      "G_split": r.get("G_split", False), "note": note})
    import csv
    with open(os.path.join(out, "predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(preds[0]) if preds else ["file"])
        w.writeheader()
        for p in preds:
            w.writerow({k: ("%.5g" % v if isinstance(v, (float, np.floating)) else v)
                        for k, v in p.items()})
    plot_prediction(preds, ref, os.path.join(out, "prediction.pdf"), title)
    counts = {}
    for p in preds:
        key = p["layer"] + (" " + p["polytype"] if p["polytype"] not in ("", "—") else "")
        counts[key] = counts.get(key, 0) + 1
    print("%s: %s  ->  %s" % (title, ", ".join("%s %d" % kv for kv in sorted(counts.items())),
                              os.path.join(out, "predictions.csv")))
    return preds


# ---------------------------------------------------------------------------
# plots (vector PDF)
# ---------------------------------------------------------------------------

def draw_reference(ax, ref, matching_only=False):
    for d in ref["datasets"]:
        c = COLORS[d["label"]]
        pts = d["points"]
        for m in (True, False):
            if matching_only and not m:
                continue
            sel = [p for p in pts if p["match"] == m]
            ax.scatter([p["2D_fwhm"] for p in sel], [p["I2D_IG"] for p in sel], s=6,
                       facecolors=c if m else "none", edgecolors=c, lw=0.4,
                       label="%s ref (%s, %s)" % (d["label"], d["name"].split("_532nm")[0],
                                                  "matching" if m else "other spots"))
    ax.set_xlabel("2D FWHM (cm$^{-1}$)")
    ax.set_ylabel("I(2D)/I(G)")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)


@styled
def plot_reference(ref, path):
    fig, axs = subplots(1, 2, figsize=(17.8 / 2.54, 7 / 2.54))
    draw_reference(axs[0], ref)
    axs[0].legend(fontsize=5, frameon=False)
    ax = axs[1]
    for d in ref["datasets"]:
        v = [p["2D_fwhm"] for p in d["points"]]
        ax.hist(v, bins=np.arange(10, 90, 2), histtype="step", color=COLORS[d["label"]],
                label="%s (%s)" % (d["label"], d["name"].split("_532nm")[0]))
    ax.set_xlabel("2D FWHM (cm$^{-1}$)")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)


@styled
def plot_prediction(preds, ref, path, title):
    has_xy = sum(np.isfinite(p["x"]) for p in preds) > 3
    fig, axs = subplots(1, 2 if has_xy else 1, figsize=(17.8 / 2.54, 7.5 / 2.54),
                            squeeze=False)
    ax = axs[0, 0]
    draw_reference(ax, ref, matching_only=True)
    for lab, c in COLORS.items():
        if lab == "no fit":                     # failed fits carry no usable values
            continue
        sel = [p for p in preds if p["layer"] == lab and np.isfinite(p["2D_fwhm"])]
        if sel:
            ax.scatter([p["2D_fwhm"] for p in sel], [p["I2D_IG"] for p in sel], s=14,
                       marker="x", color=c if lab not in ("1L", "2L") else "k", lw=0.7,
                       label="%s: %s (%d)" % (title[:20], lab, len(sel)))
    ax.legend(fontsize=5, frameon=False)
    if has_xy:
        ax = axs[0, 1]
        for lab, c in COLORS.items():
            sel = [p for p in preds if p["layer"] == lab]
            if sel:
                ax.scatter([p["x"] for p in sel], [p["y"] for p in sel], s=18, marker="s",
                           color=c, lw=0, label="%s (%d)" % (lab, len(sel)))
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser(description="graphene 1L/2L reference and prediction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ref", nargs=2, action="append", metavar=("LABEL", "FOLDER"),
                   required=True, help="LABEL is 1L or 2L")
    b.add_argument("--json", default=DEFAULT_REF)
    p = sub.add_parser("predict")
    p.add_argument("folders", nargs="+")
    p.add_argument("--json", default=DEFAULT_REF)
    a = ap.parse_args()
    if a.cmd == "build":
        build([(lab, f) for lab, f in a.ref], a.json)
    else:
        if not os.path.exists(a.json):
            raise SystemExit("No reference at %s - run 'build' first." % a.json)
        ref = load(a.json)
        bad = 0
        for f in a.folders:
            d = os.path.expanduser(f)
            if not os.path.isdir(d):
                print("skipped %s: no such folder" % f)
                bad += 1
                continue
            if not any(n.lower().endswith(rpf.SPECTRUM_EXT) for n in os.listdir(d)):
                print("skipped %s: no .txt/.dat/.csv spectra in it" % f)
                bad += 1
                continue
            predict_folder(d, ref)
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
