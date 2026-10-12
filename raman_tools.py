"""Spectral preprocessing and Lorentzian peak fitting with peak counting.

    si_shift            calibrate a spectrum on its own Si substrate peak
    background          baseline: none / linear / polynomial (iterative) / ALS
    noise_level         point-to-point noise of a band-free stretch
    fit_peaks           sum of Lorentzians on a residual linear baseline, with
                        a fixed number of peaks or the number chosen from the
                        data (n='auto')

Peak counting (n='auto'). Peaks are added one at a time, each new one
started at the largest positive residual of the previous fit, and every
model is refitted in full. A model with k+1 peaks replaces the k-peak model
only if all of these hold:
  - its BIC is lower by at least MIN_DBIC (10 = 'very strong' evidence),
  - the new peak is at least MIN_SNR × noise tall,
  - every FWHM is at least the instrument resolution (or 2 data steps),
  - every peak carries at least MIN_AREA_FRAC of the band area,
  - no two centres are closer than MIN_SEP_FRAC × their mean FWHM (two
    equal Lorentzians only show a dip between them beyond ~0.58 FWHM),
  - no parameter sits on its bound.
The search stops at nmax. Widths are Lorentzian FWHM; they include the
instrument broadening, which the result reports alongside.

Uncertainties are 1-sigma from the fit covariance, scaled by the reduced
chi-square (curve_fit default), so they reflect the actual residual scatter.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve

SI_REF = 520.7          # cm-1, crystalline Si at room temperature
MIN_DBIC = 10.0
MIN_SNR = 3.0           # an added peak must be this tall (x noise)
MIN_SNR_PRESENT = 5.0   # ...and the band's strongest peak this tall, or n = 0
MIN_SEP_FRAC = 0.6     # two Lorentzians show a dip only beyond ~FWHM/sqrt(3)
MIN_AREA_FRAC = 0.05
BACKGROUNDS = ["none", "linear", "polynomial", "als"]


# ---------------------------------------------------------------------------
# calibration and background
# ---------------------------------------------------------------------------

def si_shift(wn, y, ref=SI_REF, window=(500.0, 540.0)):
    """Shift (cm-1) that puts this spectrum's Si peak at `ref`, and the Si
    fit. Returns (0, None) when there is no usable Si peak."""
    s = (wn >= window[0]) & (wn <= window[1])
    if s.sum() < 10:
        return 0.0, None
    r = fit_peaks(wn[s], y[s] - np.median(np.r_[y[s][:5], y[s][-5:]]), n=1)
    if not r["ok"] or r["comps"][0]["height"] < 10 * noise_level(wn, y):
        return 0.0, r
    return ref - r["comps"][0]["pos"], r


def background(wn, y, method="linear", order=3, lam=1e5, p=0.01, niter=100,
               edge=10):
    """Baseline under y (same length).

    linear      straight line through the medians of the first and last
                `edge` points - for a narrow window around one band
    polynomial  iterative modified polyfit (Lieber & Mahadevan-Jansen 2003):
                fit, clip the data to the fit, refit; peaks are pushed out
    als         asymmetric least squares (Eilers & Boelens 2005); lam sets
                stiffness (1e4-1e7), p the asymmetry (0.001-0.05)
    """
    if method == "none":
        return np.zeros_like(y)
    if method == "linear":
        e = max(2, min(edge, y.size // 4))
        x0, x1 = np.median(wn[:e]), np.median(wn[-e:])
        y0, y1 = np.median(y[:e]), np.median(y[-e:])
        return y0 + (y1 - y0) * (wn - x0) / (x1 - x0)
    if method == "polynomial":
        xs = (wn - wn.mean()) / (np.ptp(wn) or 1.0)
        work = y.astype(float).copy()
        for _ in range(niter):
            fit = np.polyval(np.polyfit(xs, work, order), xs)
            new = np.minimum(work, fit)
            if np.allclose(new, work, rtol=0, atol=1e-6 * (np.ptp(y) or 1)):
                break
            work = new
        return fit
    if method == "als":
        L = y.size
        Dm = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        DD = lam * Dm.dot(Dm.transpose())
        w = np.ones(L)
        for _ in range(min(niter, 20)):
            W = sparse.spdiags(w, 0, L, L)
            z = spsolve((W + DD).tocsc(), w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
    raise ValueError("unknown background method %r" % method)


def noise_level(wn, y):
    """Point-to-point noise from a band-free stretch, falling back to the
    whole array; robust (MAD) so a peak edge does not inflate it."""
    for lo, hi in ((1750, 1950), (2200, 2400), (1400, 1500), (2500, 2560)):
        s = (wn > lo) & (wn < hi)
        if s.sum() >= 15:
            d = np.diff(y[s])
            return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    d = np.diff(y)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


# ---------------------------------------------------------------------------
# Lorentzian fitting
# ---------------------------------------------------------------------------

def _model(x, xc, *p):
    y = p[0] * (x - xc) + p[1]
    for k in range(2, len(p), 3):
        x0, w, h = p[k:k + 3]
        y = y + h * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)
    return y


def _jac(x, xc, *p):
    """Analytic Jacobian of _model: columns slope, offset, then per peak
    d/dx0, d/dw, d/dh."""
    J = np.empty((x.size, len(p)))
    J[:, 0] = x - xc
    J[:, 1] = 1.0
    for k in range(2, len(p), 3):
        x0, w, h = p[k:k + 3]
        g2 = (w / 2) ** 2
        d = x - x0
        den = d ** 2 + g2
        L = g2 / den
        J[:, k] = h * 2 * d * g2 / den ** 2
        J[:, k + 1] = h * (w / 2) * d ** 2 / den ** 2
        J[:, k + 2] = L
    return J


def _fit(x, y, starts, wmin, wmax, xc):
    """starts: [(x0, w, h), ...]. Returns (params, cov, rss) or None."""
    lo, hi = x.min(), x.max()
    p0, lb, ub = [0.0, 0.0], [-np.inf, -np.inf], [np.inf, np.inf]
    for x0, w, h in starts:
        p0 += [float(np.clip(x0, lo + 0.1, hi - 0.1)), float(np.clip(w, wmin * 1.01, wmax * 0.99)),
               max(float(h), 1e-9)]
        lb += [lo, wmin, 0.0]
        ub += [hi, wmax, np.inf]
    f = lambda xx, *p: _model(xx, xc, *p)
    p = None
    for maxfev in (3000, 30000):          # one longer retry before giving up
        try:
            p, cov = curve_fit(f, x, y, p0=p0, bounds=(lb, ub), maxfev=maxfev,
                               x_scale="jac", jac=lambda xx, *pp: _jac(xx, xc, *pp))
            break
        except Exception:
            continue
    if p is None:
        return None
    rss = float(np.sum((y - f(x, *p)) ** 2))
    at_bound = any(np.isclose(p[i], lb[i], rtol=0, atol=1e-6 * max(1, abs(lb[i])))
                   or np.isclose(p[i], ub[i], rtol=0, atol=1e-6 * max(1, abs(ub[i])))
                   for i in range(2, len(p)) if np.isfinite(lb[i]) and np.isfinite(ub[i]))
    return p, cov, rss, at_bound


def fit_peaks(x, y, n="auto", nmax=4, resolution=None, wmax=150.0, noise=None,
              starts=None):
    """Lorentzians on a residual linear baseline. y should already have its
    background removed (a small leftover slope/offset is fitted anyway).

    n       int for a fixed number of peaks, or 'auto' (1..nmax, see module doc)
    starts  optional [(x0, w, h), ...] starting values for a fixed n
    Returns dict: ok, n, comps [{pos, fwhm, height, area, *_err}], r2, bic,
    tried {k: bic}, why, curve (x, y, fit), components [arrays].
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = {"ok": False, "n": 0, "comps": [], "tried": {}, "why": ""}
    if x.size < 12:
        out["why"] = "range not covered"
        return out
    step = float(np.median(np.diff(x)))
    wmin = max(resolution or 0.0, 2 * step)
    if noise is None:
        d = np.diff(y)
        noise = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) or 1.0
    if y.max() < MIN_SNR_PRESENT * noise:          # nothing to fit: skip the work
        out["why"] = "no peak above %g x noise" % MIN_SNR_PRESENT
        return out
    xc = float(x.mean())
    N = x.size

    def bic(rss, k):
        return N * np.log(max(rss, 1e-12) / N) + k * np.log(N)

    def describe(res):
        p, cov, rss, _ = res
        err = np.sqrt(np.clip(np.diag(cov), 0, np.inf)) if np.all(np.isfinite(cov)) \
            else np.full(len(p), np.nan)
        comps = []
        for k in range(2, len(p), 3):
            x0, w, h = p[k:k + 3]
            ex, ew, eh = err[k:k + 3]
            comps.append(dict(pos=float(x0), fwhm=float(w), height=float(h),
                              area=float(np.pi * h * w / 2), pos_err=float(ex),
                              fwhm_err=float(ew), height_err=float(eh),
                              area_err=float(np.pi / 2 * np.hypot(h * ew, w * eh))))
        comps.sort(key=lambda c: c["pos"])
        return comps

    def acceptable(res, comps):
        if res[3]:
            return "a parameter sits on its bound"
        if min(c["height"] for c in comps) < MIN_SNR * noise:
            return "a peak is below %g x noise" % MIN_SNR
        tot = sum(c["area"] for c in comps) or 1.0
        if min(c["area"] for c in comps) < MIN_AREA_FRAC * tot:
            return "a peak carries < %g%% of the band area" % (100 * MIN_AREA_FRAC)
        for a, b in zip(comps[:-1], comps[1:]):
            if b["pos"] - a["pos"] < MIN_SEP_FRAC * 0.5 * (a["fwhm"] + b["fwhm"]):
                return "two peaks closer than %.1f x their mean FWHM" % MIN_SEP_FRAC
        return ""

    if n != "auto":
        n = int(n)
        if starts is not None:
            # starting heights from the data at each starting centre
            starts = [(c, w, max(float(np.interp(c, x, y)), noise)) for c, w, _ in starts]
        if starts is None:
            # spread starting centres over the strongest part of the range
            top = x[y >= y.max() / 3] if y.max() > 0 else x
            c0 = np.linspace(top.min(), top.max(), n + 2)[1:-1] if n > 1 else [x[np.argmax(y)]]
            starts = [(c, 20.0 if n == 1 else 12.0, max(y.max(), noise) / n) for c in c0]
        res = _fit(x, y, starts, wmin, wmax, xc)
        if res is None:
            out["why"] = "fit failed"
            return out
        best, best_n = res, n
        out["tried"][n] = bic(res[2], 2 + 3 * n)
    else:
        best, best_n = None, 0
        prev_starts = []
        top = x[y >= y.max() / 5] if y.max() > 0 else x
        for k in range(1, nmax + 1):
            # several starting layouts: the previous peaks plus one at the
            # largest residual, and k peaks spread evenly over the band
            layouts = []
            if best is None:
                layouts.append([(x[np.argmax(y)], 20.0, max(y.max(), noise))])
            else:
                resid = y - _model(x, xc, *best[0])
                j = int(np.argmax(resid))
                layouts.append(prev_starts + [(x[j], max(wmin * 2, 8.0), max(resid[j], noise))])
                cs = np.linspace(top.min(), top.max(), k + 2)[1:-1]
                w0 = max(wmin * 1.5, np.ptp(top) / (k + 1))
                layouts.append([(c, w0, max(float(np.interp(c, x, y)), noise)) for c in cs])
            cand = None
            for lay in layouts:
                res = _fit(x, y, lay, wmin, wmax, xc)
                if res is None:
                    continue
                comps = describe(res)
                why = acceptable(res, comps) if k > 1 else ""
                if why:
                    out.setdefault("rejected", {})[k] = why
                    continue
                if cand is None or res[2] < cand[0][2]:
                    cand = (res, comps)
            if cand is None:
                out["stopped"] = out.get("rejected", {}).get(k, "fit failed")
                break
            res, comps = cand
            out["tried"][k] = bic(res[2], 2 + 3 * k)
            if best is not None and out["tried"][k] > out["tried"][best_n] - MIN_DBIC:
                out["stopped"] = "BIC did not improve by %g" % MIN_DBIC
                break
            best, best_n = res, k
            prev_starts = [(c["pos"], c["fwhm"], c["height"]) for c in comps]
        if best is None:
            out["why"] = "fit failed"
            return out

    if max(c["height"] for c in describe(best)) < MIN_SNR_PRESENT * noise:
        out["why"] = "no peak above %g x noise" % MIN_SNR_PRESENT
        out["tried"] = {}
        return out
    p = best[0]
    fit = _model(x, xc, *p)
    r2 = 1 - best[2] / np.sum((y - y.mean()) ** 2)
    comps = describe(best)
    out.update(ok=True, n=best_n, comps=comps, r2=float(r2), bic=out["tried"][best_n],
               noise=noise, wmin=wmin, curve=(x, y, fit),
               baseline=p[0] * (x - xc) + p[1],
               components=[c["height"] * (c["fwhm"] / 2) ** 2 /
                           ((x - c["pos"]) ** 2 + (c["fwhm"] / 2) ** 2) for c in comps])
    return out


def analyse_band(wn, y, lo, hi, n="auto", nmax=4, bg="linear", bg_opts=None,
                 resolution=None, noise=None, starts=None):
    """Background-subtract [lo, hi] and fit it. Returns the fit_peaks dict
    plus 'x', 'raw', 'bg'."""
    s = (wn >= lo) & (wn <= hi)
    x, v = wn[s], y[s]
    if x.size < 12:
        return {"ok": False, "n": 0, "comps": [], "why": "range not covered", "tried": {}}
    b = background(x, v, bg, **(bg_opts or {}))
    r = fit_peaks(x, v - b, n=n, nmax=nmax, resolution=resolution, noise=noise,
                  starts=starts)
    r.update(x=x, raw=v, bg=b)
    return r
