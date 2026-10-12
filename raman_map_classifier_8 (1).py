#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raman Map Classifier
====================

Two-tab PyQt5 application for mapping the 2D-band FWHM of multilayer graphene
Raman maps and classifying each pixel as ABA (Bernal) or ABC (rhombohedral)
stacking.

Tab 1 - Map Explorer
    Load a map file, fit a single Voigt profile to the 2D band of every
    spectrum, display FWHM(2D) as a colour map, toggle a pixel grid, click a
    pixel to inspect its raw spectrum and fit, and co-register / blend an
    optical (camera) image on top.

Tab 2 - Classification
    Assign each pixel a layer number (manual or auto), classify it as ABA or
    ABC using the 2D-band width, the 2D-band low-frequency asymmetry, or both
    (AND logic), with a fully editable threshold/colour table. Export the
    classified map as PNG and the per-pixel results as CSV.

Reference thresholds: Bouhafs et al., Carbon 177 (2021) 282-290, Fig. 2c.
NOTE: those defaults were read off a published figure. They are starting
values, not calibrated constants - validate them against your own reference
samples and edit the table before trusting a classification.

Dependencies
------------
    pip install PyQt5 numpy scipy matplotlib lmfit

lmfit is optional but recommended; a scipy.optimize.curve_fit fallback is used
if it is not installed.

Usage
-----
    python raman_map_classifier.py [map_file.txt]
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.special import voigt_profile

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.colors as mcolors
# Vector exports keep text as real, editable text rather than outlines, so
# labels can be changed in Inkscape or Illustrator before going to Overleaf.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
# Arial first (on every Mac, and what journals usually ask for), then
# Helvetica; Liberation Sans is Arial's metric twin on Linux. The maths text
# (cm⁻¹, µm superscripts) must use the same family, or every label containing
# it would still carry matplotlib's own font into Illustrator.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans",
                                          "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "sans"
matplotlib.rcParams["mathtext.it"] = "sans:italic"
matplotlib.rcParams["mathtext.bf"] = "sans:bold"
import logging as _logging
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def side_colorbar(fig, ax, mappable, size="4%", pad=0.08):
    """A colour bar glued to the map itself. fig.colorbar(ax=...) places it
    beside the axes' full box, which for an equal-aspect map in a wide panel
    is far from the map."""
    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax)

from PyQt5 import QtCore, QtGui, QtWidgets

# Lorentzian G/2D fits and the 1L/2L reference + prediction live next to
# this script; the tab that uses them is simply disabled if they are missing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import raman_peak_fit as rpf
    import graphene_reference as gref
    import raman_tools as rt
    import strain_doping as sd
    HAVE_GREF = True
except Exception:                                    # pragma: no cover
    HAVE_GREF = False

#: np.trapezoid is NumPy >= 2.0; np.trapz is the 1.x spelling.
_trapz = getattr(np, "trapezoid", None) or np.trapz

try:
    from lmfit.models import LinearModel, VoigtModel
    HAVE_LMFIT = True
except Exception:                                    # pragma: no cover
    HAVE_LMFIT = False


# ---------------------------------------------------------------------------
# Constants and default configuration
# ---------------------------------------------------------------------------

APP_NAME = "Raman Map Classifier 8"

#: Bouhafs et al., Carbon 177 (2021) 282-290, Fig. 2c - FWHM(2D) in cm-1.
BOUHAFS_FWHM: Dict[int, Dict[str, float]] = {
    3: {"ABA": 45.0, "ABC": 65.0},
    4: {"ABA": 50.0, "ABC": 70.0},
    5: {"ABA": 55.0, "ABC": 75.0},
    6: {"ABA": 58.0, "ABC": 80.0},
    7: {"ABA": 60.0, "ABC": 85.0},
    8: {"ABA": 63.0, "ABC": 88.0},
    9: {"ABA": 65.0, "ABC": 90.0},
}

#: Half-width of the acceptance window around each reference FWHM, in cm-1.
#: A measured width has to land within DEFAULT_TOL of a tabulated value to be
#: called that stacking at all. Ten is a starting point, not a constant.
DEFAULT_TOL = 10.0

DEFAULT_COLORS: Dict[int, Dict[str, str]] = {
    3: {"ABA": "#0000ff", "ABC": "#008000"},   # blue    / green
    4: {"ABA": "#00ffff", "ABC": "#00ff00"},   # cyan    / lime
    5: {"ABA": "#ffa500", "ABC": "#ff0000"},   # orange  / red
    6: {"ABA": "#800080", "ABC": "#ffc0cb"},   # purple  / pink
    7: {"ABA": "#a52a2a", "ABC": "#ffff00"},   # brown   / yellow
    8: {"ABA": "#808080", "ABC": "#ffffff"},   # grey    / white
    9: {"ABA": "#000000", "ABC": "#ff00ff"},   # black   / magenta
}

LAYERS = sorted(BOUHAFS_FWHM.keys())


def default_config() -> dict:
    """Every tunable lives here so edits survive a re-run without reloading."""
    return {
        # Per-layer FWHM(2D) thresholds, acceptance windows and colours
        "fwhm": {k: dict(v) for k, v in BOUHAFS_FWHM.items()},
        "tol": {k: {"ABA": DEFAULT_TOL, "ABC": DEFAULT_TOL} for k in BOUHAFS_FWHM},
        "overlap_policy": "nearest",   # "nearest" | "unclassified"
        "colors": {k: dict(v) for k, v in DEFAULT_COLORS.items()},
        "unclassified_color": "#e8e8e8",
        "unclassified_alpha": 0.35,   # let the optical image read through
        # Fitting
        "band_min": 2600.0,        # cm-1, crop before Voigt fit
        "band_max": 2800.0,        # cm-1
        "engine": "scipy",         # "scipy" (~12x faster) | "lmfit"
        "free_gamma": True,        # let Lorentzian width float independently
        "min_snr": 5.0,            # fitted peak height / residual noise
        "max_fwhm": 120.0,         # cm-1, reject wider "peaks" as junk fits
        "min_fwhm": 5.0,           # cm-1, set from the instrument resolution
        "despike": True,           # remove cosmic rays before fitting
        "despike_sigma": 6.0,      # MAD multiples flagged as a spike
        "smooth_window": 9,        # Savitzky-Golay points for shape metrics
        "parallel": True,          # fit across CPU cores for large maps
        # Asymmetry
        "asym_metric": "hwhm",     # "hwhm" | "area" | "residual"
        "asym_threshold": 0.15,
        "asym_span": 1.5,          # window half-width, in units of the FWHM
        # Classification
        "layer_mode": "auto",      # "auto" | "manual"
        "manual_layer": 3,
        "technique": "both",       # "fwhm" | "asym" | "both"
        # Auto mode = thickness from I(2D)/I(G). Needs both bands in the
        # spectral window, i.e. a wide (600 gr/mm) acquisition.
        "g_band_min": 1500.0,
        "g_band_max": 1680.0,
        "ratio_metric": "height",      # "height" | "area"
        # Descending cutoffs on I(2D)/I(G): above 1_2 -> 1L, above 2_3 -> 2L,
        # above 3_thick -> 3L, otherwise >=4L. Placeholders from the typical
        # literature values (1L ~4, 2L ~2, 3L ~1.1, thick ~0.3); calibrate.
        "ratio_cut": {"1_2": 3.0, "2_3": 1.5, "3_thick": 0.8},
        "thick_policy": "leave",       # "leave" | "assume"
        "thick_assume": 5,             # N used for >=4L pixels if "assume"
        # How the map's two coordinate columns become X/Y in the camera
        # frame. None = take it from the #AxisType headers. The headers have
        # proved unreliable on this instrument, so Auto-align exists.
        "map_transform": None,
        # "instrument": X = −col0, Y = col1 — verified on this LabRAM against
        # LabSpec's own overlay (004_01_8) and independently by Auto-align
        # (R5G_7TRY2). "headers": follow the #AxisType labels, which disagree.
        "orientation_default": "instrument",
        "align_all_orientations": False,     # Auto-align: lock orientation, search offset
        "align_search_um": 15.0,
        "y_down": True,
        "zoom_to_map": True,                 # frame maps on the Raman grid, not the camera                      # draw maps as LabSpec does: +Y downward
        # Layer map: a layer count per region, from AFM / optical contrast.
        "exclude_edges": True,         # drop pixels straddling a thickness step
        "contrast_levels": 2,
        "contrast_cut": 0.03,          # |optical contrast| below this = substrate
        "sio2_offset": 0.5,            # nm, AFM tip/water-layer offset on SiO2
        "ratio_colors": {"1": "#26a69a", "2": "#9575cd", "3": "#ffb300",
                         "thick": "#546e7a"},
        # (restrict_range / range_margin retired: the tolerance windows in
        #  "tol" now decide what counts as in range)
        # Display
        "cmap": "viridis",
        "optical_alpha": 0.5,
    }


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _to_float(token: str) -> float:
    """Tolerate European decimal commas as well as plain floats."""
    token = token.strip()
    try:
        return float(token)
    except ValueError:
        return float(token.replace(",", "."))


def _split(line: str) -> List[str]:
    parts = line.split("\t")
    if len([p for p in parts if p.strip()]) < 3:
        parts = line.split()
    return parts


def parse_header_meta(header: List[str]) -> dict:
    """Pull the LabSpec-style ``#Key=<tab>value`` metadata out of the header."""
    meta = {}
    for line in header:
        body = line.lstrip("#").strip()
        if "=" not in body:
            continue
        k, v = body.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def axis_roles(header: List[str]) -> Dict[int, str]:
    """Read the ``#AxisType[n]`` declarations.

    These files list axes FASTEST-VARYING FIRST after AxisType[0] (the
    intensity channel):

        AxisType[1]  the axis running along row 0
        AxisType[2]  the fastest-varying leading column
        AxisType[3]  the next slower leading column, and so on

    A Raman map declares Intens / Spectr / X / Y, so the spectral axis runs
    along the row, the *second* leading column is X and the *first* is Y. A
    camera file declares '' / X / Y: X runs along the row and the single
    leading column is Y. Both readings agree, which is what makes the
    convention trustworthy.
    """
    roles = {}
    for line in header:
        body = line.lstrip("#").strip()
        if not body.startswith("AxisType[") or "]" not in body:
            continue
        try:
            idx = int(body[len("AxisType["):body.index("]")])
        except ValueError:
            continue
        val = body.split("=", 1)[1].strip() if "=" in body else ""
        roles[idx] = val
    return roles


@dataclass
class MapFile:
    """A parsed export, in either of the two layouts these files use.

    layout "spectra"  one spectrum per row: two leading coordinate columns
                      followed by the intensities, with the wavenumbers on
                      row 0. This is the Raman map.

    layout "image"    a plain 2D image: row 0 carries the fast spatial axis
                      and every later row starts with its own slow-axis
                      coordinate followed by one value per column. This is
                      the camera file, and it is NOT the same layout as the
                      Raman map despite what the spec assumes.
    """

    axis: np.ndarray               # (n_axis,) wavenumbers, or X positions
    x: np.ndarray                  # (n_points,) X in micrometres
    y: np.ndarray                  # (n_points,) Y in micrometres
    data: np.ndarray               # (n_points, n_axis) intensities
    path: str = ""
    header: List[str] = field(default_factory=list)
    layout: str = "spectra"
    image: Optional[np.ndarray] = None      # (ny, nx), image layout only
    swapped: bool = False                   # X/Y taken from the headers
    raw0: Optional[np.ndarray] = None       # coordinate column 0, as in the file
    raw1: Optional[np.ndarray] = None       # coordinate column 1, as in the file

    ux: np.ndarray = field(init=False)
    uy: np.ndarray = field(init=False)
    index_map: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.layout == "image":
            self.ux = np.asarray(self.axis, dtype=float)
            self.uy = np.asarray(self.y, dtype=float)
            self.index_map = np.zeros((self.uy.size, self.ux.size), dtype=int)
            return
        xr = np.round(self.x, 4)
        yr = np.round(self.y, 4)
        self.ux = np.unique(xr)
        self.uy = np.unique(yr)
        idx = np.full((len(self.uy), len(self.ux)), -1, dtype=int)
        cols = np.searchsorted(self.ux, xr)
        rows = np.searchsorted(self.uy, yr)
        for i, (r, c) in enumerate(zip(rows, cols)):
            idx[r, c] = i
        self.index_map = idx

    # -- convenience --------------------------------------------------------
    @property
    def meta(self) -> dict:
        return parse_header_meta(self.header)

    @property
    def n_points(self) -> int:
        return 0 if self.data is None else int(self.data.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.index_map.shape

    @property
    def is_optical(self) -> bool:
        return self.layout == "image" or any("camera" in h.lower()
                                             for h in self.header)

    @property
    def spectral_resolution(self) -> Optional[float]:
        for k, v in self.meta.items():
            if k.lower().startswith("spectral res"):
                try:
                    return float(v.replace(",", "."))
                except ValueError:
                    return None
        return None

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(left, right, bottom, top) in micrometres, pixel-edge aligned."""
        dx = float(np.median(np.diff(self.ux))) if len(self.ux) > 1 else 1.0
        dy = float(np.median(np.diff(self.uy))) if len(self.uy) > 1 else 1.0
        return (float(self.ux[0] - dx / 2), float(self.ux[-1] + dx / 2),
                float(self.uy[0] - dy / 2), float(self.uy[-1] + dy / 2))

    def spectrum(self, row: int, col: int) -> Optional[np.ndarray]:
        if self.layout != "spectra":
            return None
        i = self.index_map[row, col]
        return None if i < 0 else self.data[i]

    def greyscale(self) -> np.ndarray:
        """A (ny, nx) image: the camera frame, or the mean over channels."""
        if self.layout == "image":
            return self.image
        vals = np.nanmean(self.data, axis=1)
        img = np.full(self.shape, np.nan)
        ny, nx = self.shape
        for r in range(ny):
            for c in range(nx):
                i = self.index_map[r, c]
                if i >= 0:
                    img[r, c] = vals[i]
        return img


def _read_lines(path: str) -> Tuple[List[str], List[str]]:
    header, body = [], []
    with open(path, "r", encoding="ISO-8859-1", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            (header if line.lstrip().startswith("#") else body).append(
                line.strip() if line.lstrip().startswith("#") else line)
    return header, body


def parse_map_file(path: str) -> MapFile:
    """Parse either export layout, detected from the axis row.

    The number of blank fields before the first number on the axis row is the
    number of leading coordinate columns each data row carries: two for a
    Raman map (X and Y per spectrum), one for a camera image (the slow axis
    coordinate per row). Guessing wrong silently transposes the whole map, so
    this is measured rather than assumed.
    """
    header, body = _read_lines(path)
    if len(body) < 2:
        raise ValueError("File contains an axis row but no data rows.")

    tabbed = "\t" in body[0]
    def fields(line):
        return line.split("\t") if tabbed else line.split()

    axis_parts = fields(body[0])
    axis_vals = [t for t in axis_parts if t.strip()]
    axis = np.array([_to_float(t) for t in axis_vals], dtype=float)
    if axis.size < 1:
        raise ValueError("Could not read an axis from the first data row.")

    # How many coordinate columns precede the values on each data row? Two
    # for a Raman map (X and Y per spectrum), one for a camera image. Count
    # the blank leading fields where the file is tab-delimited, but confirm
    # it against the width difference between a data row and the axis row,
    # which also works for whitespace-delimited exports where consecutive
    # blank columns collapse and cannot be counted.
    n_blank = 0
    while n_blank < len(axis_parts) and not axis_parts[n_blank].strip():
        n_blank += 1
    n_lead = None
    for line in body[1:]:
        width = len([t for t in fields(line) if t.strip()])
        if width > axis.size:
            n_lead = width - axis.size
            break
    if n_lead not in (1, 2):
        n_lead = n_blank if n_blank in (1, 2) else 2

    roles = axis_roles(header)

    if n_lead == 1:
        return _parse_image(path, header, body, axis, roles)
    return _parse_spectra(path, header, body, axis, roles)


def _parse_spectra(path, header, body, axis, roles) -> MapFile:
    n_axis = axis.size
    xs, ys, vals = [], [], []
    for parts in (_split(line) for line in body[1:]):
        if len(parts) < 3:
            continue
        try:
            a = _to_float(parts[0])
            b = _to_float(parts[1])
        except ValueError:
            continue
        inten = []
        for t in parts[2:]:
            if not t.strip():
                continue
            try:
                inten.append(_to_float(t))
            except ValueError:
                inten.append(np.nan)
        if inten:
            xs.append(a)
            ys.append(b)
            vals.append(inten)
    if not vals:
        raise ValueError("No data rows could be parsed.")

    width = min(n_axis, min(len(v) for v in vals))
    axis = axis[:width]
    data = np.array([v[:width] for v in vals], dtype=float)
    col0 = np.array(xs)
    col1 = np.array(ys)

    # Which leading column is X? The headers list axes fastest-first, so
    # AxisType[2] names the fast column (col1) and AxisType[3] the slow one
    # (col0). Absent headers, fall back to the spec's col0=X, col1=Y.
    def _axis_kind(label: str) -> str:
        """'X'/'XA'/'X_ABS' etc. all mean the X axis; same for Y. Matched by
        prefix, not exact equality, so an instrument-specific suffix (this
        file's 'XA'/'YA' for absolute stage position, as opposed to the
        earlier file's plain 'X'/'Y') is still recognised instead of quietly
        falling through to a guessed default."""
        u = label.strip().upper()
        if u.startswith("X"):
            return "X"
        if u.startswith("Y"):
            return "Y"
        return ""

    role2, role3 = _axis_kind(roles.get(2, "")), _axis_kind(roles.get(3, ""))
    swapped = False
    if role2 == "X" and role3 == "Y":
        x, y = col1, col0
        swapped = True
    elif role2 == "Y" and role3 == "X":
        x, y = col0, col1
    elif role2 or role3:
        # One is recognisable but they do not form a clean X/Y pair - safer
        # to say so than to guess.
        raise ValueError(
            "Cannot determine the X/Y column order: AxisType[2]=%r, "
            "AxisType[3]=%r. Expected one to start with X and the other "
            "with Y." % (roles.get(2, ""), roles.get(3, "")))
    else:
        # No AxisType header at all - only place we fall back to the
        # spec's stated order, and it is worth knowing that happened.
        x, y = col0, col1

    if axis.size > 1 and axis[0] > axis[-1]:
        order = np.argsort(axis)
        axis = axis[order]
        data = data[:, order]

    return MapFile(axis=axis, x=x, y=y, data=data, path=path, header=header,
                   raw0=col0.copy(), raw1=col1.copy(),
                   layout="spectra", swapped=swapped)


def _parse_image(path, header, body, axis, roles) -> MapFile:
    """Row 0 is the fast spatial axis; each later row is <coord> then values."""
    n_cols = axis.size
    ys = np.empty(len(body) - 1, dtype=float)
    img = np.empty((len(body) - 1, n_cols), dtype=np.float32)
    keep = 0
    for line in body[1:]:
        v = np.fromstring(line, sep="\t")
        if v.size < 2:
            v = np.fromstring(line, sep=" ")
        if v.size < 2:
            continue
        ys[keep] = v[0]
        row = v[1:]
        if row.size < n_cols:
            img[keep, :row.size] = row
            img[keep, row.size:] = np.nan
        else:
            img[keep] = row[:n_cols]
        keep += 1
    if keep == 0:
        raise ValueError("No image rows could be parsed.")
    ys = ys[:keep]
    img = img[:keep]

    xs = axis
    if xs.size > 1 and xs[0] > xs[-1]:
        order = np.argsort(xs)
        xs = xs[order]
        img = img[:, order]
    if ys.size > 1 and ys[0] > ys[-1]:          # camera rows run top-to-bottom
        ys = ys[::-1]
        img = img[::-1]

    # The row axis is X and the leading column Y unless the headers say
    # otherwise; transpose rather than mislabel if they do.
    if roles.get(1, "").upper() == "Y" and roles.get(2, "").upper() == "X":
        xs, ys, img = ys, xs, img.T

    return MapFile(axis=xs, x=xs, y=ys, data=np.empty((0, 0)), path=path,
                   header=header, layout="image", image=img)


# ---------------------------------------------------------------------------
# Single spectra and one-file-per-spot map folders
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """One spectrum that is not part of a map: a single-spot LabSpec export,
    a LabSpec .h5 spectrum, or a spectrum taken out of a loaded map."""
    wn: np.ndarray                 # ascending wavenumbers
    inten: np.ndarray
    label: str
    path: str = ""
    meta: dict = field(default_factory=dict)


def is_single_spectrum(path: str) -> bool:
    """True for a two-column (wavenumber, intensity) text export. A map's
    first data row is the wavenumber axis and has far more than two fields,
    so this never matches one."""
    if path.lower().endswith(".h5"):
        return True
    try:
        _, body = _read_lines(path)
    except OSError:
        return False
    rows = body[:50]
    return len(body) > 10 and all(
        len([t for t in _split(r) if t.strip()]) == 2 for r in rows)


def read_single_spectrum(path: str) -> Spectrum:
    if path.lower().endswith(".h5"):
        try:
            import h5py
        except ImportError:
            raise ValueError("Reading .h5 files needs h5py:  pip install h5py")
        with h5py.File(path, "r") as h:
            if "Datas/Data1" not in h:
                raise ValueError("No Datas/Data1 in this file; only LabSpec "
                                 ".h5 exports are supported.")
            ds = h["Datas/Data1"]
            if ds.ndim != 1:
                raise ValueError("This .h5 holds a %d-D dataset; only single "
                                 "spectra are supported." % ds.ndim)
            wn = np.asarray(ds.attrs["Axis1"], dtype=float)
            inten = ds[:].astype(float)
            meta = {}
            if "Datas/DataInfo1" in h:
                for r in h["Datas/DataInfo1"][:]:
                    meta[r[0].decode(errors="replace")] = r[2].decode(errors="replace")
    else:
        header, body = _read_lines(path)
        a = np.array([[_to_float(t) for t in _split(r) if t.strip()][:2]
                      for r in body], dtype=float)
        wn, inten = a[:, 0], a[:, 1]
        meta = parse_header_meta(header)
    o = np.argsort(wn)
    return Spectrum(wn=wn[o], inten=inten[o], label=short_label(path),
                    path=path, meta=meta)


def short_label(path: str) -> str:
    """A compact default label: the first three '_' parts of the file or
    folder name, with the parent folder added when that is very short
    (e.g. '007_01/5')."""
    path = path.rstrip(os.sep)
    stem = os.path.splitext(os.path.basename(path))[0]
    lab = "_".join(stem.split("_")[:3])[:28]
    if len(lab) <= 4:
        lab = os.path.basename(os.path.dirname(path)) + "/" + lab
    return lab


_SPOT_RE = re.compile(r"__X_([-+\d.]+)__Y_([-+\d.]+)__")


def read_spot_folder(folder: str) -> MapFile:
    """A map exported as one two-column file per spot, the stage position in
    each filename (``..._<n>__X_<x>__Y_<y>__Time_<t>.txt``).

    raw0 holds −X so that the instrument default (X = −col0, Y = col1)
    places the spots exactly at their filename coordinates.
    """
    xs, ys, rows, axis = [], [], [], None
    for name in sorted(os.listdir(folder)):
        m = _SPOT_RE.search(name)
        if not m or not name.lower().endswith((".txt", ".dat", ".csv")):
            continue
        a = np.loadtxt(os.path.join(folder, name), comments="#")
        a = a[np.argsort(a[:, 0])]
        if axis is None:
            axis = a[:, 0]
        elif a.shape[0] != axis.size or not np.allclose(a[:, 0], axis, atol=1e-3):
            raise ValueError("%s has a different wavenumber axis from the "
                             "other spots." % name)
        xs.append(float(m.group(1)))
        ys.append(float(m.group(2)))
        rows.append(a[:, 1])
    if not rows:
        raise ValueError("No files named like ..._X_<x>__Y_<y>__... in this folder.")
    x, y = np.array(xs), np.array(ys)
    return MapFile(axis=axis, x=x, y=y, data=np.array(rows), path=folder,
                   header=[], raw0=-x, raw1=y.copy(), layout="spectra")


def band_height(wn: np.ndarray, D: np.ndarray, centre: float,
                half: float) -> np.ndarray:
    """Peak height above the flanking baseline, no fitting: the maximum
    within ±half of centre minus the median of the flanks 2–4 half-widths
    away. D is (n_spectra, n_axis)."""
    pk = (wn > centre - half) & (wn < centre + half)
    fl = (((wn > centre - 4 * half) & (wn < centre - 2 * half))
          | ((wn > centre + 2 * half) & (wn < centre + 4 * half)))
    if pk.sum() < 2 or fl.sum() < 4:
        return np.full(D.shape[0], np.nan)
    return np.nanmax(D[:, pk], axis=1) - np.nanmedian(D[:, fl], axis=1)


def kmeans_1d(v: np.ndarray, k: int, iters: int = 200) -> np.ndarray:
    """Sorted cluster centres for 1-D data (Lloyd's algorithm)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    k = max(1, min(int(k), v.size))
    if v.size == 0:
        return np.array([])
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        nc = np.array([v[lab == j].mean() if (lab == j).any() else c[j]
                       for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return np.sort(c)


def optical_contrast(mf: MapFile, opt: MapFile) -> Tuple[np.ndarray, float]:
    """Optical contrast (I - I_substrate) / I_substrate under each spectrum.

    The camera image is averaged over each Raman pixel's footprint rather
    than sampled at its centre, so a pixel is described by the same area the
    map step implies. The substrate level is the mode of the whole frame,
    which is bare substrate in nearly every flake image.
    """
    img = opt.image.astype(float)
    px = (opt.extent[1] - opt.extent[0]) / img.shape[1]
    img = gaussian_filter(median_filter(img, size=3), max(1.0, 0.15 / px))
    hist, edges = np.histogram(img[np.isfinite(img)], bins=256)
    hist = gaussian_filter(hist.astype(float), 2.0)
    j = int(np.argmax(hist))
    sub = 0.5 * (edges[j] + edges[j + 1])
    x0, x1, y0, y1 = opt.extent
    H, W = img.shape
    dx = float(np.median(np.diff(mf.ux))) if len(mf.ux) > 1 else 1.0
    dy = float(np.median(np.diff(mf.uy))) if len(mf.uy) > 1 else 1.0
    out = np.full(mf.n_points, np.nan)
    for i in range(mf.n_points):
        c0 = int((mf.x[i] - dx / 2 - x0) / (x1 - x0) * W)
        c1 = int((mf.x[i] + dx / 2 - x0) / (x1 - x0) * W)
        r0 = int((mf.y[i] - dy / 2 - y0) / (y1 - y0) * H)
        r1 = int((mf.y[i] + dy / 2 - y0) / (y1 - y0) * H)
        c0, c1 = max(c0, 0), min(max(c1, c0 + 1), W)
        r0, r1 = max(r0, 0), min(max(r1, r0 + 1), H)
        if c0 < W and r0 < H and c1 > c0 and r1 > r0:
            out[i] = (np.nanmean(img[r0:r1, c0:c1]) - sub) / sub
    return out, sub


def frame_axes(ax, mf, cfg: dict, with_optical: bool):
    """Frame a map view on the Raman grid. With the camera image underneath
    the axes would otherwise stretch to the whole camera frame and shrink the
    map to a patch in the middle; a small margin keeps the flake's
    surroundings visible."""
    if mf is None:
        return
    left, right, bottom, top = mf.extent
    if with_optical and not cfg.get("zoom_to_map", True):
        return
    pad = 0.08 * max(right - left, top - bottom) if with_optical else 0.0
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(bottom - pad, top + pad)


def orient_axes(ax, cfg: dict):
    """Draw +Y downward when asked, matching LabSpec's display. Display only:
    data coordinates and mouse clicks are unaffected."""
    down = bool(cfg.get("y_down", True))
    if down != ax.yaxis_inverted():
        ax.invert_yaxis()


INSTRUMENT_TRANSFORM = {"swap": False, "flip_x": True, "flip_y": False,
                        "dx": 0.0, "dy": 0.0}


def default_transform(mf: MapFile, cfg: dict) -> dict:
    """Placement used before any alignment: the instrument convention, or the
    header reading if the user prefers it."""
    if cfg.get("orientation_default", "instrument") == "headers":
        return header_transform(mf)
    return dict(INSTRUMENT_TRANSFORM)


def header_transform(mf: MapFile) -> dict:
    """The orientation the #AxisType headers imply, as an explicit transform."""
    return {"swap": bool(mf.swapped), "flip_x": False, "flip_y": False,
            "dx": 0.0, "dy": 0.0}


def describe_transform(t: dict) -> str:
    a, b = ("col1", "col0") if t["swap"] else ("col0", "col1")
    def term(sign, name, off):
        s = ("−" if sign else "") + name
        return s + (" %+.1f µm" % off if abs(off) > 1e-9 else "")
    return "X = %s,  Y = %s" % (term(t["flip_x"], a, t["dx"]),
                                term(t["flip_y"], b, t["dy"]))


def apply_map_transform(mf: MapFile, t: Optional[dict]) -> MapFile:
    """Rebuild X/Y from the raw coordinate columns.

    Only coordinates change; spectra stay in file order, so fits made before
    a transform stay valid and nothing needs refitting.
    """
    if t is None or mf.layout != "spectra" or mf.raw0 is None:
        return mf
    a, b = (mf.raw1, mf.raw0) if t["swap"] else (mf.raw0, mf.raw1)
    x = (-a if t["flip_x"] else a) + float(t["dx"])
    y = (-b if t["flip_y"] else b) + float(t["dy"])
    return MapFile(axis=mf.axis, x=x, y=y, data=mf.data, path=mf.path,
                   header=mf.header, layout="spectra", swapped=bool(t["swap"]),
                   raw0=mf.raw0, raw1=mf.raw1)


def band_presence(mf: MapFile, cfg: dict) -> Tuple[np.ndarray, str]:
    """A per-spectrum 'graphene is here' signal straight from the raw data.

    Peak height above the flanking baseline, using the G band when the file
    has one and the 2D band otherwise. Deliberately independent of whether a
    fit was accepted, so weak pixels still contribute to the alignment.
    """
    wn, D = mf.axis, mf.data
    use_g = int(((wn > 1560) & (wn < 1605)).sum()) >= 3
    if use_g:
        c, half, name = 1582.0, 22.0, "G band"
    else:
        c, half, name = 0.5 * (cfg["band_min"] + cfg["band_max"]), 45.0, "2D band"
    pk = (wn > c - half) & (wn < c + half)
    fl = ((wn > c - 4 * half) & (wn < c - 2 * half)) | ((wn > c + 2 * half) & (wn < c + 4 * half))
    if pk.sum() < 2 or fl.sum() < 4:
        return np.zeros(D.shape[0]), name
    return np.nanmax(D[:, pk], axis=1) - np.nanmedian(D[:, fl], axis=1), name


def auto_align(mf: MapFile, opt: MapFile, cfg: dict, search_um: float = 15.0,
               orientations=None, sign: Optional[int] = None):
    """Find the orientation and offset that put graphene where the camera sees it.

    Every combination of swap / flip-X / flip-Y is tried, each with a search
    over translations; the score is the correlation between the Raman band
    height and the (smoothed, dust-clipped) optical brightness at each
    spectrum's position. Absolute correlation is used because a flake can look
    brighter or darker than its substrate depending on oxide and lighting.

    Returns (candidates sorted best-first, signal name). Each candidate is
    (|r|, r, transform).
    """
    sig, name = band_presence(mf, cfg)
    if mf.raw0 is None or not np.isfinite(sig).any() or np.nanstd(sig) == 0:
        return [], name
    img = opt.image.astype(float)
    px = (opt.extent[1] - opt.extent[0]) / img.shape[1]
    img = gaussian_filter(median_filter(img, size=3), max(1.0, 0.25 / px))
    img = np.minimum(img, np.nanpercentile(img, 97))      # dust is not graphene
    x0, x1, y0, y1 = opt.extent
    H, W = img.shape
    sig = np.nan_to_num(sig)
    if sig.size > 4000:                                   # plenty for a correlation
        keep = np.linspace(0, sig.size - 1, 4000).astype(int)
    else:
        keep = np.arange(sig.size)
    s_ = sig[keep]

    def score(X, Y):
        ix = ((X - x0) / (x1 - x0) * W).astype(int)
        iy = ((Y - y0) / (y1 - y0) * H).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if ok.sum() < max(20, 0.5 * ok.size):
            return np.nan
        a, b = s_[ok], img[iy[ok], ix[ok]]
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    out = []
    r0, r1 = mf.raw0[keep], mf.raw1[keep]
    if orientations is None:
        orientations = [(sw, a, b) for sw in (False, True)
                        for a in (False, True) for b in (False, True)]
    for swap, fx, fy in orientations:
        if True:
            if True:
                a, b = (r1, r0) if swap else (r0, r1)
                A, B = (-a if fx else a), (-b if fy else b)
                best = (0.0, 0.0, 0.0, 0.0)
                grid = np.arange(-search_um, search_um + 1e-9, 1.0)
                for dx in grid:                            # coarse ...
                    for dy in grid:
                        r = score(A + dx, B + dy)
                        if np.isfinite(r) and (abs(r) if sign is None else sign * r) > best[0]:
                            best = ((abs(r) if sign is None else sign * r), r, dx, dy)
                fine = np.arange(-1.0, 1.0 + 1e-9, 0.25)
                _, _, cx, cy = best
                for dx in cx + fine:                       # ... then fine
                    for dy in cy + fine:
                        r = score(A + dx, B + dy)
                        if np.isfinite(r) and (abs(r) if sign is None else sign * r) > best[0]:
                            best = ((abs(r) if sign is None else sign * r), r, dx, dy)
                out.append((best[0], best[1],
                            {"swap": swap, "flip_x": fx, "flip_y": fy,
                             "dx": round(best[2], 2), "dy": round(best[3], 2)}))
    out.sort(key=lambda t: -t[0])
    return out, name


# ---------------------------------------------------------------------------
# 2D-band analysis
# ---------------------------------------------------------------------------

def voigt_fwhm(sigma: float, gamma: float) -> float:
    """Olivero & Longbothum pseudo-analytic Voigt FWHM (accurate to ~0.02%).

    f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2), with f_L = 2*gamma and
    f_G = 2*sigma*sqrt(2 ln2).
    """
    fl = 2.0 * abs(gamma)
    fg = 2.0 * abs(sigma) * np.sqrt(2.0 * np.log(2.0))
    return 0.5346 * fl + np.sqrt(0.2166 * fl * fl + fg * fg)


def _voigt_plus_line(x, amplitude, center, sigma, gamma, slope, intercept):
    return amplitude * voigt_profile(x - center, sigma, gamma) + slope * x + intercept


def despike(y: np.ndarray, sigma: float = 6.0, width: int = 5):
    """Replace cosmic-ray spikes with the local median.

    Returns (cleaned, n_removed). A spike is a point sitting more than
    ``sigma`` robust deviations above a running median. The 2D band spans
    hundreds of channels at these dispersions, so a 5-point median cannot
    touch it, while a cosmic ray is a handful of channels wide and would
    otherwise be fitted as an implausibly narrow, very high-SNR "band".
    """
    if y.size < width * 3:
        return y, 0
    med = median_filter(y, size=width, mode="nearest")
    resid = y - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    if mad <= 0:
        return y, 0
    bad = resid > sigma * 1.4826 * mad
    if not bad.any():
        return y, 0
    out = y.copy()
    out[bad] = med[bad]
    return out, int(bad.sum())


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing, used only for shape metrics - never for fitting."""
    w = int(window) | 1
    if w < 5 or y.size <= w:
        return y
    try:
        return savgol_filter(y, w, 3)
    except Exception:
        return y


def _initial_guess(x: np.ndarray, y: np.ndarray, smooth_window: int = 9) -> dict:
    ys = _smooth(y, smooth_window)
    n_edge = max(3, len(x) // 10)
    xe = np.concatenate([x[:n_edge], x[-n_edge:]])
    ye = np.concatenate([ys[:n_edge], ys[-n_edge:]])
    slope, intercept = np.polyfit(xe, ye, 1)
    yc = ys - (slope * x + intercept)

    i = int(np.nanargmax(yc))
    center = float(x[i])
    height = float(yc[i])
    if not np.isfinite(height) or height <= 0:
        height = float(np.nanmax(np.abs(yc))) or 1.0

    above = np.where(yc >= height / 2.0)[0]
    fw = abs(float(x[above[-1]] - x[above[0]])) if above.size >= 2 else 30.0
    fw = float(np.clip(fw, 2.0, max(4.0, x[-1] - x[0])))

    sigma = fw / 2.3548 * 0.7
    gamma = fw / 2.0 * 0.5
    peak = voigt_profile(0.0, sigma, gamma)
    amplitude = height / peak if peak > 0 else height * fw
    return dict(amplitude=amplitude, center=center, sigma=sigma, gamma=gamma,
                slope=float(slope), intercept=float(intercept), height=height)


def _asymmetry_metrics(x, y, fit, fwhm, cfg):
    """Low-frequency asymmetry of the 2D band; returns (hwhm, area, residual).

    All three are anchored at the maximum of the smoothed band, NOT at the
    fitted Voigt centre. That distinction matters: a single Voigt fitted to a
    band carrying a low-frequency shoulder drags its centroid into the
    shoulder, which re-balances the two lobes and hides the very asymmetry
    being measured. Anchoring at the maximum keeps the reference where the
    dominant component sits.

    hwhm     - (HWHM_low - HWHM_high) / (HWHM_low + HWHM_high), the classic
               peak-asymmetry ratio. Least sensitive to baseline and to the
               far wings.
    area     - normalised difference of the two lobe integrals out to
               asym_span x FWHM either side of the maximum.
    residual - what the single Voigt fails to describe on the low-frequency
               side, as a fraction of the fitted peak area.

    All are positive when there is excess weight on the low-frequency side
    (~2680 cm-1 for a band near 2700 cm-1), the ABC signature, and all are
    dimensionless.
    """
    nan3 = (np.nan, np.nan, np.nan)
    ys = _smooth(y, cfg["smooth_window"])
    base = float(np.percentile(ys, 5))
    yb = ys - base
    i = int(np.argmax(yb))
    peak = float(yb[i])
    if peak <= 0 or x.size < 10:
        return nan3
    xc = float(x[i])

    # --- HWHM ratio, by linear interpolation of the half-maximum crossings ---
    hw = np.nan
    half = peak / 2.0
    j = i
    while j > 0 and yb[j] > half:
        j -= 1
    lo_x = np.interp(half, [yb[j], yb[j + 1]], [x[j], x[j + 1]]) if j < i else np.nan
    j = i
    while j < x.size - 1 and yb[j] > half:
        j += 1
    hi_x = np.interp(half, [yb[j], yb[j - 1]], [x[j], x[j - 1]]) if j > i else np.nan
    left_w, right_w = xc - lo_x, hi_x - xc
    if np.isfinite(left_w) and np.isfinite(right_w) and (left_w + right_w) > 0:
        hw = float((left_w - right_w) / (left_w + right_w))

    # --- lobe areas about the maximum ---
    span = float(np.clip(cfg["asym_span"] * fwhm, 5.0, (x[-1] - x[0]) / 2.0))
    lmask = (x >= xc - span) & (x < xc)
    rmask = (x > xc) & (x <= xc + span)
    ar = np.nan
    if lmask.sum() >= 3 and rmask.sum() >= 3:
        yc = np.clip(yb, 0.0, None)
        a_l = float(_trapz(yc[lmask], x[lmask]))
        a_r = float(_trapz(yc[rmask], x[rmask]))
        if a_l + a_r > 0:
            ar = float((a_l - a_r) / (a_l + a_r))

    # --- unmodelled low-frequency shoulder ---
    rs = np.nan
    if fit is not None and lmask.sum() >= 3:
        resid = y - fit
        peak_area = float(_trapz(np.clip(fit - np.min(fit), 0, None), x))
        if peak_area > 0:
            shoulder = float(_trapz(np.clip(resid[lmask], 0, None), x[lmask]))
            rs = float(shoulder / peak_area)

    return hw, ar, rs


def fwhm_uncertainty(sigma: float, gamma: float, cov: np.ndarray) -> float:
    """1-sigma error on the Voigt FWHM, propagated from the fit covariance.

    Worth having, not decoration: the ABA/ABC boundary is only ~20 cm-1 from
    either threshold, so an FWHM carrying a +/-10 cm-1 error cannot support a
    stacking call no matter how confident the colour map looks.
    """
    if cov is None or not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return np.nan
    h = 1e-6 * max(abs(sigma), abs(gamma), 1.0)
    d_s = (voigt_fwhm(sigma + h, gamma) - voigt_fwhm(sigma - h, gamma)) / (2 * h)
    d_g = (voigt_fwhm(sigma, gamma + h) - voigt_fwhm(sigma, gamma - h)) / (2 * h)
    var = (d_s * d_s * cov[0, 0] + d_g * d_g * cov[1, 1]
           + 2.0 * d_s * d_g * cov[0, 1])
    return float(np.sqrt(var)) if var > 0 else np.nan


def _empty_result() -> dict:
    return dict(ok=False, fwhm=np.nan, center=np.nan, height=np.nan, snr=np.nan,
                amplitude=np.nan, sigma=np.nan, gamma=np.nan,
                slope=np.nan, intercept=np.nan, asym_hwhm=np.nan,
                asym_area=np.nan, asym_residual=np.nan, spikes=0,
                fwhm_err=np.nan, note="", why="")


def fit_band(x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    """Fit one Voigt + linear baseline to an already-cropped band region.

    Returns a dict carrying ``ok``, the FWHM, the fit parameters (so the curve
    can be rebuilt without re-fitting) and all three asymmetry metrics.
    """
    out = _empty_result()
    good = np.isfinite(y)
    if good.sum() < 10:
        out["note"] = "too few valid points"
        out["why"] = "too few valid points in the window"
        return out
    x = x[good].astype(float)
    y = y[good].astype(float)

    if np.all(y == 0):
        out["note"] = "empty spectrum (all zero)"
        out["why"] = "empty spectrum (all zero)"
        return out
    if np.ptp(y) == 0:
        out["note"] = "flat spectrum (no signal)"
        out["why"] = "flat spectrum (no signal)"
        return out

    if cfg.get("despike", True):
        y, n_spikes = despike(y, cfg.get("despike_sigma", 6.0))
        out["spikes"] = n_spikes

    g = _initial_guess(x, y, cfg["smooth_window"])
    span = float(x[-1] - x[0])
    params = None
    cov = None

    if cfg.get("engine", "scipy") == "lmfit" and HAVE_LMFIT:
        try:
            model = VoigtModel(prefix="v_") + LinearModel(prefix="b_")
            p = model.make_params(v_amplitude=g["amplitude"], v_center=g["center"],
                                  v_sigma=g["sigma"], b_slope=g["slope"],
                                  b_intercept=g["intercept"])
            p["v_center"].set(min=x[0], max=x[-1])
            p["v_sigma"].set(min=1e-3, max=span)
            p["v_amplitude"].set(min=0)
            # VoigtModel ties gamma to sigma by default; free it for real bands.
            p["v_gamma"].set(value=g["gamma"], vary=bool(cfg["free_gamma"]),
                             expr="" if cfg["free_gamma"] else None,
                             min=0, max=span)
            res = model.fit(y, p, x=x, nan_policy="omit")
            if res.covar is not None:
                names = list(res.var_names)
                if "v_sigma" in names and "v_gamma" in names:
                    i, j = names.index("v_sigma"), names.index("v_gamma")
                    cov = np.array([[res.covar[i, i], res.covar[i, j]],
                                    [res.covar[j, i], res.covar[j, j]]])
            params = dict(amplitude=res.params["v_amplitude"].value,
                          center=res.params["v_center"].value,
                          sigma=res.params["v_sigma"].value,
                          gamma=res.params["v_gamma"].value,
                          slope=res.params["b_slope"].value,
                          intercept=res.params["b_intercept"].value)
        except Exception:
            params = None

    if params is None:
        try:
            p0 = [g["amplitude"], g["center"], g["sigma"], g["gamma"],
                  g["slope"], g["intercept"]]
            lower = [0.0, x[0], 1e-3, 0.0, -np.inf, -np.inf]
            upper = [np.inf, x[-1], span, span, np.inf, np.inf]
            p0 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(p0, lower, upper)]
            if not cfg["free_gamma"]:
                # Tie gamma to sigma by fitting a reduced model.
                def _tied(xx, amp, cen, sig, sl, ic):
                    return _voigt_plus_line(xx, amp, cen, sig, sig, sl, ic)
                popt, pcov = curve_fit(_tied, x, y, p0=[p0[0], p0[1], p0[2], p0[4], p0[5]],
                                    bounds=([0, x[0], 1e-3, -np.inf, -np.inf],
                                            [np.inf, x[-1], span, np.inf, np.inf]),
                                    maxfev=20000)
                popt = [popt[0], popt[1], popt[2], popt[2], popt[3], popt[4]]
                cov = np.array([[pcov[2, 2], pcov[2, 2]],
                                [pcov[2, 2], pcov[2, 2]]])
            else:
                popt, pcov = curve_fit(_voigt_plus_line, x, y, p0=p0,
                                       bounds=(lower, upper), maxfev=20000)
                cov = pcov[np.ix_([2, 3], [2, 3])]
            params = dict(zip(["amplitude", "center", "sigma", "gamma",
                               "slope", "intercept"], map(float, popt)))
        except Exception as exc:
            out["note"] = "fit failed (%s)" % type(exc).__name__
            out["why"] = "the fit did not converge"
            return out

    fwhm = voigt_fwhm(params["sigma"], params["gamma"])
    fit = _voigt_plus_line(x, **params)
    baseline = params["slope"] * x + params["intercept"]
    height = float(np.max(fit - baseline))
    # Noise from the fit residual, not the band edges: a broad fit to pure
    # noise then scores an SNR near 1 instead of sneaking past the filter.
    noise = float(np.std(y - fit))
    snr = height / noise if noise > 0 else np.inf

    out.update(params)
    out["fwhm"] = float(fwhm)
    out["center"] = float(params["center"])
    out["height"] = height
    out["snr"] = float(snr)
    out["fwhm_err"] = fwhm_uncertainty(params["sigma"], params["gamma"], cov)

    if not np.isfinite(fwhm) or fwhm <= 1.0:
        out["note"] = "degenerate FWHM"
        out["why"] = "degenerate FWHM"
        return out
    if fwhm < cfg.get("min_fwhm", 5.0):
        # Narrower than the instrument can resolve, so it is an artefact
        # (usually a surviving cosmic ray), not a phonon band.
        out["note"] = ("FWHM %.1f below the %.1f cm-1 resolution limit"
                       % (fwhm, cfg.get("min_fwhm", 5.0)))
        out["why"] = ("narrower than the %.1f cm⁻¹ resolution limit — a spike, "
                      "not a band" % cfg.get("min_fwhm", 5.0))
        return out
    if fwhm > cfg["max_fwhm"]:
        out["note"] = "FWHM %.0f above max %.0f" % (fwhm, cfg["max_fwhm"])
        out["why"] = "wider than the %.0f cm⁻¹ limit" % cfg["max_fwhm"]
        return out
    if snr < cfg["min_snr"]:
        out["note"] = "SNR %.1f below %.1f" % (snr, cfg["min_snr"])
        out["why"] = "SNR below %.1f — no measurable band" % cfg["min_snr"]
        return out
    if not (x[0] < params["center"] < x[-1]):
        out["note"] = "centre pinned at the window edge"
        out["why"] = "fitted centre pinned at the window edge"
        return out

    hw, ar, rs = _asymmetry_metrics(x, y, fit, fwhm, cfg)
    out["asym_hwhm"], out["asym_area"], out["asym_residual"] = hw, ar, rs
    out["ok"] = True
    return out


def band_mask(wn: np.ndarray, cfg: dict) -> np.ndarray:
    return (wn >= cfg["band_min"]) & (wn <= cfg["band_max"])


def fit_2d_band(wn: np.ndarray, inten: np.ndarray, cfg: dict) -> dict:
    """Crop to the 2D-band window, then fit. Convenience wrapper."""
    m = band_mask(wn, cfg)
    if m.sum() < 10:
        out = _empty_result()
        out["note"] = "fewer than 10 points in %.0f-%.0f cm-1" % (
            cfg["band_min"], cfg["band_max"])
        out["why"] = "fewer than 10 points inside the band window"
        return out
    return fit_band(wn[m], inten[m], cfg)


def _fit_chunk(args):
    """Module-level so it can be shipped to a worker process."""
    x, block, cfg = args
    return [fit_band(x, block[i], cfg) for i in range(block.shape[0])]


def rebuild_fit(x: np.ndarray, res: dict) -> np.ndarray:
    return _voigt_plus_line(x, res["amplitude"], res["center"], res["sigma"],
                            res["gamma"], res["slope"], res["intercept"])


def asym_of(res: dict, metric: str) -> float:
    return res.get({"hwhm": "asym_hwhm", "area": "asym_area",
                    "residual": "asym_residual"}.get(metric, "asym_hwhm"), np.nan)



# ---------------------------------------------------------------------------
# Multi-Lorentzian fitting (Spectra tab)
# ---------------------------------------------------------------------------

def _lorentz_sum(x, *p):
    """Sum of n Lorentzians (height, centre, FWHM) plus a linear baseline."""
    n = (len(p) - 2) // 3
    y = p[-2] * (x - x.mean()) + p[-1]
    for k in range(n):
        h, c, w = p[3 * k:3 * k + 3]
        y = y + h * (0.5 * w) ** 2 / ((x - c) ** 2 + (0.5 * w) ** 2)
    return y


def fit_lorentzians(x: np.ndarray, y: np.ndarray, n: int = 1) -> dict:
    """Fit n Lorentzians plus a linear baseline over the given range.

    Returns ok, per-peak centre / FWHM / height / area with 1-sigma errors,
    the FWHM of the summed band (what a single width means when a band has
    several components), R², and the fitted curve. Cosmic rays are removed
    first. Note: a Lorentzian describes the phonon line itself; at low
    spectral resolution the instrument adds Gaussian broadening, so widths
    from a 600 gr/mm spectrum are upper bounds.
    """
    out = {"ok": False, "n": n, "peaks": [], "band_fwhm": np.nan, "r2": np.nan,
           "curve": None, "note": ""}
    good = np.isfinite(y)
    x, y = x[good].astype(float), y[good].astype(float)
    if x.size < 3 * n + 6:
        out["note"] = "too few points in range"
        return out
    y, _ = despike(y)
    edge = max(3, x.size // 12)
    xe = np.r_[x[:edge], x[-edge:]]
    ye = np.r_[y[:edge], y[-edge:]]
    sl, ic = np.polyfit(xe - x.mean(), ye, 1)
    base = sl * (x - x.mean()) + ic
    r = _smooth(y - base, 9)
    span = x[-1] - x[0]
    step = float(np.median(np.abs(np.diff(x))))
    # Starting guesses: the n most prominent peaks, padded if there are fewer.
    pk, pr = find_peaks(r, prominence=max(1e-9, 0.05 * np.nanmax(r)))
    order = np.argsort(-pr["prominences"])[:n] if pk.size else []
    cens = sorted(float(x[pk[i]]) for i in order)
    top = float(x[int(np.argmax(r))])
    while len(cens) < n:
        cens.append(top + (len(cens) - (n - 1) / 2.0) * 0.15 * span / max(n, 1))
    cens = sorted(cens)
    hmax = float(np.nanmax(r))
    above = x[r >= 0.5 * hmax]
    w0 = max(3 * step, (above.max() - above.min()) / n if above.size > 1 else 0.1 * span)
    p0, lo, hi = [], [], []
    for c in cens:
        hc = max(float(np.interp(c, x, r)), 0.2 * hmax)
        p0 += [hc, c, w0]
        lo += [0.0, x[0], max(step, 0.5)]
        hi += [np.inf, x[-1], span]
    p0 += [sl, ic]
    lo += [-np.inf, -np.inf]
    hi += [np.inf, np.inf]
    p0 = [float(np.clip(v, a, b)) for v, a, b in zip(p0, lo, hi)]
    try:
        popt, pcov = curve_fit(_lorentz_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=40000)
    except Exception as exc:
        out["note"] = "fit failed (%s)" % type(exc).__name__
        return out
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) \
        else np.full(len(popt), np.nan)
    fit = _lorentz_sum(x, *popt)
    ss = float(np.sum((y - fit) ** 2))
    st = float(np.sum((y - y.mean()) ** 2)) or 1.0
    peaks = []
    for k in range(n):
        h, c, w = popt[3 * k:3 * k + 3]
        eh, ec, ew = perr[3 * k:3 * k + 3]
        peaks.append({"center": float(c), "fwhm": float(w), "height": float(h),
                      "area": float(np.pi * h * w / 2.0), "err_center": float(ec),
                      "err_fwhm": float(ew)})
    peaks.sort(key=lambda d: d["center"])
    # FWHM of the summed band, baseline removed, on a fine grid.
    xf = np.linspace(x[0], x[-1], 4000)
    band = _lorentz_sum(xf, *(list(popt[:-2]) + [0.0, 0.0]))
    if band.max() > 0:
        above = xf[band >= 0.5 * band.max()]
        out["band_fwhm"] = float(above.max() - above.min()) if above.size > 1 else np.nan
    npts, k = x.size, 3 * n + 2
    # Bayesian information criterion: rewards fit quality, penalises extra
    # peaks. Adding peaks always raises R²; BIC only falls if they earn it.
    out["bic"] = float(npts * np.log(max(ss, 1e-12) / npts) + k * np.log(npts))
    out.update(ok=True, peaks=peaks, r2=1.0 - ss / st, curve=(x, fit),
               baseline=(float(popt[-2]), float(popt[-1]), float(x.mean())),
               components=[(p["height"], p["center"], p["fwhm"]) for p in peaks])
    return out

def suggest_peak_count(x: np.ndarray, y: np.ndarray, resolution: float,
                       nmax: int = 4):
    """Smallest peak count the data justify.

    Two rules, because either alone misleads: a fit whose components are
    narrower than the spectral resolution is fitting noise or shape mismatch
    and is rejected; among the rest, the fewest peaks within 10 BIC units of
    the best (strong evidence) wins. A Lorentzian is not quite the shape of an
    instrument-broadened line, so BIC on its own keeps asking for more peaks.
    """
    fits = {n: fit_lorentzians(x, y, n) for n in range(1, nmax + 1)}
    valid = {n: f for n, f in fits.items() if f["ok"] and
             all(p["fwhm"] >= resolution for p in f["peaks"])}
    if not valid:
        return 1, fits
    best = min(f["bic"] for f in valid.values())
    return min(n for n, f in valid.items() if f["bic"] <= best + 10.0), fits


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: Sentinel layer number for "four or more layers, ratio cannot say which".
LAYER_THICK = 99


def g_band_cfg(cfg: dict) -> dict:
    """Fit settings for the G band: same quality rules, its own window.

    Ceiling on width is lower than for the 2D band because a G band much
    wider than ~80 cm-1 is not a G band.
    """
    g = dict(cfg)
    g["band_min"] = cfg.get("g_band_min", 1500.0)
    g["band_max"] = cfg.get("g_band_max", 1680.0)
    g["max_fwhm"] = 80.0
    return g


def ratio_applicable(axis: np.ndarray, cfg: dict) -> Tuple[bool, str]:
    """Can this file support thickness from I(2D)/I(G)?

    Decided by spectral coverage, not by the grating named in the header:
    what matters is whether both bands were actually recorded. A high-
    resolution (1800 gr/mm) acquisition centred on the 2D band spans only a
    few hundred cm-1 and never reaches the G band.
    """
    if axis is None or axis.size < 10:
        return False, "no spectral axis"
    lo, hi = float(axis.min()), float(axis.max())
    g_ok = int(((axis >= cfg.get("g_band_min", 1500.0)) &
                (axis <= cfg.get("g_band_max", 1680.0))).sum()) >= 10
    d_ok = int(((axis >= cfg["band_min"]) & (axis <= cfg["band_max"])).sum()) >= 10
    if g_ok and d_ok:
        return True, ("G and 2D bands both inside this file's range "
                      "(%.0f–%.0f cm⁻¹)" % (lo, hi))
    missing = " and ".join(n for n, ok in (("G band", g_ok), ("2D band", d_ok))
                           if not ok)
    return False, ("this file spans %.0f–%.0f cm⁻¹ and contains no %s"
                   % (lo, hi, missing))


def layer_from_ratio(ratio: float, cfg: dict) -> Optional[int]:
    """1, 2, 3 or LAYER_THICK from I(2D)/I(G); None when there is no ratio.

    The ratio falls steeply from monolayer to trilayer and then flattens
    toward the bulk value, so above three layers it stops resolving N at all.
    That is why the last class is ">=4L" rather than 4L, 5L, ...
    """
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return None
    c = cfg["ratio_cut"]
    if ratio > c["1_2"]:
        return 1
    if ratio > c["2_3"]:
        return 2
    if ratio > c["3_thick"]:
        return 3
    return LAYER_THICK


def pixel_ratio(res_2d: Optional[dict], res_g: Optional[dict], metric: str) -> float:
    """I(2D)/I(G) from two accepted fits. Height = fitted peak height above
    baseline; area = the Voigt amplitude, which is the integrated area."""
    if not res_2d or not res_g or not res_2d.get("ok") or not res_g.get("ok"):
        return np.nan
    key = "amplitude" if metric == "area" else "height"
    a, b = res_2d.get(key, np.nan), res_g.get(key, np.nan)
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return np.nan
    return float(a / b)


def boundary_for(layer: int, cfg: dict) -> float:
    """Midpoint of the two reference widths. Still reported, but it is no
    longer what decides a pixel — see windows_for()."""
    d = cfg["fwhm"][layer]
    return 0.5 * (d["ABA"] + d["ABC"])


def windows_for(layer: int, cfg: dict) -> Tuple[Tuple[float, float],
                                                Tuple[float, float]]:
    """Acceptance windows ((ABA_lo, ABA_hi), (ABC_lo, ABC_hi)) in cm-1.

    A bare midpoint boundary forces every pixel to one side or the other, so a
    width of 20 or 200 cm-1 still comes back labelled with total confidence.
    Requiring the width to land within a tolerance of a tabulated value
    instead means a pixel that matches neither reference is reported as
    matching neither.
    """
    d = cfg["fwhm"][layer]
    t = cfg.get("tol", {}).get(layer, {"ABA": DEFAULT_TOL, "ABC": DEFAULT_TOL})
    ta = abs(float(t.get("ABA", DEFAULT_TOL)))
    tc = abs(float(t.get("ABC", DEFAULT_TOL)))
    return ((d["ABA"] - ta, d["ABA"] + ta), (d["ABC"] - tc, d["ABC"] + tc))


def width_call(fwhm: float, layer: int, cfg: dict) -> Tuple[Optional[str], str]:
    """Stacking from the 2D width alone. Returns (call, reason).

    Three outcomes rather than two:
      - inside exactly one window            -> that stacking
      - inside neither                       -> None, "outside both windows"
      - inside both (the windows overlap)    -> settled by overlap_policy

    Set each tolerance to half the gap between the two reference values and
    the windows meet exactly at the midpoint, which reproduces the old
    boundary rule everywhere except far from the table, where it now abstains.
    """
    if not np.isfinite(fwhm):
        return None, "no fit"
    (a0, a1), (c0, c1) = windows_for(layer, cfg)
    in_a = a0 <= fwhm <= a1
    in_c = c0 <= fwhm <= c1
    if in_a and not in_c:
        return "ABA", ""
    if in_c and not in_a:
        return "ABC", ""
    if in_a and in_c:
        if cfg.get("overlap_policy", "nearest") == "unclassified":
            return None, "in the ABA/ABC window overlap"
        d = cfg["fwhm"][layer]
        # Equidistant goes to ABC, matching the published rule that the
        # midpoint itself counts as the upper class.
        nearer = "ABA" if abs(fwhm - d["ABA"]) < abs(fwhm - d["ABC"]) else "ABC"
        return nearer, ""
    return None, "%.1f outside both %dL windows (%.1f-%.1f, %.1f-%.1f)" % (
        fwhm, layer, a0, a1, c0, c1)


def candidate_layers(fwhm: float, cfg: dict) -> List[int]:
    """Every layer with a window this width could belong to.

    The union of the ABA and ABC windows, not the span between them: with a
    tolerance narrower than half the gap there is a genuine dead zone in the
    middle that belongs to no layer.
    """
    if not np.isfinite(fwhm):
        return []
    out = []
    for layer in sorted(cfg["fwhm"]):
        (a0, a1), (c0, c1) = windows_for(layer, cfg)
        if (a0 <= fwhm <= a1) or (c0 <= fwhm <= c1):
            out.append(layer)
    return out


def auto_layer(fwhm: float, cfg: dict) -> Tuple[Optional[int], int]:
    """Assign a layer number from the 2D-band width alone.

    Returns (layer, n_candidates). Still under-determined, and widening the
    tolerances widens the overlap: the windows already overlap heavily at zero
    tolerance, so n_candidates is the number to watch. The tie-break follows
    cfg["auto_ambiguity"]: strict refuses to guess, nearest takes the closest
    reference value, lowest takes the smallest layer that fits.

    Where the layer count is known from AFM, optical contrast or a step edge,
    Manual mode remains the sounder choice.
    """
    cands = candidate_layers(fwhm, cfg)
    n = len(cands)
    if n == 0:
        return None, 0
    if n == 1:
        return cands[0], 1
    mode = cfg.get("auto_ambiguity", "nearest")
    if mode == "strict":
        return None, n
    if mode == "lowest":
        return cands[0], n

    def nearest_ref(layer):
        d = cfg["fwhm"][layer]
        return min(abs(fwhm - d["ABA"]), abs(fwhm - d["ABC"]))
    return min(cands, key=nearest_ref), n


def classify_pixel(fwhm: float, asym: float, cfg: dict, ratio: float = np.nan,
                   n_here: int = 0) -> Tuple[Optional[int], Optional[str], int, str]:
    """Return (layer, 'ABA'|'ABC'|'N/A'|None, n_candidate_layers, reason).

    'N/A' means the layer count is known but stacking is not defined for it:
    ABA versus ABC only exists from three layers up, and ">=4L" from the
    ratio carries no specific N to look up in the width table.
    """
    if not np.isfinite(fwhm):
        return None, None, 0, "no fit"

    if cfg["layer_mode"] == "manual":
        layer = int(cfg["manual_layer"])
        ncand = len(candidate_layers(fwhm, cfg))
        if layer not in cfg["fwhm"]:
            return None, None, ncand, "layer not in table"
    elif cfg["layer_mode"] == "map":
        ncand = 0
        if n_here <= 0:
            return None, None, 0, "no layer count assigned in the layer map"
        if n_here in (1, 2):
            return int(n_here), "N/A", 0, ""
        layer = int(n_here)
        if layer not in cfg["fwhm"]:
            return layer, None, 0, "%dL has no row in the width table" % layer
    else:
        ncand = 0
        n = layer_from_ratio(ratio, cfg)
        if n is None:
            return None, None, 0, "no I(2D)/I(G) (G band not fitted)"
        if n in (1, 2):
            return n, "N/A", 0, ""
        if n == LAYER_THICK:
            if cfg.get("thick_policy", "leave") != "assume":
                return LAYER_THICK, "N/A", 0, ""
            layer = int(cfg.get("thick_assume", 5))
            if layer not in cfg["fwhm"]:
                return LAYER_THICK, "N/A", 0, "assumed N not in table"
        else:
            layer = 3

    call_w, why_w = width_call(fwhm, layer, cfg)
    call_a = None
    if np.isfinite(asym):
        call_a = "ABC" if asym >= cfg["asym_threshold"] else "ABA"

    technique = cfg["technique"]
    if technique == "fwhm":
        return layer, call_w, ncand, why_w
    if technique == "asym":
        return layer, call_a, ncand, "" if call_a else "no asymmetry value"
    # "both": AND logic - the two techniques must agree
    if call_w is None:
        return layer, None, ncand, why_w
    if call_a is None:
        return layer, None, ncand, "no asymmetry value"
    if call_w != call_a:
        return layer, None, ncand, "width says %s, asymmetry says %s" % (call_w, call_a)
    return layer, call_w, ncand, ""


def terrace_edges(n_map: np.ndarray) -> np.ndarray:
    """Pixels with a 4-neighbour of a *different* assigned layer count.

    At a thickness step the laser spot straddles two terraces and records a
    sum of two 2D bands fitted as one. Its layer count is half wrong by
    construction, and when the two terraces differ enough in width the
    broadened sum can also be misread as ABC.
    Only steps between assigned layers count: at the flake's outer edge the
    spot's other half is bare substrate, which adds no 2D band at all.
    """
    edge = np.zeros(n_map.shape, dtype=bool)
    a = n_map
    for sl_a, sl_b in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                       ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))):
        x, y = a[sl_a], a[sl_b]
        diff = (x > 0) & (y > 0) & (x != y)
        edge[sl_a] |= diff
        edge[sl_b] |= diff
    return edge


def classify_map(fwhm_map: np.ndarray, asym_map: np.ndarray, cfg: dict,
                 ratio_map: Optional[np.ndarray] = None,
                 n_map: Optional[np.ndarray] = None):
    ny, nx = fwhm_map.shape
    if ratio_map is None:
        ratio_map = np.full((ny, nx), np.nan)
    if n_map is None:
        n_map = np.zeros((ny, nx), dtype=int)
    edge = (terrace_edges(n_map) if cfg["layer_mode"] == "map"
            and cfg.get("exclude_edges", True) else np.zeros((ny, nx), dtype=bool))
    layers = np.full((ny, nx), -1, dtype=int)
    ncands = np.zeros((ny, nx), dtype=int)
    calls = np.empty((ny, nx), dtype=object)
    reasons = np.empty((ny, nx), dtype=object)
    for r in range(ny):
        for c in range(nx):
            if edge[r, c] and np.isfinite(fwhm_map[r, c]):
                lay, call, nc, why = (int(n_map[r, c]), None, 0,
                                      "on a terrace edge (spot spans two thicknesses)")
            else:
                lay, call, nc, why = classify_pixel(fwhm_map[r, c], asym_map[r, c],
                                                    cfg, ratio_map[r, c],
                                                    int(n_map[r, c]))
            layers[r, c] = -1 if lay is None else lay
            calls[r, c] = call
            ncands[r, c] = nc
            reasons[r, c] = why
    return layers, calls, ncands, reasons


def label_of(layer: int, call: Optional[str]) -> str:
    if layer == LAYER_THICK:
        return "≥4L (thick)"
    if layer < 0 or call is None:
        return "Unclassified"
    if call == "N/A":
        return "%dL" % layer
    return "%dL-%s" % (layer, call)


# Neighbouring layer counts get strongly contrasting colours.
N_COLORS = {3: "#1e88e5", 4: "#fb8c00", 5: "#43a047", 6: "#8e24aa",
            7: "#e53935", 8: "#00acc1", 9: "#6d4c41"}


def n_color(n: int, cfg: dict) -> str:
    """One colour per layer count, for views of the layer map itself."""
    if n in (1, 2) or n == LAYER_THICK:
        return na_color(n, cfg)
    return N_COLORS.get(int(n), "#9e9e9e")


def na_color(layer: int, cfg: dict) -> str:
    rc = cfg.get("ratio_colors", {})
    key = "thick" if layer == LAYER_THICK else str(layer)
    return rc.get(key, "#607d8b")


def class_rgba(layers: np.ndarray, calls: np.ndarray, cfg: dict) -> np.ndarray:
    ny, nx = layers.shape
    rgba = np.zeros((ny, nx, 4), dtype=float)
    unc = np.array(mcolors.to_rgba(cfg["unclassified_color"]))
    unc[3] = float(cfg.get("unclassified_alpha", 1.0))
    for r in range(ny):
        for c in range(nx):
            lay, call = int(layers[r, c]), calls[r, c]
            if call == "N/A":
                rgba[r, c] = mcolors.to_rgba(na_color(lay, cfg))
            elif lay < 0 or call is None:
                rgba[r, c] = unc
            else:
                hexcol = cfg["colors"].get(lay, {}).get(call, "#000000")
                rgba[r, c] = mcolors.to_rgba(hexcol)
    return rgba


# ---------------------------------------------------------------------------
# Background fitting worker
# ---------------------------------------------------------------------------

class FitWorker(QtCore.QThread):
    """Fits every spectrum off the GUI thread, across cores where it pays."""

    progress = QtCore.pyqtSignal(int, str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    #: below this many spectra, process startup costs more than it saves
    PARALLEL_MIN = 1500
    CHUNK = 64

    def __init__(self, mapfile: MapFile, cfg: dict, parent=None):
        super().__init__(parent)
        self.mapfile = mapfile
        self.cfg = dict(cfg)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        """Fit the 2D band, then the G band if the file contains one.

        Emits {"twod": [...], "g": [...] or None}.
        """
        try:
            mf = self.mapfile
            m = band_mask(mf.axis, self.cfg)
            if m.sum() < 10:
                self.failed.emit(
                    "The axis of this file has fewer than 10 points between "
                    "%.0f and %.0f cm\u207b\u00b9."
                    % (self.cfg["band_min"], self.cfg["band_max"]))
                return
            twod = self._fit_all(mf.axis[m], mf.data[:, m], self.cfg, "2D band")
            if twod is None:
                self.failed.emit("Cancelled.")
                return

            g = None
            gcfg = g_band_cfg(self.cfg)
            gm = band_mask(mf.axis, gcfg)
            if gm.sum() >= 10:
                g = self._fit_all(mf.axis[gm], mf.data[:, gm], gcfg, "G band")
                if g is None:
                    self.failed.emit("Cancelled.")
                    return
            self.done.emit({"twod": twod, "g": g})
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))

    def _fit_all(self, x, block, cfg, label):
        n = block.shape[0]
        results = None
        if (cfg.get("parallel", True) and n >= self.PARALLEL_MIN
                and (os.cpu_count() or 1) > 1):
            results = self._run_parallel(x, block, n, cfg, label)
            if self._cancel:
                return None
        if results is None:
            results = self._run_serial(x, block, n, cfg, label)
        return results

    def _run_serial(self, x, block, n, cfg, label):
        results, step = [], max(1, n // 200)
        for i in range(n):
            if self._cancel:
                return None
            results.append(fit_band(x, block[i], cfg))
            if i % step == 0 or i == n - 1:
                self.progress.emit(int(100 * (i + 1) / n),
                                   "Fitting %s: spectrum %d / %d" % (label, i + 1, n))
        return results

    def _run_parallel(self, x, block, n, cfg, label):
        """Returns None on failure so the caller can fall back to serial."""
        chunks = [(x, block[i:i + self.CHUNK], cfg)
                  for i in range(0, n, self.CHUNK)]
        try:
            results = []
            with ProcessPoolExecutor() as pool:
                for part in pool.map(_fit_chunk, chunks):
                    if self._cancel:
                        return []
                    results.extend(part)
                    self.progress.emit(
                        int(100 * len(results) / n),
                        "Fitting %s: spectrum %d / %d (%d cores)"
                        % (label, len(results), n, os.cpu_count() or 1))
            return results if len(results) == n else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

def fit_to_screen(widget, want_w: int, want_h: int, frac_w=0.94, frac_h=0.90):
    """Open at the wanted size, but never larger than the screen it is on.

    A fixed size can push a window partly off a laptop screen: a 15-inch
    MacBook is about 1470 × 956 points at default scaling.
    """
    scr = widget.screen() if hasattr(widget, "screen") and widget.screen() else \
        QtWidgets.QApplication.primaryScreen()
    if scr is None:
        widget.resize(want_w, want_h)
        return
    g = scr.availableGeometry()
    widget.resize(min(want_w, int(g.width() * frac_w)), min(want_h, int(g.height() * frac_h)))


_EXPORT_DIR = {"path": ""}
EXPORT_FILTERS = ("Vector PDF (*.pdf);;Vector SVG (*.svg);;"
                  "PNG image, 600 dpi (*.png)")


def export_figure(fig, parent, suggested: str = "figure.pdf") -> Optional[str]:
    """Save a figure for publication.

    PDF and SVG are vector: axes, text, lines, grid, outlines and legends
    stay editable shapes and text. Pixel images inside a figure (the Raman map
    cells, the camera image) are embedded at 600 dpi, with no smoothing, so
    the map cells stay sharp-edged at any zoom.
    """
    start = os.path.join(_EXPORT_DIR["path"], os.path.basename(suggested)) \
        if _EXPORT_DIR["path"] else suggested
    path, flt = QtWidgets.QFileDialog.getSaveFileName(
        parent, "Export figure", start, EXPORT_FILTERS)
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".pdf", ".svg", ".png"):
        ext = ".svg" if "SVG" in flt else ".png" if "PNG" in flt else ".pdf"
        path += ext
    _EXPORT_DIR["path"] = os.path.dirname(path)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    return path


class CompactToolbar(NavigationToolbar):
    """Matplotlib's navigation toolbar, sized like a toolbar.

    On a Retina display Qt hands this toolbar pixmaps scaled by the device
    pixel ratio without shrinking the buttons to match, so the icon row ends
    up taller than the plot it controls. Pinning the icon size and the widget
    height keeps it to one slim strip on every display.
    """

    ICON_PX = 16

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.setIconSize(QtCore.QSize(self.ICON_PX, self.ICON_PX))
        self.setFixedHeight(self.ICON_PX + 12)
        self.setStyleSheet(
            "QToolBar { border: 0px; padding: 0px; spacing: 0px; }"
            "QToolButton { padding: 1px; margin: 0px; }")
        # The coordinate readout is set in a large default face on macOS.
        for lbl in self.findChildren(QtWidgets.QLabel):
            lbl.setStyleSheet("font-size: 10px; color: #666;")
        self.addSeparator()
        act = self.addAction("Export…")
        act.setToolTip("Save this figure as vector PDF or SVG (editable in Inkscape "
                       "or Illustrator, then upload to Overleaf), or 600-dpi PNG")
        act.triggered.connect(self._export)

    def _export(self):
        win = self.window()
        main = win if hasattr(win, "suggest_name") else getattr(win, "main", None)
        if main is None:
            par = win.parent()
            main = par.window() if par is not None else None
        stem = "figure.pdf"
        if main is not None and hasattr(main, "suggest_name"):
            stem = main.suggest_name("_figure.pdf")
        path = export_figure(self.canvas.figure, self, stem)
        if path and main is not None and hasattr(main, "status"):
            main.status("Exported %s" % path)


class Canvas(QtWidgets.QWidget):
    """A matplotlib figure plus navigation toolbar in one widget."""

    def __init__(self, parent=None, toolbar=True, figsize=(5, 4)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        if toolbar:
            self.toolbar = CompactToolbar(self.canvas, self)
            lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def draw(self):
        self.canvas.draw_idle()


class ThresholdTable(QtWidgets.QTableWidget):
    """Editable Bouhafs threshold + colour table. Writes straight into config."""

    changed = QtCore.pyqtSignal()

    COLS = ["Layer", "ABA FWHM", "ABA ±", "ABC FWHM", "ABC ±",
            "ABA window", "ABC window", "ABA colour", "ABC colour"]
    C_ABA, C_TA, C_ABC, C_TC, C_WA, C_WC, C_CA, C_CC = 1, 2, 3, 4, 5, 6, 7, 8

    def __init__(self, cfg: dict, parent=None):
        super().__init__(len(LAYERS), len(self.COLS), parent)
        self.cfg = cfg
        self._loading = False
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.setMinimumHeight(len(LAYERS) * 28 + 34)
        self.reload()
        self.itemChanged.connect(self._on_item_changed)
        self.cellClicked.connect(self._on_cell_clicked)

    # -- population ---------------------------------------------------------
    def reload(self):
        self._loading = True
        for r, layer in enumerate(LAYERS):
            d = self.cfg["fwhm"][layer]
            name = QtWidgets.QTableWidgetItem("%dL" % layer)
            name.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(r, 0, name)
            tol = self.cfg["tol"][layer]
            for c, val in ((self.C_ABA, d["ABA"]), (self.C_TA, tol["ABA"]),
                           (self.C_ABC, d["ABC"]), (self.C_TC, tol["ABC"])):
                it = QtWidgets.QTableWidgetItem("%.1f" % val)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.setItem(r, c, it)
            self._set_window_items(r, layer)
            for c, key in ((self.C_CA, "ABA"), (self.C_CC, "ABC")):
                self._set_color_item(r, c, self.cfg["colors"][layer][key])
        self._loading = False

    def _set_window_items(self, row: int, layer: int):
        """The two acceptance windows, and a warning when they collide."""
        (a0, a1), (c0, c1) = windows_for(layer, self.cfg)
        overlap = a1 >= c0
        gap = c0 - a1
        for col, (lo, hi) in ((self.C_WA, (a0, a1)), (self.C_WC, (c0, c1))):
            it = QtWidgets.QTableWidgetItem("%.1f – %.1f" % (lo, hi))
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            if overlap:
                it.setForeground(QtGui.QBrush(QtGui.QColor("#c62828")))
                it.setToolTip("The windows overlap by %.1f cm⁻¹; pixels in the "
                              "overlap are settled by the overlap policy."
                              % -gap)
            elif gap > 0.5:
                it.setForeground(QtGui.QBrush(QtGui.QColor("#8a6d00")))
                it.setToolTip("A %.1f cm⁻¹ dead zone sits between the windows; "
                              "widths landing there stay Unclassified." % gap)
            self.setItem(row, col, it)

    def _set_color_item(self, row: int, col: int, hexcol: str):
        it = QtWidgets.QTableWidgetItem(hexcol)
        it.setFlags(QtCore.Qt.ItemIsEnabled)
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        qc = QtGui.QColor(hexcol)
        it.setBackground(QtGui.QBrush(qc))
        lum = 0.299 * qc.red() + 0.587 * qc.green() + 0.114 * qc.blue()
        it.setForeground(QtGui.QBrush(QtGui.QColor("#000000" if lum > 140 else "#ffffff")))
        self.setItem(row, col, it)

    def set_all_tolerances(self, value: float):
        for layer in LAYERS:
            self.cfg["tol"][layer]["ABA"] = float(value)
            self.cfg["tol"][layer]["ABC"] = float(value)
        self.reload()
        self.changed.emit()

    def set_tolerances_to_half_gap(self):
        """Windows meet exactly at the midpoint: the old boundary rule, but
        bounded on the outside so far-off widths no longer get a label."""
        for layer in LAYERS:
            d = self.cfg["fwhm"][layer]
            half = abs(d["ABC"] - d["ABA"]) / 2.0
            self.cfg["tol"][layer]["ABA"] = half
            self.cfg["tol"][layer]["ABC"] = half
        self.reload()
        self.changed.emit()

    # -- interaction --------------------------------------------------------
    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        col = item.column()
        if self._loading or col not in (self.C_ABA, self.C_TA,
                                        self.C_ABC, self.C_TC):
            return
        layer = LAYERS[item.row()]
        is_tol = col in (self.C_TA, self.C_TC)
        key = "ABA" if col in (self.C_ABA, self.C_TA) else "ABC"
        store = self.cfg["tol"] if is_tol else self.cfg["fwhm"]
        try:
            value = float(item.text().replace(",", "."))
            # A zero tolerance is legitimate (accept only the exact value);
            # a negative one is not.
            if value < 0 or (value <= 0 and not is_tol):
                raise ValueError
        except ValueError:
            self._loading = True
            item.setText("%.1f" % store[layer][key])
            self._loading = False
            return
        store[layer][key] = value
        self._loading = True
        item.setText("%.1f" % value)
        self._set_window_items(item.row(), layer)
        self._loading = False
        self.changed.emit()

    def _on_cell_clicked(self, row: int, col: int):
        if col not in (self.C_CA, self.C_CC):
            return
        layer = LAYERS[row]
        key = "ABA" if col == self.C_CA else "ABC"
        current = QtGui.QColor(self.cfg["colors"][layer][key])
        chosen = QtWidgets.QColorDialog.getColor(
            current, self, "Colour for %dL-%s" % (layer, key))
        if chosen.isValid():
            self.cfg["colors"][layer][key] = chosen.name()
            self._loading = True
            self._set_color_item(row, col, chosen.name())
            self._loading = False
            self.changed.emit()


# ---------------------------------------------------------------------------
# Tab 1 - Map Explorer
# ---------------------------------------------------------------------------

class MapExplorerTab(QtWidgets.QWidget):

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.sel: Optional[Tuple[int, int]] = None
        self._spec: Optional[dict] = None     # last plotted spectrum, cached
        self._grid_artists: List = []
        self._build()

    # -- UI -----------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Raman map…")
        self.btn_load.clicked.connect(self.main.load_raman)
        self.btn_optical = QtWidgets.QPushButton("Load optical image…")
        self.btn_optical.clicked.connect(self.main.load_optical)
        self.chk_grid = QtWidgets.QCheckBox("Grid overlay")
        self.chk_grid.toggled.connect(self.refresh_map)
        self.chk_optical = QtWidgets.QCheckBox("Overlay optical")
        self.chk_optical.toggled.connect(self.refresh_map)
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(50)
        self.sld_alpha.setFixedWidth(120)
        self.sld_alpha.valueChanged.connect(self._alpha_changed)
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["viridis", "plasma", "inferno", "magma",
                                "cividis", "jet", "turbo", "coolwarm"])
        self.cmb_cmap.currentTextChanged.connect(self._cmap_changed)

        for w in (self.btn_load, self.btn_optical):
            bar.addWidget(w)
        bar.addSpacing(12)
        bar.addWidget(self.chk_grid)
        bar.addWidget(self.chk_optical)
        bar.addWidget(QtWidgets.QLabel("Opacity"))
        bar.addWidget(self.sld_alpha)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Map"))
        self.cmb_quantity = QtWidgets.QComboBox()
        self.cmb_quantity.addItems(["FWHM(2D)", "G height", "2D height"])
        self.cmb_quantity.setToolTip("G / 2D height: raw peak height above the "
                                     "flanking baseline, no fitting")
        self.cmb_quantity.currentIndexChanged.connect(self.refresh_map)
        bar.addWidget(self.cmb_quantity)
        bar.addWidget(QtWidgets.QLabel("Colour map"))
        bar.addWidget(self.cmb_cmap)
        bar.addStretch(1)
        root.addLayout(bar)

        # ---- map orientation relative to the camera image ----
        obar = QtWidgets.QHBoxLayout()
        obar.addWidget(QtWidgets.QLabel("Map orientation:"))
        self.chk_swap = QtWidgets.QCheckBox("Swap X/Y")
        self.chk_fx = QtWidgets.QCheckBox("Flip X")
        self.chk_fy = QtWidgets.QCheckBox("Flip Y")
        for w in (self.chk_swap, self.chk_fx, self.chk_fy):
            w.toggled.connect(self._transform_edited)
            obar.addWidget(w)
        obar.addSpacing(8)
        obar.addWidget(QtWidgets.QLabel("offset X"))
        self.spn_dx = QtWidgets.QDoubleSpinBox()
        obar.addWidget(self.spn_dx)
        obar.addWidget(QtWidgets.QLabel("Y"))
        self.spn_dy = QtWidgets.QDoubleSpinBox()
        obar.addWidget(self.spn_dy)
        for sp in (self.spn_dx, self.spn_dy):
            sp.setRange(-200.0, 200.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.5)
            sp.setSuffix(" µm")
            sp.setFixedWidth(92)
            sp.valueChanged.connect(self._transform_edited)
        obar.addSpacing(14)
        self.chk_zoom = QtWidgets.QCheckBox("Zoom to map")
        self.chk_zoom.setChecked(bool(self.main.cfg.get("zoom_to_map", True)))
        self.chk_zoom.setToolTip("Frame the maps on the Raman grid. Untick to see the "
                                 "whole camera frame around it.")
        self.chk_zoom.toggled.connect(self._zoom_toggled)
        obar.addWidget(self.chk_zoom)
        obar.addStretch(1)
        root.addLayout(obar)
        obar = QtWidgets.QHBoxLayout()          # second line: alignment controls
        obar.addSpacing(18)
        obar.addWidget(QtWidgets.QLabel("Flake appears"))
        self.cmb_contrast = QtWidgets.QComboBox()
        self.cmb_contrast.addItems(["auto", "brighter", "darker"])
        self.cmb_contrast.setToolTip(
            "How the flake looks against the substrate in the camera image. "
            "'auto' works either way but can confuse the flake with its "
            "180°-rotated look-alike; telling it removes that.")
        obar.addWidget(self.cmb_contrast)
        self.btn_align = QtWidgets.QPushButton("Auto-align to optical")
        self.btn_align.setToolTip(
            "Try every swap / flip combination, each with a search over "
            "offsets, and keep the one where the graphene band signal best "
            "matches the flake in the camera image.")
        self.btn_align.clicked.connect(self.main.run_auto_align)
        obar.addWidget(self.btn_align)
        self.chk_allorient = QtWidgets.QCheckBox("Search all orientations")
        self.chk_allorient.setToolTip(
            "Off (recommended): keep the orientation, search only the offset.\n"
            "On: try all eight swap/flip combinations — slower, and easily fooled "
            "by uneven illumination in the camera image.")
        obar.addWidget(self.chk_allorient)
        self.chk_ydown = QtWidgets.QCheckBox("Y down (as LabSpec)")
        self.chk_ydown.setChecked(bool(self.main.cfg.get("y_down", True)))
        self.chk_ydown.setToolTip("Display only: draw +Y pointing down, as LabSpec does, "
                                  "so maps look the same as on the instrument")
        self.chk_ydown.toggled.connect(self._ydown_toggled)
        obar.addWidget(self.chk_ydown)
        self.btn_hdr = QtWidgets.QPushButton("Instrument default")
        self.btn_hdr.setToolTip("X = −col0, Y = col1 with no offset — this "
                                "LabRAM's convention, verified against LabSpec")
        self.btn_hdr.clicked.connect(lambda: self.main.set_transform(None))
        obar.addWidget(self.btn_hdr)
        obar.addStretch(1)
        root.addLayout(obar)
        self.lbl_transform = QtWidgets.QLabel("")
        self.lbl_transform.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(self.lbl_transform)
        self._syncing = False

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.map_canvas = Canvas(figsize=(6, 5))
        self.map_canvas.canvas.mpl_connect("button_press_event", self._on_click)
        self.map_canvas.setMinimumHeight(240)

        spec_box = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(spec_box)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(2)

        sbar = QtWidgets.QHBoxLayout()
        sbar.setContentsMargins(4, 0, 4, 0)
        sbar.addWidget(QtWidgets.QLabel("View"))
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.addItems(["2D band", "Full spectrum", "Custom"])
        self.cmb_view.currentIndexChanged.connect(self._view_changed)
        sbar.addWidget(self.cmb_view)
        sbar.addSpacing(8)
        sbar.addWidget(QtWidgets.QLabel("from"))
        self.spn_x0 = QtWidgets.QDoubleSpinBox()
        self.spn_x1 = QtWidgets.QDoubleSpinBox()
        for sp, val in ((self.spn_x0, 2580.0), (self.spn_x1, 2820.0)):
            sp.setRange(0, 10000)
            sp.setDecimals(0)
            sp.setSingleStep(10)
            sp.setValue(val)
            sp.setFixedWidth(80)
            sp.valueChanged.connect(self._range_edited)
            sbar.addWidget(sp if sp is self.spn_x0 else QtWidgets.QLabel("to"))
            if sp is self.spn_x1:
                sbar.addWidget(sp)
        sbar.addWidget(QtWidgets.QLabel("cm⁻¹"))
        sbar.addSpacing(10)
        self.chk_yfit = QtWidgets.QCheckBox("Fit Y to view")
        self.chk_yfit.setChecked(True)
        self.chk_yfit.setToolTip(
            "Rescale the intensity axis to whatever is inside the wavenumber "
            "range, instead of to the whole spectrum. A weak 2D band sitting "
            "on a tall background is invisible without this.")
        self.chk_yfit.toggled.connect(self._apply_view)
        sbar.addWidget(self.chk_yfit)
        self.btn_reset_view = QtWidgets.QPushButton("Reset view")
        self.btn_reset_view.clicked.connect(self._reset_view)
        sbar.addWidget(self.btn_reset_view)
        self.btn_to_spec = QtWidgets.QPushButton("Add to Spectra tab")
        self.btn_to_spec.setToolTip("Add the selected pixel's spectrum to the "
                                    "comparison in tab 3")
        self.btn_to_spec.clicked.connect(self._send_to_spectra)
        sbar.addWidget(self.btn_to_spec)
        sbar.addStretch(1)
        sv.addLayout(sbar)

        self.spec_canvas = Canvas(figsize=(6, 3.2))
        self.spec_canvas.setMinimumHeight(150)
        sv.addWidget(self.spec_canvas, 1)

        split.addWidget(self.map_canvas)
        split.addWidget(spec_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 280])
        root.addWidget(split, 1)

        self.info = QtWidgets.QLabel("Load a Raman map file to begin.")
        self.info.setStyleSheet("color:#555;")
        root.addWidget(self.info)

        self._show_placeholder()

    # -- reactions ----------------------------------------------------------
    def _alpha_changed(self, v):
        self.main.cfg["optical_alpha"] = v / 100.0
        self.refresh_map()
        self.main.tab_class.refresh_map()

    def _cmap_changed(self, name):
        self.main.cfg["cmap"] = name
        self.refresh_map()

    def _zoom_toggled(self, on):
        self.main.cfg["zoom_to_map"] = bool(on)
        self.refresh_map()
        self.main.tab_class.refresh_map()

    def _ydown_toggled(self, on):
        self.main.cfg["y_down"] = bool(on)
        self.refresh_map()
        self.main.tab_class.refresh_map()
        self.main.tab_spec.refresh_mini()

    def _send_to_spectra(self):
        mf = self.main.mapfile
        if mf is None or self.sel is None:
            self.main.status("Click a pixel on the map first.")
            return
        i = int(mf.index_map[self.sel])
        if i >= 0:
            self.main.tab_spec.add_spectrum(i)
            self.main.status("Added %s to the Spectra tab (%d there now)."
                             % (self.main.tab_spec._name(i).split("  ")[0],
                                len(self.main.tab_spec.items)))

    def _view_changed(self):
        mode = self.cmb_view.currentText()
        custom = mode == "Custom"
        self.spn_x0.setEnabled(custom)
        self.spn_x1.setEnabled(custom)
        if not custom:
            lo, hi = self._view_range()
            for sp, v in ((self.spn_x0, lo), (self.spn_x1, hi)):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
        self._apply_view()

    def _range_edited(self):
        """Editing the bounds by hand implies a custom view."""
        if self.cmb_view.currentText() != "Custom":
            self.cmb_view.blockSignals(True)
            self.cmb_view.setCurrentText("Custom")
            self.cmb_view.blockSignals(False)
            self.spn_x0.setEnabled(True)
            self.spn_x1.setEnabled(True)
        self._apply_view()

    def _reset_view(self):
        self.cmb_view.setCurrentText("2D band")
        self.chk_yfit.setChecked(True)
        self._view_changed()

    def _view_range(self) -> Tuple[float, float]:
        cfg = self.main.cfg
        mode = self.cmb_view.currentText()
        if mode == "Full spectrum":
            if self._spec is not None:
                wn = self._spec["wn"]
                return float(wn[0]), float(wn[-1])
            return 1000.0, 3200.0
        if mode == "Custom":
            lo, hi = self.spn_x0.value(), self.spn_x1.value()
            return (lo, hi) if hi > lo else (lo, lo + 10.0)
        pad = 0.2 * (cfg["band_max"] - cfg["band_min"])
        return cfg["band_min"] - pad, cfg["band_max"] + pad

    def _apply_view(self):
        """Re-limit the axes without refitting anything."""
        if self._spec is None:
            return
        ax = self.spec_canvas.ax
        wn, inten = self._spec["wn"], self._spec["inten"]
        lo, hi = self._view_range()
        ax.set_xlim(lo, hi)
        if self.chk_yfit.isChecked():
            sel = (wn >= lo) & (wn <= hi) & np.isfinite(inten)
            if sel.sum() >= 2:
                y0, y1 = float(np.min(inten[sel])), float(np.max(inten[sel]))
                pad = 0.08 * (y1 - y0) if y1 > y0 else max(1.0, abs(y1) * 0.1)
                ax.set_ylim(y0 - pad, y1 + pad)
        else:
            good = np.isfinite(inten)
            if good.any():
                y0, y1 = float(np.min(inten[good])), float(np.max(inten[good]))
                pad = 0.08 * (y1 - y0) if y1 > y0 else 1.0
                ax.set_ylim(y0 - pad, y1 + pad)
        self.spec_canvas.draw()

    def sync_transform_ui(self):
        """Mirror the active transform into the controls without re-firing."""
        t = self.main.current_transform()
        self._syncing = True
        try:
            if t is not None:
                self.chk_swap.setChecked(bool(t["swap"]))
                self.chk_fx.setChecked(bool(t["flip_x"]))
                self.chk_fy.setChecked(bool(t["flip_y"]))
                self.spn_dx.setValue(float(t["dx"]))
                self.spn_dy.setValue(float(t["dy"]))
                src = (("instrument default, offset not aligned yet"
                        if self.main.cfg.get("orientation_default", "instrument") == "instrument"
                        else "from the #AxisType headers — unverified")
                       if self.main.cfg.get("map_transform") is None else "set")
                self.lbl_transform.setText("Placement (%s):  %s"
                                           % (src, describe_transform(t)))
        finally:
            self._syncing = False

    def _transform_edited(self, *_):
        if self._syncing or self.main.mapfile_raw is None:
            return
        self.main.set_transform({
            "swap": self.chk_swap.isChecked(), "flip_x": self.chk_fx.isChecked(),
            "flip_y": self.chk_fy.isChecked(),
            "dx": self.spn_dx.value(), "dy": self.spn_dy.value()})

    def _show_placeholder(self):
        self.map_canvas.clear()
        self.map_canvas.ax.text(0.5, 0.5, "No map loaded",
                                ha="center", va="center", color="#999")
        self.map_canvas.ax.set_axis_off()
        self.map_canvas.draw()
        self.spec_canvas.clear()
        self.spec_canvas.ax.set_axis_off()
        self.spec_canvas.draw()

    # -- drawing ------------------------------------------------------------
    def height_grid(self, which: str) -> np.ndarray:
        """Raw G or 2D peak height on the current grid (no fitting)."""
        mf = self.main.mapfile
        if which == "G height":
            v = band_height(mf.axis, mf.data, 1582.0, 22.0)
        else:
            c = 0.5 * (self.main.cfg["band_min"] + self.main.cfg["band_max"])
            v = band_height(mf.axis, mf.data, c, 45.0)
        g = np.full(mf.shape, np.nan)
        ok = mf.index_map >= 0
        g[ok] = v[mf.index_map[ok]]
        return g

    def refresh_map(self):
        mf = self.main.mapfile
        quantity = self.cmb_quantity.currentText()
        fwhm = quantity == "FWHM(2D)"
        if mf is None or (fwhm and self.main.fwhm_map is None):
            self._show_placeholder()
            return

        cv = self.map_canvas
        cv.clear()
        ax = cv.ax
        data = self.main.fwhm_map if fwhm else self.height_grid(quantity)
        finite = np.isfinite(data)
        if finite.any():
            vmin = float(np.nanpercentile(data[finite], 2))
            vmax = float(np.nanpercentile(data[finite], 98))
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
        else:
            vmin, vmax = 0, 1

        im = ax.imshow(np.ma.masked_invalid(data), origin="lower",
                       extent=mf.extent, cmap=self.main.cfg["cmap"],
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       aspect="equal", zorder=1)
        cb = side_colorbar(cv.figure, ax, im)
        cb.set_label("FWHM(2D)  [cm$^{-1}$]" if fwhm else "%s  [counts]" % quantity)

        if fwhm and not finite.any():
            # An all-NaN map is otherwise indistinguishable from a map that
            # simply has not been fitted yet, so spell it out.
            tally = self.main.fit_tally or {}
            lines = ["No spectrum passed the quality filters",
                     "(%d pixels, none fitted)" % mf.n_points]
            for why, n in sorted(tally.items(), key=lambda kv: -kv[1])[:4]:
                lines.append("%d × %s" % (n, why))
            lines.append("Adjust under Analysis > Fit parameters")
            ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#b71c1c",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.6", fc="white",
                              ec="#b71c1c", alpha=0.92))

        self._draw_optical(ax)
        if self.chk_grid.isChecked():
            self._draw_grid(ax, mf)
        if self.sel is not None:
            self._draw_selection(ax, mf, *self.sel)

        ax.set_xlabel("X  [µm]")
        ax.set_ylabel("Y  [µm]")
        ax.set_title("%s   (%d × %d pixels)"
                     % ("FWHM of the 2D band" if fwhm else quantity + " (raw)",
                        mf.shape[1], mf.shape[0]))
        # An equal-aspect image in a wide canvas is anchored right by default
        # once the colorbar has taken its slice, stranding the map in the
        # corner with empty figure beside it.
        ax.set_anchor("C")
        frame_axes(ax, mf, self.main.cfg,
                   self.chk_optical.isChecked() and self.main.optical is not None)
        orient_axes(ax, self.main.cfg)
        cv.draw()

    def _draw_optical(self, ax):
        opt = self.main.optical
        if opt is None or not self.chk_optical.isChecked():
            return
        img = self.main.optical_image
        if img is None:
            return
        ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=opt.extent,
                  cmap="gray", alpha=self.main.cfg["optical_alpha"],
                  interpolation="bilinear", aspect="equal", zorder=2)

    @staticmethod
    def _draw_grid(ax, mf: MapFile):
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        for xv in np.linspace(left, right, nx + 1):
            ax.axvline(xv, color="w", lw=0.4, alpha=0.6, zorder=3)
        for yv in np.linspace(bottom, top, ny + 1):
            ax.axhline(yv, color="w", lw=0.4, alpha=0.6, zorder=3)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

    @staticmethod
    def _draw_selection(ax, mf: MapFile, row: int, col: int):
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        dx = (right - left) / nx
        dy = (top - bottom) / ny
        rect = mpatches.Rectangle(
            (left + col * dx, bottom + row * dy), dx, dy,
            fill=False, edgecolor="red", lw=2.0, zorder=4)
        ax.add_patch(rect)

    # -- interaction --------------------------------------------------------
    def _on_click(self, event):
        mf = self.main.mapfile
        if mf is None or event.inaxes is None or event.xdata is None:
            return
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        col = int(np.floor((event.xdata - left) / (right - left) * nx))
        row = int(np.floor((event.ydata - bottom) / (top - bottom) * ny))
        if not (0 <= col < nx and 0 <= row < ny):
            return
        self.sel = (row, col)
        self.refresh_map()
        self.show_spectrum(row, col)

    def show_spectrum(self, row: int, col: int):
        mf = self.main.mapfile
        idx = mf.index_map[row, col]
        cv = self.spec_canvas
        cv.clear()
        ax = cv.ax
        if idx < 0:
            ax.text(0.5, 0.5, "No spectrum at this pixel",
                    ha="center", va="center", color="#999")
            ax.set_axis_off()
            cv.draw()
            return

        wn, inten = mf.axis, mf.data[idx]
        self._spec = dict(wn=wn, inten=inten, row=row, col=col)
        ax.plot(wn, inten, lw=0.8, color="#333", label="raw spectrum")
        lo, hi = self.main.cfg["band_min"], self.main.cfg["band_max"]
        ax.axvspan(lo, hi, color="#ffd54f", alpha=0.30, label="2D band region")

        res = self.main.fits[idx] if self.main.fits else None
        txt = ""
        if res and np.isfinite(res.get("center", np.nan)):
            m = (wn >= lo) & (wn <= hi)
            if m.sum() > 3:
                ax.plot(wn[m], rebuild_fit(wn[m], res), lw=1.4, color="#d81b60",
                        label="Voigt + baseline")
            ax.axvline(res["center"], color="#1e88e5", lw=0.8, ls="--")
            asym = asym_of(res, self.main.cfg["asym_metric"])
            err = res.get("fwhm_err", np.nan)
            txt = ("x = %.2f µm, y = %.2f µm\nFWHM = %.1f %s cm⁻¹   centre = %.1f cm⁻¹\n"
                   "asymmetry (%s) = %s   SNR = %.1f%s"
                   % (mf.x[idx], mf.y[idx], res["fwhm"],
                      ("± %.1f" % err) if np.isfinite(err) else "",
                      res["center"],
                      self.main.cfg["asym_metric"],
                      "%.3f" % asym if np.isfinite(asym) else "n/a",
                      res["snr"],
                      "" if res["ok"] else "   [rejected: %s]" % res["note"]))

        ax.set_xlabel("Raman shift  [cm$^{-1}$]")
        ax.set_ylabel("Intensity  [a.u.]")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=3,
                  borderpad=0.3, handlelength=1.4)
        ax.set_title("Pixel (row %d, col %d)" % (row, col), fontsize=9)
        self._apply_view()
        gf = self.main.g_fits[idx] if self.main.g_fits else None
        if gf is not None and txt:
            r = pixel_ratio(res, gf, self.main.cfg.get("ratio_metric", "height"))
            n = layer_from_ratio(r, self.main.cfg)
            txt += ("\nI(2D)/I(G) = %.2f  →  %s" % (r, label_of(n, "N/A"))
                    if n is not None else
                    "\nI(2D)/I(G) unavailable (%s)" % (gf.get("why") or "G fit rejected"))
            if gf.get("ok") and np.isfinite(gf.get("center", np.nan)):
                gcfg = g_band_cfg(self.main.cfg)
                gm = (wn >= gcfg["band_min"]) & (wn <= gcfg["band_max"])
                ax.plot(wn[gm], rebuild_fit(wn[gm], gf), lw=1.4, color="#1e88e5",
                        label="G fit")
                ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=4,
                          borderpad=0.3, handlelength=1.4)
        cv.draw()
        self.info.setText(txt or "No fit available for this pixel.")
        rejected = bool(res) and not res.get("ok", False)
        self.info.setStyleSheet("color:#b71c1c;" if rejected else "color:#555;")


# ---------------------------------------------------------------------------
# Tab 2 - Classification
# ---------------------------------------------------------------------------

class ThicknessDialog(QtWidgets.QDialog):
    """Thickness from I(2D)/I(G): maps, histogram, editable cutoffs.

    Nothing here touches the main config until 'Use as Auto mode' is pressed,
    so the cutoffs can be explored freely.
    """

    LAYER_ORDER = (1, 2, 3, LAYER_THICK)

    def __init__(self, main: "MainWindow", parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Thickness from I(2D)/I(G)")
        fit_to_screen(self, 1180, 660)
        lay = QtWidgets.QVBoxLayout(self)

        if not main.ratio_ok or main.ratio_maps.get("height") is None:
            why = main.ratio_note if not main.ratio_ok else \
                "the map has not finished fitting yet"
            msg = QtWidgets.QLabel(
                "<b>Not applicable for this file.</b><br><br>"
                "Thickness from I(2D)/I(G) needs both the G band (~1580 cm⁻¹) "
                "and the 2D band (~2700 cm⁻¹) in the same spectrum — here %s."
                "<br><br>Auto mode is unavailable; only <b>Manual</b> layer "
                "count is possible. A wide 600 gr/mm acquisition of the same "
                "area would give you the ratio." % why)
            msg.setWordWrap(True)
            lay.addWidget(msg)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            bb.rejected.connect(self.reject)
            lay.addWidget(bb)
            self.resize(520, 220)
            return

        cfg = main.cfg
        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("Intensity"))
        self.cmb_metric = QtWidgets.QComboBox()
        self.cmb_metric.addItems(["height", "area"])
        self.cmb_metric.setCurrentText(cfg.get("ratio_metric", "height"))
        self.cmb_metric.setToolTip(
            "height — fitted peak heights, what most layer-counting papers "
            "quote and what the default cutoffs assume\n"
            "area — integrated areas; unaffected by instrument broadening, "
            "but the cutoffs must then be recalibrated")
        form.addWidget(self.cmb_metric)
        form.addSpacing(14)
        self.spins = {}
        for key, label in (("1_2", "1L | 2L"), ("2_3", "2L | 3L"),
                           ("3_thick", "3L | ≥4L")):
            form.addWidget(QtWidgets.QLabel(label))
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.01, 20.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.05)
            sp.setValue(float(cfg["ratio_cut"][key]))
            sp.setFixedWidth(76)
            form.addWidget(sp)
            self.spins[key] = sp
        form.addSpacing(14)
        form.addWidget(QtWidgets.QLabel("≥4L pixels:"))
        self.cmb_thick = QtWidgets.QComboBox()
        self.cmb_thick.addItems(["no stacking call", "assume N ="])
        self.cmb_thick.setCurrentIndex(1 if cfg.get("thick_policy") == "assume" else 0)
        form.addWidget(self.cmb_thick)
        self.spn_n = QtWidgets.QSpinBox()
        self.spn_n.setRange(min(LAYERS), max(LAYERS))
        self.spn_n.setValue(int(cfg.get("thick_assume", 5)))
        self.spn_n.setSuffix(" L")
        form.addWidget(self.spn_n)
        form.addStretch(1)
        lay.addLayout(form)

        self.canvas = Canvas(figsize=(11.5, 4.6))
        lay.addWidget(self.canvas, 1)

        self.lbl = QtWidgets.QLabel()
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("font-family:monospace; font-size:11px;")
        lay.addWidget(self.lbl)

        caveat = QtWidgets.QLabel(
            "The default cutoffs are placeholders taken from typical literature "
            "values (1L ~4, 2L ~2, 3L ~1.1, thick ~0.3), and published values "
            "for 2L vary between roughly 1 and 2. The ratio also moves with "
            "doping, strain, oxide thickness and laser energy, and at "
            "14.7 cm⁻¹ resolution the narrow G band is broadened more than "
            "the 2D, which inflates height ratios. Calibrate the cutoffs on "
            "this instrument against a region of known thickness (AFM, "
            "or a clear monolayer) before trusting the layer counts.")
        caveat.setWordWrap(True)
        caveat.setStyleSheet("color:#8a6d00; font-size:11px;")
        lay.addWidget(caveat)

        bb = QtWidgets.QDialogButtonBox()
        self.btn_use = bb.addButton("Use as Auto mode", QtWidgets.QDialogButtonBox.AcceptRole)
        bb.addButton(QtWidgets.QDialogButtonBox.Close)
        bb.accepted.connect(self._use)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        for w in list(self.spins.values()) + [self.spn_n]:
            w.valueChanged.connect(self.redraw)
        self.cmb_metric.currentIndexChanged.connect(self.redraw)
        self.cmb_thick.currentIndexChanged.connect(self.redraw)
        self.redraw()

    # -- current, uncommitted settings --------------------------------------
    def _local_cfg(self) -> dict:
        c = dict(self.main.cfg)
        c["ratio_cut"] = {k: sp.value() for k, sp in self.spins.items()}
        c["ratio_metric"] = self.cmb_metric.currentText()
        c["thick_policy"] = "assume" if self.cmb_thick.currentIndex() else "leave"
        c["thick_assume"] = int(self.spn_n.value())
        return c

    def _valid(self, c: dict) -> bool:
        k = c["ratio_cut"]
        return k["1_2"] > k["2_3"] > k["3_thick"] > 0

    def redraw(self):
        c = self._local_cfg()
        self.spn_n.setEnabled(c["thick_policy"] == "assume")
        valid = self._valid(c)
        self.btn_use.setEnabled(valid)
        mf = self.main.mapfile
        rm = self.main.ratio_maps[c["ratio_metric"]]
        ny, nx = rm.shape
        lm = np.full((ny, nx), np.nan)
        counts = {k: 0 for k in self.LAYER_ORDER}
        for r in range(ny):
            for cc in range(nx):
                n = layer_from_ratio(rm[r, cc], c) if valid else None
                if n is not None:
                    lm[r, cc] = self.LAYER_ORDER.index(n)
                    counts[n] += 1
        with_ratio = int(np.isfinite(rm).sum())

        fig = self.canvas.figure
        fig.clear()
        a1 = fig.add_subplot(131)
        a2 = fig.add_subplot(132)
        a3 = fig.add_subplot(133)
        fin = rm[np.isfinite(rm)]

        opt = self.main.optical
        for ax in (a1, a2):
            if opt is not None and self.main.optical_image is not None:
                ax.imshow(np.ma.masked_invalid(self.main.optical_image),
                          origin="lower", extent=opt.extent, cmap="gray",
                          aspect="equal", zorder=0)
        im = a1.imshow(np.ma.masked_invalid(rm), origin="lower", extent=mf.extent,
                       cmap="magma", aspect="equal", interpolation="nearest",
                       alpha=0.85, zorder=1)
        fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03).set_label(
            "I(2D)/I(G) (%s)" % c["ratio_metric"])
        a1.set_title("I(2D)/I(G)", fontsize=10)

        cols = [na_color(n, c) for n in self.LAYER_ORDER]
        cmap = mcolors.ListedColormap(cols)
        a2.imshow(np.ma.masked_invalid(lm), origin="lower", extent=mf.extent,
                  cmap=cmap, vmin=-0.5, vmax=3.5, aspect="equal",
                  interpolation="nearest", alpha=0.9, zorder=1)
        a2.legend(handles=[Patch(facecolor=cols[i], edgecolor="#555",
                                 label="%s  (%d)" % (label_of(n, "N/A"), counts[n]))
                           for i, n in enumerate(self.LAYER_ORDER)],
                  fontsize=8, loc="upper left", framealpha=0.9)
        a2.set_title("Thickness from the ratio", fontsize=10)
        for ax in (a1, a2):
            ax.set_xlim(mf.extent[0], mf.extent[1])
            ax.set_ylim(mf.extent[2], mf.extent[3])
            orient_axes(ax, self.main.cfg)
            ax.set_xlabel("X [µm]")
        a1.set_ylabel("Y [µm]")

        if fin.size:
            a3.hist(fin, bins=max(8, min(50, fin.size // 3)), color="#607d8b",
                    edgecolor="white", linewidth=0.4)
            for key, lab in (("1_2", "1|2"), ("2_3", "2|3"), ("3_thick", "3|≥4")):
                v = c["ratio_cut"][key]
                a3.axvline(v, color="#d81b60", ls="--", lw=1.1)
                a3.text(v, a3.get_ylim()[1] * 0.96, " " + lab, color="#d81b60",
                        fontsize=8, va="top")
        a3.set_xlabel("I(2D)/I(G)")
        a3.set_ylabel("pixels")
        a3.set_title("Distribution, with the cutoffs", fontsize=10)
        self.canvas.draw()

        if not valid:
            self.lbl.setText("Cutoffs must decrease: 1L|2L > 2L|3L > 3L|≥4L > 0.")
            self.lbl.setStyleSheet("color:#b71c1c; font-family:monospace; font-size:11px;")
            return
        self.lbl.setStyleSheet("font-family:monospace; font-size:11px;")
        med = float(np.median(fin)) if fin.size else float("nan")
        self.lbl.setText(
            "Pixels with both bands fitted: %d of %d   median ratio %.2f\n"
            "1L %d   2L %d   3L %d   ≥4L %d"
            % (with_ratio, mf.n_points, med, counts[1], counts[2],
               counts[3], counts[LAYER_THICK]))

    def _use(self):
        c = self._local_cfg()
        if not self._valid(c):
            return
        for k in ("ratio_cut", "ratio_metric", "thick_policy", "thick_assume"):
            self.main.cfg[k] = c[k]
        self.main.cfg["layer_mode"] = "auto"
        self.main.tab_class.rb_auto.setChecked(True)
        self.accept()
        self.main.tab_class.run_classification()


LAYER_NM = 0.335          # graphene interlayer spacing
REGION_COLORS = ["#1e88e5", "#fb8c00", "#8e24aa", "#43a047", "#e53935", "#00acc1"]


class LayerRegionsDialog(QtWidgets.QDialog):
    """Define terraces from optical contrast and give each its layer count.

    Contrast groups the pixels into regions; it cannot count layers. Each
    region's N is entered from AFM, either directly (a terrace step worked
    out in Gwyddion) or as a height measured down to bare SiO2, converted
    with the editable SiO2 offset. Non-modal, so it can stay open beside
    Gwyddion; Apply re-runs the classification behind it.
    """

    def __init__(self, main: "MainWindow", parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Define layer regions")
        fit_to_screen(self, 1400, 780)
        self.main = main
        mf = main.mapfile
        self.region_of = np.full(mf.n_points, -1, dtype=int)
        self.levels = None
        self.contrast = None
        self.selected = -1
        self.rows = []
        self._texts = []

        root = QtWidgets.QHBoxLayout(self)
        self.canvas = Canvas(figsize=(7.4, 6.8))
        root.addWidget(self.canvas, 1)
        side = QtWidgets.QVBoxLayout()
        root.addLayout(side, 1)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Regions"))
        self.spn_k = QtWidgets.QSpinBox()
        self.spn_k.setRange(1, len(REGION_COLORS))
        self.spn_k.setValue(int(main.cfg.get("contrast_levels", 2)))
        top.addWidget(self.spn_k)
        top.addSpacing(8)
        top.addWidget(QtWidgets.QLabel("Substrate if |contrast| <"))
        self.spn_cut = QtWidgets.QDoubleSpinBox()
        self.spn_cut.setRange(0.0, 1.0)
        self.spn_cut.setDecimals(3)
        self.spn_cut.setSingleStep(0.005)
        self.spn_cut.setValue(float(main.cfg.get("contrast_cut", 0.03)))
        top.addWidget(self.spn_cut)
        self.btn_seg = QtWidgets.QPushButton("Segment")
        self.btn_seg.clicked.connect(self.segment)
        top.addWidget(self.btn_seg)
        top.addStretch(1)
        side.addLayout(top)

        self.lbl_place = QtWidgets.QLabel()
        self.lbl_place.setWordWrap(True)
        self.lbl_place.setStyleSheet("font-size:11px;")
        side.addWidget(self.lbl_place)

        self.tbl = QtWidgets.QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(["Region", "Contrast", "Pixels",
                                            "FWHM", "Entry", "Value", "N"])
        self.tbl.horizontalHeaderItem(3).setToolTip("Median 2D FWHM of the region, cm⁻¹")
        self.tbl.verticalHeader().setVisible(False)
        hh = self.tbl.horizontalHeader()
        # Every column visible at once: Value and N are the ones that matter,
        # so nothing may be pushed behind a scrollbar.
        for col in range(7):
            hh.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents
                                    if col < 4 else QtWidgets.QHeaderView.Stretch)
        self.tbl.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl.itemSelectionChanged.connect(self._row_selected)
        self.tbl.setMinimumHeight(190)
        side.addWidget(self.tbl, 1)

        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(QtWidgets.QLabel("SiO₂ offset"))
        self.spn_off = QtWidgets.QDoubleSpinBox()
        self.spn_off.setRange(0.0, 3.0)
        self.spn_off.setDecimals(2)
        self.spn_off.setSingleStep(0.05)
        self.spn_off.setSuffix(" nm")
        self.spn_off.setValue(float(main.cfg.get("sio2_offset", 0.5)))
        self.spn_off.setToolTip(
            "Extra apparent height of a flake-to-SiO₂ step in AFM (tip and "
            "water-layer effects), typically 0.3–1.0 nm. Only used for rows "
            "entered as 'Height from SiO₂'.")
        self.spn_off.valueChanged.connect(self.update_status)
        orow.addWidget(self.spn_off)
        orow.addWidget(QtWidgets.QLabel("   N = (height − offset) / %.3f nm" % LAYER_NM))
        orow.addStretch(1)
        side.addLayout(orow)

        self.chk_edges = QtWidgets.QCheckBox("Exclude step-edge pixels")
        self.chk_edges.setChecked(bool(main.cfg.get("exclude_edges", True)))
        self.chk_edges.setToolTip(
            "Pixels next to a region of different N: the laser spot spans "
            "both terraces there, so the spectrum mixes two thicknesses.")
        self.chk_edges.toggled.connect(self.update_status)
        side.addWidget(self.chk_edges)

        self.lbl_warn = QtWidgets.QLabel()
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("font-size:11px;")
        side.addWidget(self.lbl_warn)

        bot = QtWidgets.QHBoxLayout()
        b_sv = QtWidgets.QPushButton("Save…")
        b_sv.clicked.connect(self.save_regions)
        b_ld = QtWidgets.QPushButton("Load…")
        b_ld.clicked.connect(self.load_regions)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setStyleSheet("font-weight:bold;")
        self.btn_apply.clicked.connect(self.apply)
        b_close = QtWidgets.QPushButton("Close")
        b_close.clicked.connect(self.close)
        for w in (b_sv, b_ld):
            bot.addWidget(w)
        bot.addStretch(1)
        bot.addWidget(self.btn_apply)
        bot.addWidget(b_close)
        side.addLayout(bot)

        self._build_figure()
        self.canvas.canvas.mpl_connect("button_press_event", self._clicked)

        if main.optical is None or main.optical_image is None:
            self.lbl_place.setText("No camera image loaded — regions come from optical "
                                   "contrast, so load one in tab 1 first.")
            self.lbl_place.setStyleSheet("color:#b71c1c; font-size:11px;")
            for w in (self.spn_k, self.spn_cut, self.btn_seg, self.btn_apply):
                w.setEnabled(False)
            return
        if main.cfg.get("map_transform") is None:
            self.lbl_place.setText("⚠ The map's offset has not been aligned yet — run "
                                   "Auto-align in tab 1 first, or contrast will be "
                                   "read from slightly the wrong place.")
            self.lbl_place.setStyleSheet("color:#b71c1c; font-size:11px;")
        else:
            self.lbl_place.setText("Placement: %s" %
                                   describe_transform(main.cfg["map_transform"]))
            self.lbl_place.setStyleSheet("color:#555; font-size:11px;")
        self.segment()

    # -- figure -------------------------------------------------------------
    def _build_figure(self):
        mf = self.main.mapfile
        fig = self.canvas.figure
        fig.clear()
        gsp = fig.add_gridspec(3, 1)
        self.ax = fig.add_subplot(gsp[:2, 0])
        self.axh = fig.add_subplot(gsp[2, 0])
        opt = self.main.optical
        if opt is not None and self.main.optical_image is not None:
            self.ax.imshow(np.ma.masked_invalid(self.main.optical_image),
                           origin="lower", extent=opt.extent, cmap="gray",
                           aspect="equal", zorder=0)
        ny, nx = mf.shape
        self.overlay = self.ax.imshow(np.zeros((ny, nx, 4)), origin="lower",
                                      extent=mf.extent, interpolation="nearest",
                                      aspect="equal", zorder=1)
        left, right, bottom, top = mf.extent
        pad = 0.15 * max(right - left, top - bottom)
        self.ax.set_xlim(left - pad, right + pad)
        self.ax.set_ylim(bottom - pad, top + pad)
        self.ax.set_xlabel("X [µm]")
        self.ax.set_ylabel("Y [µm]")
        self.ax.set_title("Regions (click one to select its row)", fontsize=10)
        orient_axes(self.ax, self.main.cfg)

    def _n_by_spec(self) -> np.ndarray:
        n = np.zeros(self.main.mapfile.n_points, dtype=int)
        for j in range(len(self.rows)):
            nj = self.row_n(j)[1]
            n[self.region_of == j] = max(nj, 0)
        return n

    def refresh_map(self):
        mf = self.main.mapfile
        ny, nx = mf.shape
        rgba = np.zeros((ny, nx, 4))
        reg = np.full((ny, nx), -1, dtype=int)
        ok = mf.index_map >= 0
        reg[ok] = self.region_of[mf.index_map[ok]]
        for j in range(len(self.rows)):
            a = 0.85 if j == self.selected else 0.5
            rgba[reg == j] = mcolors.to_rgba(REGION_COLORS[j % len(REGION_COLORS)], a)
        if self.chk_edges.isChecked():
            ng = np.zeros((ny, nx), dtype=int)
            nb = self._n_by_spec()
            ng[ok] = nb[mf.index_map[ok]]
            rgba[terrace_edges(ng)] = (1.0, 1.0, 1.0, 0.85)
        self.overlay.set_data(rgba)
        for t in self._texts:
            t.remove()
        self._texts = []
        for j in range(len(self.rows)):
            m = self.region_of == j
            if m.any():
                self._texts.append(self.ax.text(
                    float(np.median(mf.x[m])), float(np.median(mf.y[m])), "R%d" % (j + 1),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc=REGION_COLORS[j % 6],
                              ec="none", alpha=0.9)))

        self.axh.clear()
        if self.contrast is not None:
            fin = self.contrast[np.isfinite(self.contrast)]
            if fin.size:
                self.axh.hist(fin, bins=max(10, min(60, fin.size // 3)),
                              color="#90a4ae", edgecolor="white", linewidth=0.4)
            cut = self.spn_cut.value()
            for v in (-cut, cut):
                self.axh.axvline(v, color="#555", ls=":", lw=1)
            if self.levels is not None:
                top = self.axh.get_ylim()[1]
                for j, c in enumerate(self.levels):
                    self.axh.axvline(c, color=REGION_COLORS[j % 6], lw=1.6)
                    self.axh.text(c, top * 0.92, " R%d" % (j + 1),
                                  color=REGION_COLORS[j % 6], fontsize=9,
                                  fontweight="bold")
                srt = np.sort(self.levels)
                for a, b in zip(srt[:-1], srt[1:]):
                    self.axh.axvline(0.5 * (a + b), color="#555", ls="--", lw=0.8)
            self.axh.set_xlabel(r"optical contrast  $(I - I_{\mathrm{SiO_2}})\,/\,I_{\mathrm{SiO_2}}$")
            self.axh.set_ylabel("pixels")
        self.canvas.draw()

    # -- segmentation --------------------------------------------------------
    def segment(self):
        mf, opt = self.main.mapfile, self.main.optical
        if opt is None:
            return
        self.contrast, _ = optical_contrast(mf, opt)
        cut = self.spn_cut.value()
        on = np.isfinite(self.contrast) & (np.abs(self.contrast) >= cut)
        k = self.spn_k.value()
        self.region_of[:] = -1
        if on.sum() < k:
            self.levels = None
            self._build_rows([])
            self.lbl_warn.setText("Too few pixels above the substrate cut to make "
                                  "%d regions — lower the cut or the region count." % k)
            self.lbl_warn.setStyleSheet("color:#b71c1c; font-size:11px;")
            self.refresh_map()
            return
        c = kmeans_1d(self.contrast[on], k)
        c = c[np.argsort(np.abs(c))]          # R1 = faintest = thinnest
        self.levels = c
        idx = np.where(on)[0]
        self.region_of[idx] = np.argmin(np.abs(self.contrast[idx, None] - c[None, :]), axis=1)

        # Defaults: a region with almost no 2D-band fits, or contrast opposite
        # to the flake's, is not a graphene terrace and starts unassigned.
        fit_ok = np.array([bool(r.get("ok")) for r in self.main.fits])
        n_pos = int(((self.contrast >= cut) & fit_ok).sum())
        n_neg = int(((self.contrast <= -cut) & fit_ok).sum())
        sign = 1.0 if n_pos >= n_neg else -1.0
        nxt = int(self.main.cfg.get("manual_layer", 3))
        entries = []
        for j, cj in enumerate(c):
            m = self.region_of == j
            frac = float(fit_ok[m].mean()) if m.any() else 0.0
            if np.sign(cj) == sign and frac >= 0.2:
                entries.append(("N", float(min(9, nxt))))
                nxt += 1
            else:
                entries.append(("N", 0.0))
        self._build_rows(entries)

    def _build_rows(self, entries):
        fw = np.array([r["fwhm"] if r.get("ok") else np.nan for r in self.main.fits])
        fit_ok = np.isfinite(fw)
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(entries))
        self.rows = []
        for j, (kind, val) in enumerate(entries):
            m = self.region_of == j
            med = np.nanmedian(fw[m]) if (m & fit_ok).any() else np.nan
            cells = ("R%d" % (j + 1),
                     "%+.3f" % self.levels[j] if self.levels is not None else "—",
                     "%d" % m.sum(),
                     "—" if not np.isfinite(med) else "%.1f" % med)
            for col, txt in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                if col == 0:
                    it.setBackground(QtGui.QBrush(QtGui.QColor(REGION_COLORS[j % 6])))
                    it.setForeground(QtGui.QBrush(QtGui.QColor("white")))
                self.tbl.setItem(j, col, it)
            cmb = QtWidgets.QComboBox()
            cmb.addItems(["N", "SiO₂ height"])
            cmb.setToolTip("N — layer count, e.g. from a terrace step in Gwyddion\n"
                           "SiO₂ height — AFM step down to bare SiO₂, converted "
                           "with the offset below")
            cmb.setCurrentIndex(1 if kind == "height" else 0)
            spn = QtWidgets.QDoubleSpinBox()
            self.tbl.setCellWidget(j, 4, cmb)
            self.tbl.setCellWidget(j, 5, spn)
            nit = QtWidgets.QTableWidgetItem("—")
            nit.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            nit.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl.setItem(j, 6, nit)
            self.rows.append({"entry": cmb, "value": spn})
            self._configure_value(j, val)
            cmb.currentIndexChanged.connect(lambda _i, j=j: self._configure_value(j, None))
            spn.valueChanged.connect(self.update_status)
        self.tbl.blockSignals(False)
        self.update_status()

    def _configure_value(self, j: int, val):
        spn = self.rows[j]["value"]
        is_h = self.rows[j]["entry"].currentIndex() == 1
        spn.blockSignals(True)
        if is_h:
            spn.setDecimals(2)
            spn.setRange(0.0, 60.0)
            spn.setSingleStep(0.05)
            spn.setSuffix(" nm")
            spn.setToolTip("Step height from the flake down to bare SiO₂, from AFM")
        else:
            spn.setDecimals(0)
            spn.setRange(0, 9)
            spn.setSingleStep(1)
            spn.setSuffix("")
            spn.setToolTip("Layer count, e.g. from a terrace step in Gwyddion. 0 = unassigned")
        if val is not None:
            spn.setValue(val)
        elif is_h:
            spn.setValue(0.0)
        spn.blockSignals(False)
        self.update_status()

    # -- layer numbers + checks ---------------------------------------------
    def row_n(self, j: int):
        """(unrounded N or None, N used (0 = unassigned), note)."""
        is_h = self.rows[j]["entry"].currentIndex() == 1
        v = self.rows[j]["value"].value()
        if not is_h:
            n = int(round(v))
            return (float(n) if n > 0 else None), n, ""
        if v <= 0:
            return None, 0, ""
        raw = (v - self.spn_off.value()) / LAYER_NM
        n = int(round(raw))
        note = ""
        if n < 1:
            note = "height is below the offset"
            n = 0
        elif abs(raw - n) > 0.25:
            note = "%.2f is not close to a whole number of layers" % raw
        return raw, n, note

    def update_status(self, *_):
        if not self.rows:
            self.refresh_map()
            return
        warns, infos = [], []
        assigned = []
        for j in range(len(self.rows)):
            raw, n, note = self.row_n(j)
            it = self.tbl.item(j, 6)
            is_h = self.rows[j]["entry"].currentIndex() == 1
            if n <= 0:
                it.setText("—" if not note else "⚠ —")
                it.setForeground(QtGui.QBrush(QtGui.QColor("#999")))
            else:
                it.setText(("%.2f → %d" % (raw, n) if is_h else "%d" % n) +
                           (" ⚠" if note else ""))
                it.setForeground(QtGui.QBrush(QtGui.QColor("#b71c1c" if note else "#1b5e20")))
                assigned.append((abs(self.levels[j]) if self.levels is not None else 0, j, n))
                if n not in self.main.cfg["fwhm"] and n not in (1, 2):
                    warns.append("R%d: no reference widths for %dL in the table." % (j + 1, n))
            if note:
                warns.append("R%d: %s — check the SiO₂ offset or the line "
                             "profile." % (j + 1, note))
        assigned.sort()
        for (ca, ja, na), (cb, jb, nb) in zip(assigned[:-1], assigned[1:]):
            if nb < na:
                warns.append("R%d has stronger contrast than R%d but fewer layers "
                             "(%d vs %d) — check the values or which terrace is "
                             "which." % (jb + 1, ja + 1, nb, na))
            elif nb == na:
                infos.append("R%d and R%d are both %dL and will be treated as one "
                             "terrace." % (ja + 1, jb + 1, na))
        nb = self._n_by_spec()
        mf = self.main.mapfile
        g = np.zeros(mf.shape, dtype=int)
        ok = mf.index_map >= 0
        g[ok] = nb[mf.index_map[ok]]
        ne = int(terrace_edges(g).sum())
        infos.append("%d pixels assigned; %d step-edge pixels %s."
                     % (int((nb > 0).sum()), ne,
                        "excluded" if self.chk_edges.isChecked() else "kept"))
        html = "".join("<div style='color:#b71c1c'>⚠ %s</div>" % w for w in warns) + \
               "".join("<div style='color:#555'>%s</div>" % i for i in infos)
        self.lbl_warn.setText(html)
        self.btn_apply.setEnabled(bool(assigned))
        self.refresh_map()

    # -- selection ------------------------------------------------------------
    def _row_selected(self):
        rows = self.tbl.selectionModel().selectedRows()
        self.selected = rows[0].row() if rows else -1
        self.refresh_map()

    def _clicked(self, event):
        mf = self.main.mapfile
        if event.inaxes is not self.ax or event.xdata is None:
            return
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        col = int(np.floor((event.xdata - left) / (right - left) * nx))
        row = int(np.floor((event.ydata - bottom) / (top - bottom) * ny))
        if 0 <= col < nx and 0 <= row < ny and mf.index_map[row, col] >= 0:
            j = int(self.region_of[mf.index_map[row, col]])
            if j >= 0:
                self.tbl.selectRow(j)

    # -- apply / persistence ----------------------------------------------------
    def apply(self):
        cfg = self.main.cfg
        nb = self._n_by_spec()
        if not (nb > 0).any():
            return
        self.main.layer_by_spec = nb
        cfg["exclude_edges"] = self.chk_edges.isChecked()
        cfg["contrast_levels"] = self.spn_k.value()
        cfg["contrast_cut"] = self.spn_cut.value()
        cfg["sio2_offset"] = self.spn_off.value()
        cfg["layer_mode"] = "map"
        tab = self.main.tab_class
        tab.rb_map.setChecked(True)
        tab.update_map_status()
        tab.run_classification()

    def save_regions(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save layer regions", self.main.suggest_name("_regions.json"),
            "JSON (*.json)")
        if not path:
            return
        mf = self.main.mapfile
        data = {"sio2_offset": self.spn_off.value(), "cut": self.spn_cut.value(),
                "levels": [float(v) for v in (self.levels if self.levels is not None else [])],
                "regions": [{"entry": "height" if r["entry"].currentIndex() else "N",
                             "value": r["value"].value()} for r in self.rows],
                "spectra": [[round(float(mf.raw0[i]), 4), round(float(mf.raw1[i]), 4),
                             int(self.region_of[i])] for i in range(mf.n_points)]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.main.status("Regions saved to %s" % path)

    def load_regions(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load layer regions", "", "JSON (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        mf = self.main.mapfile
        key = {(round(float(a), 3), round(float(b), 3)): i
               for i, (a, b) in enumerate(zip(mf.raw0, mf.raw1))}
        self.region_of[:] = -1
        hit = 0
        for a, b, r in data.get("spectra", []):
            i = key.get((round(a, 3), round(b, 3)))
            if i is not None:
                self.region_of[i] = int(r)
                hit += 1
        if hit == 0:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME, "No positions in that file match this map — it "
                "was saved for a different acquisition.")
            return
        self.spn_off.blockSignals(True)
        self.spn_off.setValue(float(data.get("sio2_offset", 0.5)))
        self.spn_off.blockSignals(False)
        lv = np.asarray(data.get("levels", []), dtype=float)
        self.levels = lv if lv.size else None
        self._build_rows([(r["entry"], float(r["value"])) for r in data.get("regions", [])])


class ClassificationTab(QtWidgets.QWidget):

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.layers: Optional[np.ndarray] = None
        self.calls: Optional[np.ndarray] = None
        self.ncands: Optional[np.ndarray] = None
        self.reasons: Optional[np.ndarray] = None
        self._build()

    # -- UI -----------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)

        # ---------------- left: inputs ----------------
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(620)
        pl = QtWidgets.QVBoxLayout(panel)

        # Layer assignment
        gb_layer = QtWidgets.QGroupBox("Layer count")
        gl = QtWidgets.QGridLayout(gb_layer)
        self.rb_manual = QtWidgets.QRadioButton("Manual")
        self.rb_auto = QtWidgets.QRadioButton("Auto — thickness from I(2D)/I(G)")
        self.rb_manual.setChecked(True)
        self.cmb_layer = QtWidgets.QComboBox()
        self.cmb_layer.addItems(["%dL" % n for n in LAYERS])
        self.rb_manual.toggled.connect(self.cmb_layer.setEnabled)
        gl.addWidget(self.rb_manual, 0, 0)
        gl.addWidget(self.cmb_layer, 0, 1)
        gl.addWidget(self.rb_auto, 1, 0, 1, 2)
        self.btn_thick = QtWidgets.QPushButton("Calculate thickness from I(2D)/I(G)…")
        self.btn_thick.clicked.connect(self.show_thickness)
        gl.addWidget(self.btn_thick, 2, 0, 1, 2)
        self.lbl_auto = QtWidgets.QLabel("Load a map to see whether Auto is available.")
        self.lbl_auto.setWordWrap(True)
        self.lbl_auto.setStyleSheet("font-size:11px;")
        gl.addWidget(self.lbl_auto, 3, 0, 1, 2)
        note = QtWidgets.QLabel(
            "Auto sets each pixel's layer count from the ratio of the 2D and "
            "G band intensities, so it needs a wide acquisition (600 gr/mm) "
            "containing both. The ratio drops steeply from 1L to 3L and then "
            "flattens, so it resolves 1L, 2L, 3L and '≥4L' — not 5L versus "
            "7L. ABA/ABC is only defined from 3 layers up, so 1L and 2L "
            "pixels get a thickness but no stacking call. A 2D-only file "
            "(1800 gr/mm) has no G band: Manual only.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666; font-size:11px;")
        gl.addWidget(note, 4, 0, 1, 2)
        self.rb_map = QtWidgets.QRadioButton("Layer map — N per region (from AFM)")
        gl.addWidget(self.rb_map, 5, 0, 1, 2)
        self.btn_layermap = QtWidgets.QPushButton("Define layer regions…")
        self.btn_layermap.clicked.connect(self.show_layermap)
        gl.addWidget(self.btn_layermap, 6, 0, 1, 2)
        self.lbl_map = QtWidgets.QLabel("No layer map yet.")
        self.lbl_map.setWordWrap(True)
        self.lbl_map.setStyleSheet("font-size:11px;")
        gl.addWidget(self.lbl_map, 7, 0, 1, 2)
        note2 = QtWidgets.QLabel(
            "Optional, for flakes with terraces of different thickness. Optical "
            "contrast draws the regions, you enter each region's N or its AFM "
            "height from SiO₂, and each region is classified against its own "
            "width windows. Simple flakes: just use Manual.")
        note2.setWordWrap(True)
        note2.setStyleSheet("color:#666; font-size:11px;")
        gl.addWidget(note2, 8, 0, 1, 2)
        pl.addWidget(gb_layer)

        # Technique
        gb_tech = QtWidgets.QGroupBox("Technique")
        tl = QtWidgets.QVBoxLayout(gb_tech)
        self.rb_w = QtWidgets.QRadioButton("FWHM(2D) width only")
        self.rb_a = QtWidgets.QRadioButton("2D shape asymmetry only")
        self.rb_b = QtWidgets.QRadioButton("Both combined (AND)")
        self.rb_b.setChecked(True)
        for rb in (self.rb_w, self.rb_a, self.rb_b):
            tl.addWidget(rb)
        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(QtWidgets.QLabel("Metric"))
        self.cmb_metric = QtWidgets.QComboBox()
        self.cmb_metric.addItems(["hwhm", "area", "residual"])
        self.cmb_metric.setToolTip(
            "hwhm     — ratio of the two half-widths at half maximum\n"
            "area     — ratio of the two lobe areas\n"
            "residual — low-frequency shoulder left over by the Voigt fit")
        arow.addWidget(self.cmb_metric)
        arow.addWidget(QtWidgets.QLabel("Threshold"))
        self.spn_asym = QtWidgets.QDoubleSpinBox()
        self.spn_asym.setRange(-1.0, 1.0)
        self.spn_asym.setSingleStep(0.01)
        self.spn_asym.setDecimals(3)
        self.spn_asym.setValue(0.15)
        arow.addWidget(self.spn_asym)
        arow.addStretch(1)
        tl.addLayout(arow)
        anote = QtWidgets.QLabel(
            "Asymmetry is positive when the band carries extra weight on its "
            "low-frequency shoulder (~2680 cm⁻¹), the ABC signature. All three "
            "metrics are measured about the maximum of the smoothed band, not "
            "the fitted centre — a single Voigt drifts into the shoulder and "
            "masks the asymmetry. The 0.15 default is a placeholder: use "
            "'Threshold histograms' to set it from your own data.")
        anote.setWordWrap(True)
        anote.setStyleSheet("color:#666; font-size:11px;")
        tl.addWidget(anote)
        pl.addWidget(gb_tech)

        # Threshold table
        gb_tab = QtWidgets.QGroupBox("FWHM thresholds and colours (editable)")
        tabl = QtWidgets.QVBoxLayout(gb_tab)
        self.table = ThresholdTable(self.main.cfg)
        self.table.changed.connect(self._table_changed)
        tabl.addWidget(self.table)
        hint = QtWidgets.QLabel(
            "A width must land inside a window to be called that stacking; one "
            "matching neither is left Unclassified rather than pushed to the "
            "nearer side. Windows shown in red overlap, in amber leave a dead "
            "zone between them. Click a colour cell to change it. Reference "
            "values from Bouhafs et al., Carbon 177 (2021) Fig. 2c — verify "
            "them, and the tolerances, against your own samples.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        tabl.addWidget(hint)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Set every ± to"))
        self.spn_tol = QtWidgets.QDoubleSpinBox()
        self.spn_tol.setRange(0.0, 200.0)
        self.spn_tol.setDecimals(1)
        self.spn_tol.setSingleStep(1.0)
        self.spn_tol.setValue(DEFAULT_TOL)
        self.spn_tol.setSuffix(" cm⁻¹")
        self.spn_tol.setFixedWidth(100)
        trow.addWidget(self.spn_tol)
        btn_apply = QtWidgets.QPushButton("Apply to all")
        btn_apply.clicked.connect(
            lambda: self.table.set_all_tolerances(self.spn_tol.value()))
        trow.addWidget(btn_apply)
        btn_half = QtWidgets.QPushButton("Set ± to half-gap")
        btn_half.setToolTip(
            "Each window reaches exactly to the midpoint, so the ABA/ABC cut "
            "is the old boundary rule — but a width far from the table is now "
            "left Unclassified instead of being forced to a side.")
        btn_half.clicked.connect(self.table.set_tolerances_to_half_gap)
        trow.addWidget(btn_half)
        trow.addStretch(1)
        tabl.addLayout(trow)

        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(QtWidgets.QLabel("Widths inside both windows:"))
        self.cmb_overlap = QtWidgets.QComboBox()
        self.cmb_overlap.addItems(["nearest reference value",
                                   "leave Unclassified"])
        self.cmb_overlap.setToolTip(
            "Only reached when a tolerance is wide enough that the ABA and "
            "ABC windows collide. 'nearest' keeps the midpoint behaviour "
            "inside the overlap; 'Unclassified' declines to call it.")
        orow.addWidget(self.cmb_overlap)
        orow.addStretch(1)
        tabl.addLayout(orow)

        rowb = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Reset to Bouhafs defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        rowb.addWidget(btn_reset)
        rowb.addStretch(1)
        tabl.addLayout(rowb)
        pl.addWidget(gb_tab)

        # Run + export
        self.btn_hist = QtWidgets.QPushButton("Threshold histograms…")
        self.btn_hist.clicked.connect(self.show_histograms)
        pl.addWidget(self.btn_hist)

        self.btn_run = QtWidgets.QPushButton("Classify map")
        self.btn_run.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_run.clicked.connect(self.run_classification)
        pl.addWidget(self.btn_run)

        erow = QtWidgets.QHBoxLayout()
        self.btn_png = QtWidgets.QPushButton("Export figure (PDF/SVG/PNG)…")
        self.btn_csv = QtWidgets.QPushButton("Export CSV…")
        self.btn_png.clicked.connect(self.export_png)
        self.btn_csv.clicked.connect(self.export_csv)
        erow.addWidget(self.btn_png)
        erow.addWidget(self.btn_csv)
        pl.addLayout(erow)

        self.stats = QtWidgets.QTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(150)
        self.stats.setStyleSheet("font-family:monospace; font-size:11px;")
        pl.addWidget(self.stats)
        pl.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(640)
        root.addWidget(scroll)

        # ---------------- right: output map ----------------
        right = QtWidgets.QVBoxLayout()
        obar = QtWidgets.QHBoxLayout()
        self.chk_optical = QtWidgets.QCheckBox("Overlay optical")
        self.chk_optical.toggled.connect(self.refresh_map)
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(50)
        self.sld_alpha.setFixedWidth(140)
        self.sld_alpha.valueChanged.connect(self._alpha_changed)
        self.chk_grid = QtWidgets.QCheckBox("Grid overlay")
        self.chk_grid.toggled.connect(self.refresh_map)
        obar.addWidget(self.chk_optical)
        obar.addWidget(QtWidgets.QLabel("Opacity"))
        obar.addWidget(self.sld_alpha)
        obar.addWidget(self.chk_grid)
        obar.addStretch(1)
        right.addLayout(obar)
        self.canvas = Canvas(figsize=(6, 6))
        right.addWidget(self.canvas, 1)
        root.addLayout(right, 1)

        self._placeholder()

    # -- reactions ----------------------------------------------------------
    def _alpha_changed(self, v):
        self.main.cfg["optical_alpha"] = v / 100.0
        self.refresh_map()

    def _table_changed(self):
        self.stats.append("Thresholds edited — press 'Classify map' to re-run.")

    def _reset_defaults(self):
        self.main.cfg["fwhm"] = {k: dict(v) for k, v in BOUHAFS_FWHM.items()}
        self.main.cfg["colors"] = {k: dict(v) for k, v in DEFAULT_COLORS.items()}
        self.main.cfg["tol"] = {k: {"ABA": DEFAULT_TOL, "ABC": DEFAULT_TOL}
                                for k in BOUHAFS_FWHM}
        self.spn_tol.setValue(DEFAULT_TOL)
        self.table.reload()

    def _pull_config(self):
        cfg = self.main.cfg
        cfg["layer_mode"] = ("manual" if self.rb_manual.isChecked()
                             else "map" if self.rb_map.isChecked() else "auto")
        cfg["manual_layer"] = LAYERS[self.cmb_layer.currentIndex()]
        cfg["technique"] = ("fwhm" if self.rb_w.isChecked()
                            else "asym" if self.rb_a.isChecked() else "both")
        cfg["asym_metric"] = self.cmb_metric.currentText()
        cfg["asym_threshold"] = float(self.spn_asym.value())
        cfg["overlap_policy"] = ("unclassified"
                                 if self.cmb_overlap.currentIndex() else "nearest")

    def push_config(self):
        """Mirror a freshly loaded config into the widgets."""
        cfg = self.main.cfg
        self.rb_manual.setChecked(cfg["layer_mode"] == "manual")
        self.rb_auto.setChecked(cfg["layer_mode"] == "auto")
        self.rb_map.setChecked(cfg["layer_mode"] == "map")
        if cfg["manual_layer"] in LAYERS:
            self.cmb_layer.setCurrentIndex(LAYERS.index(cfg["manual_layer"]))
        self.rb_w.setChecked(cfg["technique"] == "fwhm")
        self.rb_a.setChecked(cfg["technique"] == "asym")
        self.rb_b.setChecked(cfg["technique"] == "both")
        self.cmb_metric.setCurrentText(cfg["asym_metric"])
        self.spn_asym.setValue(cfg["asym_threshold"])
        self.cmb_overlap.setCurrentIndex(
            1 if cfg.get("overlap_policy") == "unclassified" else 0)
        self.table.cfg = cfg
        self.table.reload()

    # -- classification -----------------------------------------------------
    def update_auto_availability(self):
        """Enable Auto only when the file holds both bands."""
        ok, note = self.main.ratio_ok, self.main.ratio_note
        self.rb_auto.setEnabled(ok)
        if ok:
            self.lbl_auto.setText("✓ Auto available: " + note + ".")
            self.lbl_auto.setStyleSheet("color:#1b5e20; font-size:11px;")
        else:
            if self.rb_auto.isChecked():
                self.rb_manual.setChecked(True)
            self.lbl_auto.setText("✗ Auto not applicable for this file: %s. "
                                  "Only Manual is possible." % note)
            self.lbl_auto.setStyleSheet("color:#b71c1c; font-size:11px;")

    def update_map_status(self):
        lb = self.main.layer_by_spec
        if lb is None or not (lb > 0).any():
            self.lbl_map.setText("No layer map yet.")
            self.lbl_map.setStyleSheet("color:#666; font-size:11px;")
            return
        parts = "  ".join("%dL: %d" % (n, int((lb == n).sum()))
                          for n in sorted(set(lb[lb > 0].tolist())))
        self.lbl_map.setText("Layer map: %s   (unassigned %d)"
                             % (parts, int((lb == 0).sum())))
        self.lbl_map.setStyleSheet("color:#1b5e20; font-size:11px;")

    def show_layermap(self):
        if self.main.mapfile is None or not self.main.fits:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load and fit a Raman map first (tab 1).")
            return
        dlg = getattr(self, "_regions_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        self._regions_dlg = LayerRegionsDialog(self.main, self)
        self._regions_dlg.show()

    def close_regions_dialog(self):
        dlg = getattr(self, "_regions_dlg", None)
        if dlg is not None:
            dlg.close()
            self._regions_dlg = None

    def show_thickness(self):
        if self.main.mapfile is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load a Raman map first (tab 1).")
            return
        ThicknessDialog(self.main, self).exec_()

    def run_classification(self):
        if self.main.fwhm_map is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load a Raman map first (tab 1).")
            return
        self._pull_config()
        cfg = self.main.cfg
        if cfg["layer_mode"] == "auto" and not self.main.ratio_ok:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                "Auto mode is not applicable for this file: %s.\n\nUsing "
                "Manual." % self.main.ratio_note)
            cfg["layer_mode"] = "manual"
            self.rb_manual.setChecked(True)
        asym = self.main.asym_maps[cfg["asym_metric"]]
        if cfg["layer_mode"] == "map":
            lb = self.main.layer_by_spec
            if lb is None or not (lb > 0).any():
                QtWidgets.QMessageBox.information(
                    self, APP_NAME, "Layer-map mode needs a layer map — use "
                    "'Define layer regions…' first.")
                return
        ratio = self.main.ratio_maps.get(cfg.get("ratio_metric", "height"))
        self.layers, self.calls, self.ncands, self.reasons = classify_map(
            self.main.fwhm_map, asym, cfg, ratio, self.main.layer_grid())
        self.refresh_map()
        self._report()
        ts = getattr(self.main, "tab_spec", None)
        if ts is not None:
            ts._rebuild_list()           # names carry the class label
            ts.refresh_mini()

    def _report(self):
        labels: Dict[str, int] = {}
        ny, nx = self.layers.shape
        for r in range(ny):
            for c in range(nx):
                lab = label_of(int(self.layers[r, c]), self.calls[r, c])
                labels[lab] = labels.get(lab, 0) + 1
        total = ny * nx
        cfg = self.main.cfg
        fitted = int(np.isfinite(self.main.fwhm_map).sum())
        lines = ["Pixels: %d   (%d × %d)   fitted: %d" % (total, nx, ny, fitted),
                 "Mode: %s   technique: %s   metric: %s   asym ≥ %.3f"
                 % (cfg["layer_mode"], cfg["technique"],
                    cfg["asym_metric"], cfg["asym_threshold"])]
        if fitted:
            errs = np.array([r.get("fwhm_err", np.nan) for r in self.main.fits
                             if r.get("ok")], dtype=float)
            errs = errs[np.isfinite(errs)]
            if errs.size:
                med_err = float(np.median(errs))
                gaps = [abs(cfg["fwhm"][L]["ABC"] - cfg["fwhm"][L]["ABA"]) / 2.0
                        for L in cfg["fwhm"]]
                lines.append("Median FWHM uncertainty: ± %.1f cm⁻¹  "
                             "(threshold-to-boundary gap ≈ %.0f cm⁻¹)%s"
                             % (med_err, float(np.median(gaps)),
                                "   ← too large to separate ABA from ABC"
                                if med_err > 0.5 * float(np.median(gaps)) else ""))
            out = 0
            for r in range(ny):
                for c in range(nx):
                    why = self.reasons[r, c] or ""
                    if self.calls[r, c] is None and "outside both" in why:
                        out += 1
            if out:
                lines.append("Outside every tolerance window: %d of %d fitted "
                             "pixels (%.1f %%)" % (out, fitted, 100.0 * out / fitted))
            if cfg["layer_mode"] == "map":
                g = self.main.layer_grid()
                fitted_mask = np.isfinite(self.main.fwhm_map)
                per = "  ".join("%dL %d" % (n, int(((g == n) & fitted_mask).sum()))
                                for n in sorted(set(g[g > 0].tolist())))
                lines.append("Layer map: fitted pixels per region — %s   "
                             "(no N assigned: %d)"
                             % (per, int(((g == 0) & fitted_mask).sum())))
                if cfg.get("exclude_edges", True):
                    e = int((terrace_edges(g) & fitted_mask).sum())
                    lines.append("Terrace-edge pixels excluded: %d" % e)
            elif cfg["layer_mode"] == "manual":
                amb = int(((self.ncands > 1) & np.isfinite(self.main.fwhm_map)).sum())
                lines.append("Ambiguous layer assignment: %d of %d fitted pixels "
                             "(%.1f %%) matched more than one layer window"
                             % (amb, fitted, 100.0 * amb / fitted))
            else:
                k = cfg["ratio_cut"]
                rm = self.main.ratio_maps.get(cfg.get("ratio_metric", "height"))
                nr = int(np.isfinite(rm).sum()) if rm is not None else 0
                lines.append("Thickness from I(2D)/I(G) (%s), cutoffs %.2f / %.2f / "
                             "%.2f: ratio available for %d of %d fitted pixels"
                             % (cfg.get("ratio_metric", "height"), k["1_2"],
                                k["2_3"], k["3_thick"], nr, fitted))
                if cfg.get("thick_policy") == "assume":
                    lines.append("≥4L pixels assumed to be %dL for the stacking call"
                                 % int(cfg.get("thick_assume", 5)))
            res = self.main.mapfile.spectral_resolution if self.main.mapfile else None
            if res and res > 8.0 and cfg["technique"] != "asym":
                lines.append("⚠ Spectral resolution %.1f cm⁻¹: the 2D widths are "
                             "instrument-broadened, so ABA/ABC calls against the "
                             "Bouhafs table are unreliable. Use this map for "
                             "thickness and a 1800 gr/mm map for stacking." % res)
        rejected = total - fitted
        if rejected:
            lines.append("")
            lines.append("Rejected before classification (%d pixels):" % rejected)
            for why, n in sorted((self.main.fit_tally or {}).items(),
                                 key=lambda kv: -kv[1]):
                lines.append("   %5d   %s" % (n, why))
        lines.append("")
        for lab in sorted(labels, key=lambda s: (s == "Unclassified", s)):
            n = labels[lab]
            lines.append("%-16s %6d   %5.1f %%" % (lab, n, 100.0 * n / total))
        self.stats.setPlainText("\n".join(lines))

    def show_histograms(self):
        """Show where the thresholds actually fall relative to the data.

        Thresholds copied from a paper describe that paper's samples. This is
        how you check whether they describe yours: a bimodal FWHM histogram
        with the boundary sitting in the valley means the cut is doing real
        work; a single broad hump means it is slicing one population in half.
        """
        if self.main.fwhm_map is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Load and fit a map first (tab 1).")
            return
        self._pull_config()
        cfg = self.main.cfg
        fw = self.main.fwhm_map[np.isfinite(self.main.fwhm_map)]
        am = self.main.asym_maps[cfg["asym_metric"]]
        am = am[np.isfinite(am)]
        if fw.size == 0:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No successful fits.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Threshold histograms")
        dlg.resize(900, 460)
        lay = QtWidgets.QVBoxLayout(dlg)
        cv = Canvas(figsize=(9, 4))
        cv.figure.clear()
        ax1 = cv.figure.add_subplot(121)
        ax2 = cv.figure.add_subplot(122)

        ax1.hist(fw, bins=max(10, min(60, fw.size // 8)),
                 color="#607d8b", edgecolor="white", linewidth=0.4)
        if cfg["layer_mode"] == "manual":
            shown = [int(cfg["manual_layer"])]
        elif cfg["layer_mode"] == "map" and self.main.layer_by_spec is not None:
            shown = sorted(n for n in set(self.main.layer_by_spec.tolist())
                           if n in cfg["fwhm"]) or sorted(cfg["fwhm"])
        else:
            shown = sorted(cfg["fwhm"])
        top = ax1.get_ylim()[1]
        for layer in shown:
            (a0, a1), (c0, c1) = windows_for(layer, cfg)
            for (lo, hi), col, key in (((a0, a1), "#1e88e5", "ABA"),
                                       ((c0, c1), "#e53935", "ABC")):
                if hi < fw.min() - 5 or lo > fw.max() + 5:
                    continue
                ax1.axvspan(lo, hi, color=col, alpha=0.16, lw=0)
                ax1.axvline(cfg["fwhm"][layer][key], color=col, lw=1.0, ls="--")
            if len(shown) == 1:
                ax1.text(cfg["fwhm"][layer]["ABA"], top * 0.96, " ABA",
                         color="#1e88e5", fontsize=8, va="top")
                ax1.text(cfg["fwhm"][layer]["ABC"], top * 0.96, " ABC",
                         color="#e53935", fontsize=8, va="top")
        ax1.set_xlabel("FWHM(2D)  [cm$^{-1}$]")
        ax1.set_ylabel("pixels")
        ax1.set_title("Width, with the ABA (blue) and ABC (red) windows"
                      if len(shown) == 1 else
                      "Width, with every layer's ABA/ABC windows", fontsize=9)

        if am.size:
            ax2.hist(am, bins=max(10, min(60, am.size // 8)),
                     color="#8d6e63", edgecolor="white", linewidth=0.4)
            ax2.axvline(cfg["asym_threshold"], color="#d81b60", lw=1.4, ls="--",
                        label="threshold %.3f" % cfg["asym_threshold"])
            ax2.legend(fontsize=8)
        ax2.set_xlabel("asymmetry (%s)" % cfg["asym_metric"])
        ax2.set_ylabel("pixels")
        ax2.set_title("Shape asymmetry, with the ABC cut", fontsize=10)

        lay.addWidget(cv)
        msg = QtWidgets.QLabel(
            "Widen each ± until the shaded windows cover the humps your data "
            "actually has, and no further. Windows that between them cover the "
            "whole axis classify everything and test nothing; windows that "
            "cover neither hump mean the reference values do not describe this "
            "sample. A window edge cutting through the middle of a hump is "
            "splitting one population in two, and that map will look decisive "
            "without meaning much.")
        msg.setWordWrap(True)
        msg.setStyleSheet("color:#555; font-size:11px;")
        lay.addWidget(msg)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        cv.draw()
        dlg.exec_()

    # -- drawing ------------------------------------------------------------
    def _placeholder(self):
        self.canvas.clear()
        self.canvas.ax.text(0.5, 0.5, "No classification yet",
                            ha="center", va="center", color="#999")
        self.canvas.ax.set_axis_off()
        self.canvas.draw()

    def refresh_map(self):
        mf = self.main.mapfile
        if mf is None or self.layers is None:
            self._placeholder()
            return
        cv = self.canvas
        cv.clear()
        ax = cv.ax
        rgba = class_rgba(self.layers, self.calls, self.main.cfg)
        ax.imshow(rgba, origin="lower", extent=mf.extent,
                  interpolation="nearest", aspect="equal", zorder=1)

        opt = self.main.optical
        if opt is not None and self.chk_optical.isChecked() \
                and self.main.optical_image is not None:
            ax.imshow(np.ma.masked_invalid(self.main.optical_image),
                      origin="lower", extent=opt.extent, cmap="gray",
                      alpha=self.main.cfg["optical_alpha"],
                      interpolation="bilinear", aspect="equal", zorder=2)
        if self.chk_grid.isChecked():
            MapExplorerTab._draw_grid(ax, mf)

        present = set()
        ny, nx = self.layers.shape
        for r in range(ny):
            for c in range(nx):
                present.add((int(self.layers[r, c]), self.calls[r, c]))
        handles = []
        for lay, call in sorted(present, key=lambda t: (t[0] < 0, t[0], str(t[1]))):
            if call is None or (lay < 0 and call != "N/A"):
                continue
            col = (na_color(lay, self.main.cfg) if call == "N/A"
                   else self.main.cfg["colors"][lay][call])
            handles.append(Patch(facecolor=col, edgecolor="#555",
                                 label=label_of(lay, call)))
        handles.append(Patch(facecolor=self.main.cfg["unclassified_color"],
                             edgecolor="#555", label="Unclassified"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=True, title="Class")
        ax.set_xlabel("X  [µm]")
        ax.set_ylabel("Y  [µm]")
        ax.set_title("Stacking-order classification")
        ax.set_anchor("C")
        frame_axes(ax, mf, self.main.cfg,
                   self.chk_optical.isChecked() and self.main.optical is not None)
        orient_axes(ax, self.main.cfg)
        cv.draw()

    # -- export -------------------------------------------------------------
    def export_png(self):
        if self.layers is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Nothing to export yet.")
            return
        path = export_figure(self.canvas.figure, self,
                             self.main.suggest_name("_classified.pdf"))
        if path:
            self.main.status("Saved %s" % path)

    def export_csv(self):
        if self.layers is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save classification results",
            self.main.suggest_name("_classification.csv"), "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        mf = self.main.mapfile
        ny, nx = self.layers.shape
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "col", "x_um", "y_um", "fwhm_2d_cm-1",
                        "fwhm_err_cm-1", "center_cm-1", "asym_hwhm", "asym_area",
                        "asym_residual", "snr", "spikes_removed",
                        "g_height", "twod_height", "ratio_2d_g", "n_layers",
                        "stacking", "label", "n_candidate_layers",
                        "unclassified_reason", "fit_note"])
            for r in range(ny):
                for c in range(nx):
                    i = mf.index_map[r, c]
                    res = self.main.fits[i] if i >= 0 and self.main.fits else {}
                    gr = (self.main.g_fits[i] if i >= 0 and self.main.g_fits
                          else {})
                    lay = int(self.layers[r, c])
                    call = self.calls[r, c]
                    w.writerow([
                        r, c,
                        "" if i < 0 else "%.4f" % mf.x[i],
                        "" if i < 0 else "%.4f" % mf.y[i],
                        _fmt(res.get("fwhm")), _fmt(res.get("fwhm_err"), 2),
                        _fmt(res.get("center")),
                        _fmt(res.get("asym_hwhm"), 4),
                        _fmt(res.get("asym_area"), 4),
                        _fmt(res.get("asym_residual"), 4),
                        _fmt(res.get("snr"), 2), res.get("spikes", 0),
                        _fmt(gr.get("height") if gr.get("ok") else None, 2),
                        _fmt(res.get("height") if res.get("ok") else None, 2),
                        _fmt(pixel_ratio(res, gr, self.main.cfg.get("ratio_metric", "height")), 3),
                        (">=4" if lay == LAYER_THICK else ("" if lay < 0 else lay)),
                        call or "",
                        label_of(lay, call),
                        int(self.ncands[r, c]),
                        self.reasons[r, c] or "",
                        res.get("note", ""),
                    ])
        self.main.status("Saved %s" % path)


def _fmt(v, nd=3) -> str:
    return "" if v is None or not np.isfinite(v) else ("%.*f" % (nd, v))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

TRACE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


class SpectraTab(QtWidgets.QWidget):
    """Pick grid cells, compare their spectra stacked, fit Lorentzians.

    Selections are kept by spectrum index, so they survive a change of map
    orientation; they are cleared when a new map is loaded.
    """

    PRESETS = [("2D band", (2550.0, 2850.0)), ("G band", (1500.0, 1680.0)),
               ("D band", (1270.0, 1430.0)), ("Full spectrum", None), ("Custom", None)]
    MAX_SPECTRA = 12

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.items: List[dict] = []
        self._build()

    # -- UI ------------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)

        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(self.split)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        vr = QtWidgets.QHBoxLayout()
        vr.addWidget(QtWidgets.QLabel("Show"))
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.addItems(["FWHM(2D)", "Classification", "Layer map (N)"])
        self.cmb_view.setToolTip(
            "FWHM(2D) — the fitted 2D width, no classification\n"
            "Classification — the result from tab 2, whichever layer mode "
            "produced it (one N, Auto, or layer regions)\n"
            "Layer map (N) — the per-region layer counts from the regions dialog")
        self.cmb_view.currentIndexChanged.connect(self.refresh_mini)
        vr.addWidget(self.cmb_view, 1)
        lv.addLayout(vr)
        orow = QtWidgets.QHBoxLayout()
        self.chk_opt = QtWidgets.QCheckBox("Optical image")
        self.chk_opt.setChecked(True)
        self.chk_opt.toggled.connect(self.refresh_mini)
        orow.addWidget(self.chk_opt)
        orow.addWidget(QtWidgets.QLabel("map opacity"))
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(10, 100)
        self.sld_alpha.setValue(75)
        self.sld_alpha.setToolTip("Opacity of the map over the optical image")
        self.sld_alpha.valueChanged.connect(self.refresh_mini)
        orow.addWidget(self.sld_alpha, 1)
        lv.addLayout(orow)
        hint = QtWidgets.QLabel("Click grid cells to add or remove their spectra")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        self.mini = Canvas(figsize=(4.4, 4.6))
        self.mini.setMinimumWidth(240)
        self.mini.canvas.mpl_connect("button_press_event", self._mini_click)
        self.mini.setMinimumHeight(360)
        lv.addWidget(self.mini, 5)
        self.lst = QtWidgets.QListWidget()
        self.lst.setMaximumHeight(170)
        self.lst.itemChanged.connect(self._item_toggled)
        lv.addWidget(self.lst, 1)
        br = QtWidgets.QHBoxLayout()
        b_rm = QtWidgets.QPushButton("Remove selected")
        b_rm.clicked.connect(self._remove_selected)
        b_cl = QtWidgets.QPushButton("Clear all")
        b_cl.clicked.connect(self.clear)
        br.addWidget(b_rm)
        br.addWidget(b_cl)
        lv.addLayout(br)
        self.split.addWidget(left)

        right = QtWidgets.QVBoxLayout()
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(QtWidgets.QLabel("Range"))
        self.cmb_range = QtWidgets.QComboBox()
        self.cmb_range.addItems([n for n, _ in self.PRESETS])
        self.cmb_range.currentIndexChanged.connect(self._preset_changed)
        r1.addWidget(self.cmb_range)
        r1.addWidget(QtWidgets.QLabel("from"))
        self.spn_lo = QtWidgets.QDoubleSpinBox()
        r1.addWidget(self.spn_lo)
        r1.addWidget(QtWidgets.QLabel("to"))
        self.spn_hi = QtWidgets.QDoubleSpinBox()
        r1.addWidget(self.spn_hi)
        for sp, v in ((self.spn_lo, 2550.0), (self.spn_hi, 2850.0)):
            sp.setRange(0, 10000)
            sp.setDecimals(0)
            sp.setSingleStep(10)
            sp.setValue(v)
            sp.setSuffix(" cm⁻¹")
            sp.setFixedWidth(100)
            sp.valueChanged.connect(self._range_edited)
        self._lbl_stack = QtWidgets.QLabel("Stack offset")
        self.sld_off = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_off.setRange(0, 150)
        self.sld_off.setValue(60)
        self.sld_off.setFixedWidth(120)
        self.sld_off.valueChanged.connect(self.redraw)
        self.chk_norm = QtWidgets.QCheckBox("Normalise")
        self.chk_norm.setToolTip("Scale each spectrum to its own maximum within the range")
        self.chk_norm.setChecked(True)
        self.chk_norm.toggled.connect(self.redraw)
        r1.addStretch(1)
        right.addLayout(r1)

        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("Lorentzian peaks"))
        self.spn_n = QtWidgets.QSpinBox()
        self.spn_n.setRange(1, 4)
        self.spn_n.valueChanged.connect(self._n_changed)
        r2.addWidget(self.spn_n)
        self.btn_suggest = QtWidgets.QPushButton("Suggest peak count")
        self.btn_suggest.setToolTip(
            "Fewest peaks within 10 BIC units of the best fit, rejecting any fit "
            "with a component narrower than the spectral resolution")
        self.btn_suggest.clicked.connect(self.suggest)
        r2.addWidget(self.btn_suggest)
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.setStyleSheet("font-weight:bold;")
        self.btn_fit.clicked.connect(self.fit_all)
        r2.addWidget(self.btn_fit)
        self.chk_auto = QtWidgets.QCheckBox("Refit on change")
        self.chk_auto.setChecked(True)
        r2.addWidget(self.chk_auto)
        r2.addStretch(1)
        right.addLayout(r2)
        r2 = QtWidgets.QHBoxLayout()            # second line: display and export
        self.chk_fit = QtWidgets.QCheckBox("Show fits")
        self.chk_fit.setChecked(True)
        self.chk_fit.toggled.connect(self.redraw)
        self.chk_comp = QtWidgets.QCheckBox("Show components")
        self.chk_comp.setChecked(True)
        self.chk_comp.toggled.connect(self.redraw)
        r2.addWidget(self._lbl_stack)
        r2.addWidget(self.sld_off)
        r2.addWidget(self.chk_norm)
        r2.addSpacing(12)
        r2.addWidget(self.chk_fit)
        r2.addWidget(self.chk_comp)
        r2.addStretch(1)
        self.btn_csv = QtWidgets.QPushButton("Export CSV…")
        self.btn_csv.clicked.connect(self.export_csv)
        r2.addWidget(self.btn_csv)
        right.addLayout(r2)

        self.plot = Canvas(figsize=(8, 5))
        right.addWidget(self.plot, 3)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("font-size:11px; color:#555;")
        right.addWidget(self.lbl)
        self.tbl = QtWidgets.QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(["Spectrum", "Peak", "Centre", "FWHM",
                                            "± FWHM", "Height", "Area", "Band FWHM", "R²"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        right.addWidget(self.tbl, 2)
        rightw = QtWidgets.QWidget()
        rightw.setLayout(right)
        self.split.addWidget(rightw)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 3)
        self.split.setSizes([420, 1100])
        self.refresh_mini()
        self.redraw()

    # -- helpers ------------------------------------------------------------
    def resolution(self) -> float:
        mf = self.main.mapfile
        return float((mf.spectral_resolution if mf is not None else None)
                     or self.main.cfg.get("min_fwhm", 5.0))

    def rng(self) -> Tuple[float, float]:
        lo, hi = self.spn_lo.value(), self.spn_hi.value()
        return (lo, hi) if hi > lo else (lo, lo + 10.0)

    def _name(self, idx: int) -> str:
        mf = self.main.mapfile
        r, c = np.argwhere(mf.index_map == idx)[0]
        name = "R%dC%d (%.1f, %.1f) µm" % (r, c, mf.x[idx], mf.y[idx])
        tc = self.main.tab_class
        if tc.layers is not None:
            name += "  " + label_of(int(tc.layers[r, c]), tc.calls[r, c])
        return name

    # -- selection ---------------------------------------------------------------
    def add_spectrum(self, idx: int):
        if any(it["idx"] == idx for it in self.items):
            return
        if len(self.items) >= self.MAX_SPECTRA:
            self.main.status("The comparison holds at most %d spectra." % self.MAX_SPECTRA)
            return
        used = {it["color"] for it in self.items}
        color = next((c for c in TRACE_COLORS if c not in used), TRACE_COLORS[0])
        self.items.append({"idx": int(idx), "color": color, "visible": True, "fit": None})
        if self.chk_auto.isChecked():
            self._fit_item(self.items[-1])
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _toggle(self, idx: int):
        for k, it in enumerate(self.items):
            if it["idx"] == idx:
                self.items.pop(k)
                self._rebuild_list()
                self.refresh_mini()
                self.redraw()
                return
        self.add_spectrum(idx)

    def clear(self):
        self.items = []
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.lst.selectedIndexes()}, reverse=True)
        for r in rows:
            self.items.pop(r)
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    def _rebuild_list(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        for it in self.items:
            li = QtWidgets.QListWidgetItem(self._name(it["idx"]))
            li.setFlags(li.flags() | QtCore.Qt.ItemIsUserCheckable)
            li.setCheckState(QtCore.Qt.Checked if it["visible"] else QtCore.Qt.Unchecked)
            pm = QtGui.QPixmap(14, 14)
            pm.fill(QtGui.QColor(it["color"]))
            li.setIcon(QtGui.QIcon(pm))
            self.lst.addItem(li)
        self.lst.blockSignals(False)

    def _item_toggled(self, li):
        r = self.lst.row(li)
        if 0 <= r < len(self.items):
            self.items[r]["visible"] = li.checkState() == QtCore.Qt.Checked
            self.refresh_mini()
            self.redraw()

    def _mini_click(self, event):
        mf = self.main.mapfile
        if mf is None or event.inaxes is None or event.xdata is None:
            return
        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        col = int(np.floor((event.xdata - left) / (right - left) * nx))
        row = int(np.floor((event.ydata - bottom) / (top - bottom) * ny))
        if 0 <= col < nx and 0 <= row < ny and mf.index_map[row, col] >= 0:
            self._toggle(int(mf.index_map[row, col]))

    def reset(self):
        self.items = []
        self._rebuild_list()
        self.refresh_mini()
        self.redraw()

    # -- range ------------------------------------------------------------------
    def _preset_changed(self):
        name = self.cmb_range.currentText()
        rngd = dict(self.PRESETS).get(name)
        mf = self.main.mapfile
        if name == "Full spectrum" and mf is not None:
            rngd = (float(mf.axis[0]), float(mf.axis[-1]))
        if rngd is not None:
            for sp, v in ((self.spn_lo, rngd[0]), (self.spn_hi, rngd[1])):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
            self._changed()

    def _range_edited(self):
        if self.cmb_range.currentText() != "Custom":
            self.cmb_range.blockSignals(True)
            self.cmb_range.setCurrentText("Custom")
            self.cmb_range.blockSignals(False)
        self._changed()

    def _n_changed(self):
        self._changed()

    def _changed(self):
        if self.chk_auto.isChecked():
            self.fit_all(redraw=False)
        self.redraw()

    # -- fitting ----------------------------------------------------------------
    def _fit_item(self, it):
        mf = self.main.mapfile
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        it["fit"] = ((lo, hi, self.spn_n.value()),
                     fit_lorentzians(mf.axis[m], mf.data[it["idx"]][m], self.spn_n.value()))

    def fit_all(self, redraw=True):
        if self.main.mapfile is None:
            return
        for it in self.items:
            if it["visible"]:
                self._fit_item(it)
        self._fill_table()
        if redraw:
            self.redraw()

    def suggest(self):
        mf = self.main.mapfile
        if mf is None or not self.items:
            return
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        res = self.resolution()
        picks = []
        for it in self.items:
            if it["visible"]:
                n, _ = suggest_peak_count(mf.axis[m], mf.data[it["idx"]][m], res)
                picks.append((it, n))
        if not picks:
            return
        counts = [n for _, n in picks]
        best = max(set(counts), key=counts.count)
        self.spn_n.setValue(best)
        self.lbl.setText("Suggested peak counts (resolution %.1f cm⁻¹): %s  →  using %d."
                         % (res, ",  ".join("%s: %d" % (self._name(it["idx"]).split()[0], n)
                                            for it, n in picks), best))
        self.fit_all()

    # -- drawing ---------------------------------------------------------------
    def refresh_mini(self):
        cv = self.mini
        cv.clear()
        ax = cv.ax
        mf = self.main.mapfile
        if mf is None or mf.layout != "spectra":
            ax.text(0.5, 0.5, "No map loaded", ha="center", va="center", color="#999")
            ax.set_axis_off()
            cv.draw()
            return
        cfg = self.main.cfg
        view = self.cmb_view.currentText()
        opt = self.main.optical
        with_opt = (self.chk_opt.isChecked() and opt is not None
                    and self.main.optical_image is not None)
        alpha = self.sld_alpha.value() / 100.0 if with_opt else 1.0
        self.sld_alpha.setEnabled(with_opt)
        if with_opt:
            ax.imshow(np.ma.masked_invalid(self.main.optical_image), origin="lower",
                      extent=opt.extent, cmap="gray", aspect="equal", zorder=0)
        handles, note, title = [], "", view

        if view == "Classification":
            tc = self.main.tab_class
            if tc.layers is None or tc.layers.shape != mf.shape:
                note = "No classification yet —\nrun 'Classify map' in tab 2"
            else:
                rgba = class_rgba(tc.layers, tc.calls, cfg)
                rgba[..., 3] *= alpha
                ax.imshow(rgba, origin="lower", extent=mf.extent,
                          interpolation="nearest", aspect="equal", zorder=1)
                present = sorted({(int(tc.layers[r, c]), tc.calls[r, c])
                                  for r in range(mf.shape[0]) for c in range(mf.shape[1])},
                                 key=lambda t: (t[0] < 0, t[0], str(t[1])))
                for lay, call in present:
                    if call is None or (lay < 0 and call != "N/A"):
                        continue
                    col = (na_color(lay, cfg) if call == "N/A"
                           else cfg["colors"][lay][call])
                    handles.append(Patch(facecolor=col, edgecolor="#555",
                                         label=label_of(lay, call)))
                handles.append(Patch(facecolor=cfg["unclassified_color"],
                                     edgecolor="#555", label="Unclassified"))
                mode = {"manual": "Manual %dL" % int(cfg.get("manual_layer", 3)),
                        "auto": "Auto, I(2D)/I(G)",
                        "map": "layer regions"}.get(cfg.get("layer_mode"), "")
                title = "Classification (%s)" % mode
        elif view == "Layer map (N)":
            g = self.main.layer_grid() if self.main.layer_by_spec is not None else None
            if g is None or not (g > 0).any():
                note = "No layer map yet —\ndefine layer regions in tab 2"
            else:
                rgba = np.zeros(g.shape + (4,))
                for n in sorted(set(g[g > 0].tolist())):
                    rgba[g == n] = mcolors.to_rgba(n_color(n, cfg), alpha)
                    handles.append(Patch(facecolor=n_color(n, cfg), edgecolor="#555",
                                         label="%dL" % n))
                ax.imshow(rgba, origin="lower", extent=mf.extent,
                          interpolation="nearest", aspect="equal", zorder=1)
                title = "Layer map (N per region)"
        if view == "FWHM(2D)" or note:
            data = (self.main.fwhm_map if self.main.fwhm_map is not None
                    else mf.greyscale())
            im = ax.imshow(np.ma.masked_invalid(data), origin="lower", extent=mf.extent,
                           cmap=cfg.get("cmap", "viridis"), interpolation="nearest",
                           aspect="equal", alpha=alpha, zorder=1)
            if not note:
                cb = side_colorbar(cv.figure, ax, im, size="5%", pad=0.05)
                cb.set_label("FWHM(2D) [cm$^{-1}$]", fontsize=7)
                cb.ax.tick_params(labelsize=7)
                title = "FWHM(2D)"
        if note:
            ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#b71c1c", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b71c1c", alpha=0.9))

        left, right, bottom, top = mf.extent
        ny, nx = mf.shape
        dx, dy = (right - left) / nx, (top - bottom) / ny
        for xv in np.linspace(left, right, nx + 1):
            ax.axvline(xv, color="w", lw=0.3, alpha=0.5, zorder=2)
        for yv in np.linspace(bottom, top, ny + 1):
            ax.axhline(yv, color="w", lw=0.3, alpha=0.5, zorder=2)
        for it in self.items:
            r, c = np.argwhere(mf.index_map == it["idx"])[0]
            ax.add_patch(mpatches.Rectangle((left + c * dx, bottom + r * dy), dx, dy,
                                            fill=False, edgecolor=it["color"],
                                            lw=2.6 if it["visible"] else 1.0,
                                            ls="-" if it["visible"] else ":", zorder=4))
        pad = 0.08 * max(right - left, top - bottom) if with_opt else 0.0
        ax.set_xlim(left - pad, right + pad)
        ax.set_ylim(bottom - pad, top + pad)
        ax.set_title(title + " — click cells", fontsize=9)
        ax.tick_params(labelsize=7)
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="upper center",
                      bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(handles)),
                      frameon=False, handlelength=1.2, columnspacing=0.8)
        orient_axes(ax, cfg)
        cv.draw()

    def redraw(self):
        cv = self.plot
        cv.clear()
        ax = cv.ax
        mf = self.main.mapfile
        vis = [it for it in self.items if it["visible"]]
        if mf is None or not vis:
            ax.text(0.5, 0.5, "Select grid cells on the left to compare their spectra",
                    ha="center", va="center", color="#999", transform=ax.transAxes)
            ax.set_axis_off()
            cv.draw()
            self._fill_table()
            return
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        x = mf.axis[m]
        norm = self.chk_norm.isChecked()
        scaled = []
        for it in vis:
            y = mf.data[it["idx"]][m]
            ymin = float(np.nanpercentile(y, 2))
            ymax = float(np.nanmax(y))
            k = (ymax - ymin) if norm and ymax > ymin else 1.0
            scaled.append((it, y, ymin if norm else 0.0, k))
        span = max(((np.nanmax(y) - (b if norm else np.nanmin(y))) / k) for _, y, b, k in scaled)
        step = self.sld_off.value() / 100.0 * span
        for j, (it, y, b, k) in enumerate(scaled):
            off = j * step
            ax.plot(x, (y - b) / k + off, color=it["color"], lw=0.9, label=self._name(it["idx"]))
            fr = it.get("fit")
            txt = ""
            if fr is not None and fr[0] == (lo, hi, self.spn_n.value()) and fr[1]["ok"]:
                res = fr[1]
                fx, fy = res["curve"]
                if self.chk_fit.isChecked():
                    ax.plot(fx, (fy - b) / k + off, color="k", lw=1.1, ls="--")
                if self.chk_comp.isChecked() and len(res["peaks"]) > 1:
                    sl, ic, xm = res["baseline"]
                    basel = sl * (fx - xm) + ic
                    for (h, c, w) in res["components"]:
                        comp = h * (w / 2) ** 2 / ((fx - c) ** 2 + (w / 2) ** 2)
                        ax.fill_between(fx, (basel - b) / k + off, (basel + comp - b) / k + off,
                                        color=it["color"], alpha=0.18, lw=0)
                txt = "FWHM " + " / ".join("%.1f" % p["fwhm"] for p in res["peaks"])
                if len(res["peaks"]) > 1:
                    txt += "  (band %.1f)" % res["band_fwhm"]
            ax.text(x[-1], float(np.nanmedian((y[-8:] - b) / k)) + off, "  " + txt,
                    color=it["color"], fontsize=8, va="bottom", ha="right")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Raman shift  [cm$^{-1}$]")
        ax.set_ylabel("Intensity (normalised, stacked)" if norm else "Intensity [counts], stacked")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.set_title("%d spectra · %d Lorentzian peak%s · resolution %.1f cm$^{-1}$"
                     % (len(vis), self.spn_n.value(), "" if self.spn_n.value() == 1 else "s",
                        self.resolution()), fontsize=9)
        cv.draw()
        self._fill_table()

    def _fill_table(self):
        lo, hi = self.rng()
        res_lim = self.resolution()
        rows = []
        for it in self.items:
            fr = it.get("fit")
            if not it["visible"] or fr is None or fr[0] != (lo, hi, self.spn_n.value()):
                continue
            r = fr[1]
            if not r["ok"]:
                rows.append((it, None, r["note"], r))
                continue
            for k, pk in enumerate(r["peaks"]):
                rows.append((it, k, pk, r))
        self.tbl.setRowCount(len(rows))
        for i, (it, k, pk, r) in enumerate(rows):
            name = self._name(it["idx"]).split("  ")[0]
            if k is None:
                vals = [name, "—", pk, "", "", "", "", "", ""]
            else:
                narrow = pk["fwhm"] < res_lim
                vals = [name, "%d" % (k + 1), "%.1f" % pk["center"],
                        "%.1f%s" % (pk["fwhm"], " ⚠" if narrow else ""),
                        "%.1f" % pk["err_fwhm"] if np.isfinite(pk["err_fwhm"]) else "—",
                        "%.1f" % pk["height"], "%.0f" % pk["area"],
                        "%.1f" % r["band_fwhm"] if k == 0 else "", "%.4f" % r["r2"] if k == 0 else ""]
            for c, v in enumerate(vals):
                ti = QtWidgets.QTableWidgetItem(v)
                ti.setTextAlignment(QtCore.Qt.AlignCenter)
                if c == 0:
                    ti.setForeground(QtGui.QBrush(QtGui.QColor(it["color"])))
                if c == 3 and k is not None and pk["fwhm"] < res_lim:
                    ti.setForeground(QtGui.QBrush(QtGui.QColor("#b71c1c")))
                    ti.setToolTip("Narrower than the %.1f cm⁻¹ spectral resolution — "
                                  "this component is fitting noise, not a band. "
                                  "Try fewer peaks." % res_lim)
                self.tbl.setItem(i, c, ti)

    # -- export -------------------------------------------------------------------
    def export_csv(self):
        mf = self.main.mapfile
        vis = [it for it in self.items if it["visible"]]
        if mf is None or not vis:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export spectra and fits", self.main.suggest_name("_spectra.csv"),
            "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        lo, hi = self.rng()
        m = (mf.axis >= lo) & (mf.axis <= hi)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["raman_shift_cm-1"] + [self._name(it["idx"]) for it in vis])
            for j, xv in enumerate(mf.axis[m]):
                w.writerow(["%.3f" % xv] + ["%.3f" % mf.data[it["idx"]][m][j] for it in vis])
        fpath = path[:-4] + "_fits.csv"
        with open(fpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["spectrum", "x_um", "y_um", "range_lo", "range_hi", "n_peaks", "peak",
                        "centre_cm-1", "fwhm_cm-1", "fwhm_err_cm-1", "height", "area",
                        "band_fwhm_cm-1", "r2", "bic"])
            for it in vis:
                fr = it.get("fit")
                if fr is None or not fr[1]["ok"]:
                    continue
                (flo, fhi, n), r = fr
                for k, pk in enumerate(r["peaks"]):
                    w.writerow([self._name(it["idx"]), "%.3f" % mf.x[it["idx"]],
                                "%.3f" % mf.y[it["idx"]], flo, fhi, n, k + 1,
                                "%.2f" % pk["center"], "%.2f" % pk["fwhm"],
                                "%.2f" % pk["err_fwhm"], "%.2f" % pk["height"],
                                "%.1f" % pk["area"], "%.2f" % r["band_fwhm"],
                                "%.5f" % r["r2"], "%.1f" % r["bic"]])
        self.main.status("Saved %s and %s" % (os.path.basename(path), os.path.basename(fpath)))


# ---------------------------------------------------------------------------
# Tab 4 - Stacked spectra (publication figure, no fitting)
# ---------------------------------------------------------------------------

STACK_COLORS = ["#000000", "#e8402a", "#1f5fa8", "#2a8c4a", "#8a5a00",
                "#7b3fa0", "#17becf", "#7f7f7f"]
_NUM = r"(\d+(?:\.\d+)?)"


def parse_windows(text: str) -> List[Tuple[float, float]]:
    """'1450-1650, 2550-2800' -> [(1450, 1650), (2550, 2800)]."""
    out = []
    for lo, hi in re.findall(_NUM + r"\s*[-–:]\s*" + _NUM, text):
        lo, hi = float(lo), float(hi)
        if hi > lo:
            out.append((lo, hi))
    return out


def parse_band_labels(text: str) -> List[Tuple[str, float]]:
    """'G 1583, 2D 2690' -> [('G', 1583.0), ('2D', 2690.0)]."""
    out = []
    for piece in re.split(r"[,;]", text):
        m = re.match(r"\s*(.+?)\s*@?\s*" + _NUM + r"\s*$", piece)
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out


def _tick_step(span: float) -> float:
    for s in (10, 25, 50, 100, 200, 500, 1000):
        if span / s <= 3:
            return s
    return 1000.0


def draw_stack(fig, traces: List[dict], windows: List[Tuple[float, float]],
               step: float = 1.15, normalise: bool = True,
               trace_labels: bool = False,
               band_labels: List[Tuple[str, float]] = (), title: str = ""):
    """Stacked spectra in one closed frame, the x axis broken between
    spectral windows with // marks, no y ticks. Every window has the same
    cm⁻¹ per centimetre. Raw data only: each trace is shifted by its 5th
    percentile inside the windows and scaled to its maximum there (or, with
    normalise off, all traces by one common factor)."""
    fig.clear()
    if not traces or not windows:
        fig.text(0.5, 0.5, "No spectra" if windows else "No ranges",
                 ha="center", va="center", color="#999", fontsize=8)
        return
    spans = [hi - lo for lo, hi in windows]
    axs = np.atleast_1d(fig.subplots(
        1, len(windows), sharey=True,
        gridspec_kw={"width_ratios": spans, "wspace": 0.04}))

    def in_windows(wn):
        m = np.zeros(wn.shape, bool)
        for lo, hi in windows:
            m |= (wn >= lo) & (wn <= hi)
        return m

    scale = []
    for t in traces:
        m = in_windows(t["wn"])
        if m.sum() < 2:
            scale.append(None)
            continue
        base = float(np.nanpercentile(t["inten"][m], 5))
        scale.append((base, float(np.nanmax(t["inten"][m])) - base))
    common = max([s[1] for s in scale if s] or [1.0]) or 1.0

    n = len(traces)
    lo0, hi0 = windows[0]
    for i, (t, sc) in enumerate(zip(traces, scale)):
        if sc is None:
            continue
        off = (n - 1 - i) * step               # first trace at the top
        y = (t["inten"] - sc[0]) / ((sc[1] or 1.0) if normalise else common) + off
        for ax, (lo, hi) in zip(axs, windows):
            s = (t["wn"] >= lo) & (t["wn"] <= hi)
            ax.plot(t["wn"][s], y[s], color=t["color"], lw=0.6)
        if trace_labels:
            left = (t["wn"] >= lo0) & (t["wn"] <= lo0 + 0.3 * (hi0 - lo0))
            ly = float(np.nanmax(y[left])) + 0.06 if left.any() else off + 0.25
            axs[0].text(lo0 + 0.02 * (hi0 - lo0), ly, t["label"], color=t["color"],
                        fontsize=6.5, style="italic", va="bottom")

    top = (n - 1) * step + 1.06
    ymax = top + (0.3 if band_labels else 0.08) + (0.32 if title else 0.0)
    axs[0].set_ylim(-0.12, ymax)
    for text, pos in band_labels:
        for ax, (lo, hi) in zip(axs, windows):
            if lo <= pos <= hi:
                ax.text(pos, top, text, ha="center", va="bottom", fontsize=8)
    if title:
        axs[0].text(lo0 + 0.02 * (hi0 - lo0), ymax - 0.04, title, style="italic",
                    fontsize=8, va="top")

    for k, (ax, (lo, hi)) in enumerate(zip(axs, windows)):
        ax.set_xlim(lo, hi)
        st = _tick_step(hi - lo)
        last = hi if k == len(windows) - 1 else hi - st / 4   # no label on a break
        ax.set_xticks(np.arange(np.ceil(lo / st) * st, last + 1e-9, st))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(st / 2))
        ax.tick_params(axis="x", direction="out", labelsize=7, width=0.6,
                       length=3, top=False)
        ax.tick_params(axis="x", which="minor", direction="out", width=0.5,
                       length=1.6, top=False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if k > 0:
            ax.spines["left"].set_visible(False)
        if k < len(windows) - 1:
            ax.spines["right"].set_visible(False)
    for a, b in zip(axs[:-1], axs[1:]):                # // marks on each break
        for ax, xb in ((a, 1), (b, 0)):
            ax.plot([xb, xb], [0, 1], transform=ax.transAxes, ls="none",
                    marker=[(-1, -1.6), (1, 1.6)], markersize=5,
                    markeredgewidth=0.6, color="k", clip_on=False)
    axs[0].set_ylabel("Intensity (arb. u.)", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.12)
    # One x label centred under all windows: an empty frameless axes over
    # the same area carries it, so it sits just below the tick labels.
    l = axs[0].get_position().x0
    r = axs[-1].get_position().x1
    b, t = axs[0].get_position().y0, axs[0].get_position().y1
    big = fig.add_axes([l, b, r - l, t - b], frameon=False, zorder=-1)
    big.set_xticks([])
    big.set_yticks([])
    big.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8, labelpad=13)


class StackTab(QtWidgets.QWidget):
    """Stack spectra from files or from the loaded map into one figure and
    export it. Nothing here fits anything."""

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.items: List[dict] = []          # {spec, color, visible}
        self._build()

    # -- UI -----------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()

        self.tbl = QtWidgets.QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Label", "", "Source"])
        self.tbl.verticalHeader().setVisible(False)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tbl.setColumnWidth(1, 26)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.itemChanged.connect(self._item_changed)
        self.tbl.cellClicked.connect(self._cell_clicked)
        left.addWidget(self.tbl, 1)

        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Add files…", self.add_files),
                         ("Add selected pixel", self.add_map_pixel),
                         ("Add map median", self.add_map_median)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Up", lambda: self._move(-1)), ("Down", lambda: self._move(1)),
                         ("Remove", self._remove), ("Clear", self.clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)

        form = QtWidgets.QFormLayout()
        self.ed_win = QtWidgets.QLineEdit("1450-1650, 2550-2800")
        form.addRow("Ranges (cm⁻¹)", self.ed_win)
        self.spn_step = QtWidgets.QDoubleSpinBox()
        self.spn_step.setRange(0.0, 5.0)
        self.spn_step.setSingleStep(0.05)
        self.spn_step.setValue(1.15)
        form.addRow("Offset", self.spn_step)
        self.chk_norm = QtWidgets.QCheckBox("Normalise each spectrum")
        self.chk_norm.setChecked(True)
        form.addRow(self.chk_norm)
        self.chk_labels = QtWidgets.QCheckBox("Spectrum labels")
        form.addRow(self.chk_labels)
        self.chk_bands = QtWidgets.QCheckBox("Peak labels")
        self.ed_bands = QtWidgets.QLineEdit("G 1583, 2D 2690")
        rb = QtWidgets.QHBoxLayout()
        rb.addWidget(self.chk_bands)
        rb.addWidget(self.ed_bands, 1)
        form.addRow(rb)
        self.ed_title = QtWidgets.QLineEdit("")
        form.addRow("Title", self.ed_title)
        self.spn_w = QtWidgets.QDoubleSpinBox()
        self.spn_w.setRange(3.0, 30.0)
        self.spn_w.setValue(8.7)
        self.spn_w.setSuffix(" cm")
        self.spn_h = QtWidgets.QDoubleSpinBox()
        self.spn_h.setRange(0.0, 30.0)
        self.spn_h.setSpecialValueText("auto")
        self.spn_h.setSuffix(" cm")
        rs = QtWidgets.QHBoxLayout()
        rs.addWidget(self.spn_w)
        rs.addWidget(QtWidgets.QLabel("×"))
        rs.addWidget(self.spn_h)
        form.addRow("Size", rs)
        left.addLayout(form)
        b = QtWidgets.QPushButton("Export PDF…")
        b.clicked.connect(self.export_pdf)
        left.addWidget(b)

        for w in (self.ed_win, self.ed_bands, self.ed_title):
            w.editingFinished.connect(self.redraw)
        for w in (self.spn_step, self.spn_w, self.spn_h):
            w.valueChanged.connect(self.redraw)
        for w in (self.chk_norm, self.chk_labels, self.chk_bands):
            w.toggled.connect(self.redraw)

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(430)
        root.addWidget(lw)
        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("background:white;")
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.preview)
        root.addWidget(sc, 1)
        self.redraw()

    # -- items ----------------------------------------------------------------
    def add_spectrum(self, spec: Spectrum):
        used = {it["color"] for it in self.items}
        color = next((c for c in STACK_COLORS if c not in used), STACK_COLORS[0])
        self.items.append({"spec": spec, "color": color, "visible": True})
        self._rebuild()
        self.redraw()

    def add_paths(self, paths: List[str]):
        bad = []
        for p in paths:
            try:
                if os.path.isdir(p):
                    mf = read_spot_folder(p)
                    self.add_spectrum(Spectrum(
                        mf.axis, np.nanmedian(mf.data, 0),
                        short_label(p) + " median", p))
                elif is_single_spectrum(p):
                    self.add_spectrum(read_single_spectrum(p))
                else:
                    mf = parse_map_file(p)
                    if mf.layout != "spectra":
                        raise ValueError("not a spectrum")
                    self.add_spectrum(Spectrum(
                        mf.axis, np.nanmedian(mf.data, 0),
                        short_label(p) + " median", p))
            except Exception as exc:
                bad.append("%s: %s" % (os.path.basename(p), exc))
        if bad:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Could not read:\n\n"
                                          + "\n".join(bad))

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add spectra", "",
            "Spectra (*.txt *.dat *.csv *.h5);;All files (*)")
        if paths:
            self.add_paths(paths)
            self.main.tabs.setCurrentWidget(self)

    def add_map_pixel(self):
        mf, sel = self.main.mapfile, self.main.tab_map.sel
        if mf is None or sel is None or mf.index_map[sel] < 0:
            self.main.status("Click a pixel in the Map Explorer first.")
            return
        i = int(mf.index_map[sel])
        name = short_label(mf.path)
        self.add_spectrum(Spectrum(mf.axis, mf.data[i].astype(float),
                                   "%s (%.1f, %.1f)" % (name, mf.x[i], mf.y[i]),
                                   mf.path))

    def add_map_median(self):
        mf = self.main.mapfile
        if mf is None:
            self.main.status("Load a map first.")
            return
        name = short_label(mf.path)
        self.add_spectrum(Spectrum(mf.axis, np.nanmedian(mf.data, 0),
                                   "%s median" % name, mf.path))

    def clear(self):
        self.items = []
        self._rebuild()
        self.redraw()

    def _remove(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.items.pop(r)
        self._rebuild()
        self.redraw()

    def _move(self, d: int):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        if len(rows) != 1:
            return
        r, t = rows[0], rows[0] + d
        if 0 <= t < len(self.items):
            self.items[r], self.items[t] = self.items[t], self.items[r]
            self._rebuild()
            self.tbl.selectRow(t)
            self.redraw()

    def _rebuild(self):
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(self.items))
        for r, it in enumerate(self.items):
            a = QtWidgets.QTableWidgetItem(it["spec"].label)
            a.setFlags(a.flags() | QtCore.Qt.ItemIsUserCheckable)
            a.setCheckState(QtCore.Qt.Checked if it["visible"] else QtCore.Qt.Unchecked)
            self.tbl.setItem(r, 0, a)
            c = QtWidgets.QTableWidgetItem("")
            c.setFlags(QtCore.Qt.ItemIsEnabled)
            c.setBackground(QtGui.QColor(it["color"]))
            self.tbl.setItem(r, 1, c)
            s = QtWidgets.QTableWidgetItem(os.path.basename(it["spec"].path.rstrip(os.sep)))
            s.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            s.setToolTip(it["spec"].path)
            self.tbl.setItem(r, 2, s)
        self.tbl.blockSignals(False)

    def _item_changed(self, item):
        r = item.row()
        if item.column() == 0 and 0 <= r < len(self.items):
            self.items[r]["spec"].label = item.text()
            self.items[r]["visible"] = item.checkState() == QtCore.Qt.Checked
            self.redraw()

    def _cell_clicked(self, r: int, c: int):
        if c != 1 or not (0 <= r < len(self.items)):
            return
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.items[r]["color"]), self)
        if col.isValid():
            self.items[r]["color"] = col.name()
            self._rebuild()
            self.redraw()

    # -- figure ---------------------------------------------------------------
    def _figure(self) -> Figure:
        traces = [{"wn": it["spec"].wn, "inten": it["spec"].inten,
                   "label": it["spec"].label, "color": it["color"]}
                  for it in self.items if it["visible"]]
        w = self.spn_w.value() / 2.54
        h = self.spn_h.value() / 2.54 or (2.6 + 1.25 * max(1, len(traces))) / 2.54
        fig = Figure(figsize=(w, h))
        FigureCanvasAgg(fig)
        draw_stack(fig, traces, parse_windows(self.ed_win.text()),
                   step=self.spn_step.value(), normalise=self.chk_norm.isChecked(),
                   trace_labels=self.chk_labels.isChecked(),
                   band_labels=parse_band_labels(self.ed_bands.text())
                   if self.chk_bands.isChecked() else [],
                   title=self.ed_title.text().strip())
        return fig

    def redraw(self):
        import io
        fig = self._figure()
        buf = io.BytesIO()
        ratio = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        fig.savefig(buf, format="png", dpi=150 * ratio, bbox_inches="tight",
                    pad_inches=0.03, facecolor="white")
        pm = QtGui.QPixmap()
        pm.loadFromData(buf.getvalue())
        pm.setDevicePixelRatio(ratio)
        self.preview.setPixmap(pm)

    def export_pdf(self):
        stem = "stacked_spectra.pdf"
        vis = [it for it in self.items if it["visible"]]
        if vis and vis[0]["spec"].path:
            stem = os.path.join(os.path.dirname(vis[0]["spec"].path.rstrip(os.sep)), stem)
        start = os.path.join(_EXPORT_DIR["path"], os.path.basename(stem)) \
            if _EXPORT_DIR["path"] else stem
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", start, "Vector PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        _EXPORT_DIR["path"] = os.path.dirname(path)
        self._figure().savefig(path, bbox_inches="tight", pad_inches=0.03,
                               facecolor="white")
        self.main.status("Exported %s" % path)


# ---------------------------------------------------------------------------
# Tab 5 - Layer prediction against 1L / 2L reference datasets
# ---------------------------------------------------------------------------

REF_ROWS = [("2D_fwhm", "2D FWHM (cm⁻¹)"), ("I2D_IG", "I(2D)/I(G)"),
            ("A2D_AG", "A(2D)/A(G)"), ("2D_pos", "2D position (cm⁻¹)"),
            ("2D_height", "2D intensity"), ("G_pos", "G position (cm⁻¹)"),
            ("G_fwhm", "G FWHM (cm⁻¹)"), ("G_height", "G intensity"),
            ("Gminus_pos", "G⁻ position (cm⁻¹)"), ("Gplus_pos", "G⁺ position (cm⁻¹)")]


class PredictTab(QtWidgets.QWidget):
    """Reference values (Lorentzian fits of the 1L and 2L datasets, mean ±
    SD) and layer-number / stacking predictions for new data."""

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.ref = None
        self.preds: Optional[List[dict]] = None
        self.pred_title = ""
        self._build()
        if HAVE_GREF and os.path.exists(gref.DEFAULT_REF):
            self._set_ref(gref.load(gref.DEFAULT_REF), gref.DEFAULT_REF)

    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Build reference…", self.build_ref),
                         ("Load reference…", self.load_ref)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            b.setEnabled(HAVE_GREF)
            r.addWidget(b)
        left.addLayout(r)
        self.lbl_ref = QtWidgets.QLabel("No reference loaded")
        self.lbl_ref.setWordWrap(True)
        self.lbl_ref.setStyleSheet("color:#555; font-size:11px;")
        left.addWidget(self.lbl_ref)
        self.tbl_ref = QtWidgets.QTableWidget(0, 0)
        self.tbl_ref.verticalHeader().setDefaultSectionSize(22)
        self.tbl_ref.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_ref, 2)
        r = QtWidgets.QHBoxLayout()
        self.btn_map = QtWidgets.QPushButton("Predict loaded map")
        self.btn_map.clicked.connect(self.predict_map)
        self.btn_dir = QtWidgets.QPushButton("Predict folder…")
        self.btn_dir.clicked.connect(self.predict_dir)
        for b in (self.btn_map, self.btn_dir):
            b.setEnabled(False)
            r.addWidget(b)
        left.addLayout(r)
        self.tbl_pred = QtWidgets.QTableWidget(0, 2)
        self.tbl_pred.setHorizontalHeaderLabels(["Prediction", "Spectra"])
        self.tbl_pred.verticalHeader().setVisible(False)
        self.tbl_pred.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tbl_pred.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_pred, 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMinimumWidth(600)
        lw.setMaximumWidth(640)
        root.addWidget(lw)
        self.plot = Canvas(figsize=(8, 5))
        root.addWidget(self.plot, 1)
        self.redraw()

    # -- reference --------------------------------------------------------------
    def _set_ref(self, ref, path):
        self.ref = ref
        ds = ref["datasets"]
        self.lbl_ref.setText("Reference %s (built %s). Predictions use the 'matching' "
                             "spots: fitted 2D FWHM %s." % (
                                 os.path.basename(path), ref.get("created", "?"),
                                 ", ".join("%s %g–%g" % (k, *v)
                                           for k, v in ref["match_rule"].items())))
        cols = []
        for d in ds:
            for which in ("all", "matching"):
                cols.append((d, which))
        self.tbl_ref.clear()
        self.tbl_ref.setColumnCount(len(cols))
        self.tbl_ref.setRowCount(len(REF_ROWS) + 1)
        self.tbl_ref.setHorizontalHeaderLabels(
            ["%s %s\n%s" % (d["label"], d["name"].split("_")[0], w) for d, w in cols])
        self.tbl_ref.setVerticalHeaderLabels(["spectra"] + [lab for _, lab in REF_ROWS])
        for j, (d, w) in enumerate(cols):
            n = d["n_fitted"] if w == "all" else d["n_matching"]
            self.tbl_ref.setItem(0, j, QtWidgets.QTableWidgetItem("%d" % n))
            for i, (q, _) in enumerate(REF_ROWS, start=1):
                s = d[w].get(q, {})
                txt = "" if s.get("mean") is None else (
                    "%.4g ± %.2g" % (s["mean"], s["std"] or 0))
                self.tbl_ref.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.tbl_ref.resizeColumnsToContents()
        for b in (self.btn_map, self.btn_dir):
            b.setEnabled(True)
        self.redraw()

    def build_ref(self):
        f1 = QtWidgets.QFileDialog.getExistingDirectory(
            self, "1L reference dataset (folder of spectra, e.g. the ML map)")
        if not f1:
            return
        f2 = QtWidgets.QFileDialog.getExistingDirectory(
            self, "2L reference dataset (folder of spectra, e.g. the BL map)")
        if not f2:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            ref = gref.build([("1L", f1), ("2L", f2)], gref.DEFAULT_REF)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, APP_NAME, "Could not build:\n\n%s" % exc)
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._set_ref(ref, gref.DEFAULT_REF)
        self.main.status("Reference saved to %s" % os.path.dirname(gref.DEFAULT_REF))

    def load_ref(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load reference", os.path.dirname(gref.DEFAULT_REF), "JSON (*.json)")
        if path:
            self._set_ref(gref.load(path), path)

    # -- prediction -------------------------------------------------------------
    def predict_map(self):
        mf = self.main.mapfile
        if mf is None:
            self.main.status("Load a map first.")
            return
        prog = QtWidgets.QProgressDialog("Fitting G and 2D…", "Cancel", 0, mf.n_points, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        rows = []
        for i in range(mf.n_points):
            if prog.wasCanceled():
                return
            prog.setValue(i)
            r, _, _ = rpf.analyse_spectrum(mf.axis, mf.data[i].astype(float))
            r.update(file="spot %d" % i, x=float(mf.x[i]), y=float(mf.y[i]), idx=i)
            rows.append(r)
        prog.setValue(mf.n_points)
        p = mf.path.rstrip(os.sep)
        out = os.path.join(p, "output") if os.path.isdir(p) else \
            os.path.splitext(p)[0] + "_output"
        self._show(gref.write_predictions(rows, self.ref, out, short_label(p)),
                   short_label(p), out)

    def predict_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Dataset to predict")
        if not folder:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            preds = gref.predict_folder(folder, self.ref)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._show(preds, short_label(folder), os.path.join(folder, "output"))

    def _show(self, preds, title, out):
        self.preds, self.pred_title = preds, title
        counts: Dict[str, int] = {}
        for p in preds:
            key = p["layer"] + ("  " + p["polytype"] if p["polytype"] not in ("", "—") else "")
            counts[key] = counts.get(key, 0) + 1
        self.tbl_pred.setRowCount(len(counts))
        for i, (k, n) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
            self.tbl_pred.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.tbl_pred.setItem(i, 1, QtWidgets.QTableWidgetItem("%d" % n))
        self.redraw()
        self.main.status("Predictions saved to %s" % os.path.join(out, "predictions.csv"))

    # -- figure -----------------------------------------------------------------
    def redraw(self):
        fig = self.plot.figure
        fig.clear()
        if not HAVE_GREF or self.ref is None:
            fig.text(0.5, 0.5, "No reference" if HAVE_GREF else
                     "raman_peak_fit.py / graphene_reference.py not found",
                     ha="center", va="center", color="#999")
            self.plot.draw()
            return
        preds = self.preds or []
        has_xy = sum(np.isfinite(p["x"]) for p in preds) > 3
        axs = np.atleast_1d(fig.subplots(1, 2 if has_xy else 1))
        ax = axs[0]
        gref.draw_reference(ax, self.ref, matching_only=not preds)
        for lab, c in gref.COLORS.items():
            if lab == "no fit":                 # failed fits carry no usable values
                continue
            sel = [p for p in preds if p["layer"] == lab and np.isfinite(p["2D_fwhm"])]
            if sel:
                ax.scatter([p["2D_fwhm"] for p in sel], [p["I2D_IG"] for p in sel], s=18,
                           marker="x", lw=0.8, color="k" if lab in ("1L", "2L") else c,
                           label="%s: %s (%d)" % (self.pred_title, lab, len(sel)))
        ax.legend(fontsize=7, frameon=False)
        if has_xy:
            ax = axs[1]
            for lab, c in gref.COLORS.items():
                sel = [p for p in preds if p["layer"] == lab]
                if sel:
                    ax.scatter([p["x"] for p in sel], [p["y"] for p in sel], s=40,
                               marker="s", color=c, lw=0, label="%s (%d)" % (lab, len(sel)))
            ax.set_aspect("equal")
            ax.set_xlabel("X  [µm]")
            ax.set_ylabel("Y  [µm]")
            ax.set_title(self.pred_title, fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
            orient_axes(ax, self.main.cfg)
        self.plot.draw()


# ---------------------------------------------------------------------------
# Tab 6 - Peak fitting: Si calibration, background, Lorentzians, peak count
# ---------------------------------------------------------------------------

BAND_DEFAULTS = [("G", 1500.0, 1680.0, "Auto", 3), ("2D", 2560.0, 2810.0, "Auto", 4)]
COMP_COLORS = ["#1f5fa8", "#e8402a", "#2a8c4a", "#8a5a00", "#7b3fa0", "#17becf"]


class PeakFitTab(QtWidgets.QWidget):
    """Fit any spectra band by band: optional Si calibration, a background
    model, and Lorentzians with the number of peaks set by hand or chosen
    from the data. Batch over files, folders or the loaded map."""

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.items: List[dict] = []      # {label, wn, y, path, x, ypos}
        self.results: Dict[int, dict] = {}
        self._build()

    # -- UI -------------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()

        self.lst = QtWidgets.QListWidget()
        self.lst.currentRowChanged.connect(self._selected)
        left.addWidget(self.lst, 2)
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Add files…", self.add_files), ("Add folder…", self.add_folder),
                         ("Use loaded map", self.use_map), ("Clear", self.clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)

        form = QtWidgets.QFormLayout()
        self.chk_si = QtWidgets.QCheckBox("Calibrate on Si (520.7 cm⁻¹)")
        self.chk_si.setChecked(True)
        form.addRow(self.chk_si)
        self.cmb_bg = QtWidgets.QComboBox()
        self.cmb_bg.addItems(["linear", "polynomial", "als", "none"])
        form.addRow("Background", self.cmb_bg)
        rb = QtWidgets.QHBoxLayout()
        self.spn_order = QtWidgets.QSpinBox()
        self.spn_order.setRange(1, 8)
        self.spn_order.setValue(3)
        self.ed_lam = QtWidgets.QLineEdit("1e5")
        self.ed_p = QtWidgets.QLineEdit("0.01")
        for lab, w in (("order", self.spn_order), ("λ", self.ed_lam), ("p", self.ed_p)):
            rb.addWidget(QtWidgets.QLabel(lab))
            rb.addWidget(w)
        form.addRow(rb)
        left.addLayout(form)

        self.tbl_bands = QtWidgets.QTableWidget(0, 5)
        self.tbl_bands.setHorizontalHeaderLabels(["Band", "From", "To", "Peaks", "Max"])
        self.tbl_bands.verticalHeader().setVisible(False)
        self.tbl_bands.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_bands.setMaximumHeight(140)
        for band in BAND_DEFAULTS:
            self._add_band_row(*band)
        left.addWidget(self.tbl_bands)
        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Add band", lambda: self._add_band_row("band", 1000, 1100, "Auto", 3)),
                         ("Remove band", self._remove_band)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)

        r = QtWidgets.QHBoxLayout()
        for text, fn in (("Fit selected", self.fit_selected), ("Fit all…", self.fit_all),
                         ("ω_G–ω_2D plot…", self.correlation)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            b.setEnabled(HAVE_GREF)
            r.addWidget(b)
        left.addLayout(r)

        self.tbl_peaks = QtWidgets.QTableWidget(0, 6)
        self.tbl_peaks.setHorizontalHeaderLabels(["Band", "#", "Centre", "FWHM", "Height", "Area"])
        self.tbl_peaks.verticalHeader().setVisible(False)
        self.tbl_peaks.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_peaks.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_peaks, 2)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#555; font-size:11px;")
        left.addWidget(self.lbl)

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(500)
        root.addWidget(lw)
        self.plot = Canvas(figsize=(8, 5))
        root.addWidget(self.plot, 1)
        self._placeholder()

    def _add_band_row(self, name, lo, hi, peaks, nmax):
        r = self.tbl_bands.rowCount()
        self.tbl_bands.insertRow(r)
        self.tbl_bands.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        self.tbl_bands.setItem(r, 1, QtWidgets.QTableWidgetItem("%g" % lo))
        self.tbl_bands.setItem(r, 2, QtWidgets.QTableWidgetItem("%g" % hi))
        cmb = QtWidgets.QComboBox()
        cmb.addItems(["Auto"] + [str(k) for k in range(1, 7)])
        cmb.setCurrentText(str(peaks))
        self.tbl_bands.setCellWidget(r, 3, cmb)
        sp = QtWidgets.QSpinBox()
        sp.setRange(1, 6)
        sp.setValue(int(nmax))
        self.tbl_bands.setCellWidget(r, 4, sp)

    def _remove_band(self):
        r = self.tbl_bands.currentRow()
        if r >= 0:
            self.tbl_bands.removeRow(r)

    def bands(self) -> List[tuple]:
        out = []
        for r in range(self.tbl_bands.rowCount()):
            try:
                lo = float(self.tbl_bands.item(r, 1).text())
                hi = float(self.tbl_bands.item(r, 2).text())
            except (AttributeError, ValueError):
                continue
            n = self.tbl_bands.cellWidget(r, 3).currentText()
            out.append((self.tbl_bands.item(r, 0).text(), lo, hi,
                        "auto" if n == "Auto" else int(n),
                        self.tbl_bands.cellWidget(r, 4).value()))
        return out

    def bg_opts(self) -> dict:
        try:
            return {"order": self.spn_order.value(), "lam": float(self.ed_lam.text()),
                    "p": float(self.ed_p.text())}
        except ValueError:
            return {"order": self.spn_order.value()}

    # -- spectra --------------------------------------------------------------
    def _add(self, label, wn, y, path, x=np.nan, ypos=np.nan):
        self.items.append({"label": label, "wn": np.asarray(wn, float),
                           "y": np.asarray(y, float), "path": path, "x": x, "ypos": ypos})
        self.lst.addItem(label)

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add spectra", "", "Spectra (*.txt *.dat *.csv *.h5);;All files (*)")
        for p in paths:
            try:
                s = read_single_spectrum(p)
                self._add(s.label, s.wn, s.inten, p)
            except Exception as exc:
                self.main.status("Skipped %s: %s" % (os.path.basename(p), exc))
        if self.items and self.lst.currentRow() < 0:
            self.lst.setCurrentRow(0)

    def add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of spectra")
        if not folder:
            return
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if not name.lower().endswith((".txt", ".dat", ".csv")) or not is_single_spectrum(p):
                continue
            s = read_single_spectrum(p)
            m = _SPOT_RE.search(name)
            self._add(short_label(p), s.wn, s.inten, p,
                      float(m.group(1)) if m else np.nan, float(m.group(2)) if m else np.nan)
        if self.items and self.lst.currentRow() < 0:
            self.lst.setCurrentRow(0)

    def use_map(self):
        mf = self.main.mapfile
        if mf is None:
            self.main.status("Load a map first.")
            return
        for i in range(mf.n_points):
            self._add("spot %d (%.1f, %.1f)" % (i, mf.x[i], mf.y[i]), mf.axis, mf.data[i],
                      mf.path, float(mf.x[i]), float(mf.y[i]))
        if self.lst.currentRow() < 0:
            self.lst.setCurrentRow(0)

    def clear(self):
        self.items, self.results = [], {}
        self.lst.clear()
        self.tbl_peaks.setRowCount(0)
        self._placeholder()

    # -- fitting --------------------------------------------------------------
    def fit_item(self, i: int) -> dict:
        it = self.items[i]
        wn, y = it["wn"], it["y"]
        shift, si = rt.si_shift(wn, y) if self.chk_si.isChecked() else (0.0, None)
        w = wn + shift
        nz = rt.noise_level(w, y)
        res = {"shift": shift, "si": si, "noise": nz, "bands": {}}
        for name, lo, hi, n, nmax in self.bands():
            res["bands"][name] = rt.analyse_band(w, y, lo, hi, n=n, nmax=nmax,
                                                 bg=self.cmb_bg.currentText(),
                                                 bg_opts=self.bg_opts(), noise=nz)
        self.results[i] = res
        return res

    def _selected(self, i: int):
        if 0 <= i < len(self.items):
            self.fit_selected()

    def fit_selected(self):
        i = self.lst.currentRow()
        if not (0 <= i < len(self.items)):
            return
        self.draw_fit(i, self.fit_item(i))

    def fit_all(self):
        if not self.items:
            return
        prog = QtWidgets.QProgressDialog("Fitting…", "Cancel", 0, len(self.items), self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        for i in range(len(self.items)):
            if prog.wasCanceled():
                return
            prog.setValue(i)
            self.fit_item(i)
        prog.setValue(len(self.items))
        src = self.items[0]["path"].rstrip(os.sep)
        base = src if os.path.isdir(src) else os.path.dirname(src)
        out = os.path.join(base, "output")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "peak_fits.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["spectrum", "x", "y", "si_shift", "band", "n_peaks", "peak",
                        "centre", "centre_err", "fwhm", "fwhm_err", "height", "height_err",
                        "area", "area_err", "r2", "note"])
            for i, it in enumerate(self.items):
                res = self.results[i]
                for name, b in res["bands"].items():
                    rows = b.get("comps") or [None]
                    for k, c in enumerate(rows):
                        w.writerow([it["label"], _fmt(it["x"]), _fmt(it["ypos"]),
                                    _fmt(res["shift"]), name, b.get("n", 0), k + 1 if c else ""]
                                   + (["%.3f" % c[q] for q in ("pos", "pos_err", "fwhm", "fwhm_err",
                                                                "height", "height_err", "area",
                                                                "area_err")] if c else [""] * 8)
                                   + [_fmt(b.get("r2", np.nan)),
                                      b.get("why") or b.get("stopped", "")])
        self.draw_summary()
        self.main.status("Fitted %d spectra — saved %s" % (len(self.items), path))

    # -- drawing --------------------------------------------------------------
    def _placeholder(self):
        self.plot.figure.clear()
        self.plot.figure.text(0.5, 0.5, "Add spectra", ha="center", va="center", color="#999")
        self.plot.draw()

    def draw_fit(self, i: int, res: dict):
        fig = self.plot.figure
        fig.clear()
        bands = [(n, b) for n, b in res["bands"].items()]
        if not bands:
            self._placeholder()
            return
        gs = fig.add_gridspec(2, len(bands), height_ratios=[3, 1], hspace=0.05)
        self.tbl_peaks.setRowCount(0)
        for j, (name, b) in enumerate(bands):
            ax = fig.add_subplot(gs[0, j])
            axr = fig.add_subplot(gs[1, j], sharex=ax)
            if "x" not in b:
                ax.text(0.5, 0.5, "%s: %s" % (name, b.get("why", "")), transform=ax.transAxes,
                        ha="center", color="#999")
                continue
            x, raw, bg = b["x"], b["raw"], b["bg"]
            ax.plot(x, raw, color="#999", lw=0.7, label="data")
            ax.plot(x, bg, color="k", lw=0.6, ls="--", label="background")
            if b.get("ok"):
                base = bg + b["baseline"]
                ax.plot(x, base + b["curve"][2] - b["baseline"], color="#d81b60", lw=1.0,
                        label="fit")
                for k, comp in enumerate(b["components"]):
                    ax.plot(x, base + comp, color=COMP_COLORS[k % 6], lw=0.7)
                axr.plot(x, (b["curve"][1] - b["curve"][2]) / (b.get("noise") or 1),
                         color="#555", lw=0.6)
                for k, c in enumerate(b["comps"]):
                    r = self.tbl_peaks.rowCount()
                    self.tbl_peaks.insertRow(r)
                    for col, txt in enumerate([name, str(k + 1),
                                               "%.2f ± %.2f" % (c["pos"], c["pos_err"]),
                                               "%.2f ± %.2f" % (c["fwhm"], c["fwhm_err"]),
                                               "%.0f ± %.0f" % (c["height"], c["height_err"]),
                                               "%.0f" % c["area"]]):
                        self.tbl_peaks.setItem(r, col, QtWidgets.QTableWidgetItem(txt))
            ax.set_title("%s: %d peak%s%s" % (name, b.get("n", 0), "" if b.get("n") == 1 else "s",
                                              "" if b.get("ok") else " (failed)"), fontsize=9)
            ax.tick_params(labelbottom=False, labelsize=7)
            ax.set_yticks([])
            axr.axhline(0, color="k", lw=0.4)
            axr.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8)
            axr.tick_params(labelsize=7)
            if j == 0:
                axr.set_ylabel("resid./σ", fontsize=7)
                ax.legend(fontsize=7, frameon=False)
        notes = ["Si shift %+.2f cm⁻¹" % res["shift"] if self.chk_si.isChecked() else "no Si calibration",
                 "noise %.1f counts" % res["noise"]]
        for name, b in bands:
            if b.get("stopped"):
                notes.append("%s: stopped at %d — %s" % (name, b.get("n", 0), b["stopped"]))
        self.lbl.setText(";  ".join(notes))
        self.plot.draw()

    def draw_summary(self):
        """Peak count per band: a map when the spectra have positions,
        otherwise a histogram."""
        fig = self.plot.figure
        fig.clear()
        names = [b[0] for b in self.bands()]
        has_xy = sum(np.isfinite(it["x"]) for it in self.items) > 3
        axs = np.atleast_1d(fig.subplots(1, len(names)))
        for ax, name in zip(axs, names):
            n = np.array([self.results[i]["bands"].get(name, {}).get("n", 0)
                          if self.results[i]["bands"].get(name, {}).get("ok") else 0
                          for i in range(len(self.items))])
            if has_xy:
                x = np.array([it["x"] for it in self.items])
                y = np.array([it["ypos"] for it in self.items])
                cmap = matplotlib.colors.ListedColormap(["#eeeeee"] + COMP_COLORS)
                sc = ax.scatter(x, y, c=n, cmap=cmap, vmin=-0.5, vmax=6.5, s=40, marker="s", lw=0)
                ax.set_aspect("equal")
                orient_axes(ax, self.main.cfg)
                cb = fig.colorbar(sc, ax=ax, ticks=range(0, 7), shrink=0.8)
                cb.set_label("peaks (0 = no fit)", fontsize=7)
            else:
                ax.hist(n, bins=np.arange(-0.5, 7.5, 1), color="#777")
                ax.set_xlabel("peaks")
            ax.set_title("%s peak count" % name, fontsize=9)
        self.plot.draw()

    def correlation(self):
        if not self.items:
            return
        ref = self.main.tab_pred.ref
        if ref is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Build or load a 1L/2L "
                                              "reference in tab 5 first.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("ω_G – ω_2D correlation")
        lay = QtWidgets.QHBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        eds = {}
        for k, lab in (("G0", "ω_G0 (cm⁻¹)"), ("D0", "ω_2D0 (cm⁻¹)"),
                       ("a_G", "∂ω_G/∂ε (cm⁻¹/%)"), ("s_strain", "strain slope"),
                       ("b_G", "∂ω_G/∂n (cm⁻¹/10¹³)"), ("s_doping", "doping slope")):
            eds[k] = QtWidgets.QLineEdit("%g" % sd.PARAMS[k])
            form.addRow(lab, eds[k])
        btn = QtWidgets.QPushButton("Redraw")
        form.addRow(btn)
        fw = QtWidgets.QWidget()
        fw.setLayout(form)
        lay.addWidget(fw)
        cv = Canvas(figsize=(5, 5))
        lay.addWidget(cv, 1)
        prog = QtWidgets.QProgressDialog("Analysing…", "Cancel", 0, len(self.items), self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        def tick(k, n):
            prog.setValue(k)
            QtWidgets.QApplication.processEvents()
            return not prog.wasCanceled()
        rows = sd.analyse_many([(it["wn"], it["y"]) for it in self.items], ref,
                               progress=tick)
        prog.setValue(len(self.items))
        if rows is None:
            return

        def redraw():
            try:
                p = {k: float(e.text()) for k, e in eds.items()}
            except ValueError:
                return
            cv.figure.clear()
            ax = cv.figure.add_subplot(111)
            sd.draw_correlation(ax, rows, p)
            cv.draw()
        btn.clicked.connect(redraw)
        redraw()
        fit_to_screen(dlg, 1000, 700)
        dlg.exec_()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        fit_to_screen(self, 1500, 960)

        self.cfg = default_config()
        self.mapfile: Optional[MapFile] = None
        self.mapfile_raw: Optional[MapFile] = None
        self.layer_by_spec: Optional[np.ndarray] = None   # N per spectrum
        self.optical: Optional[MapFile] = None
        self.optical_image: Optional[np.ndarray] = None
        self.fits: List[dict] = []
        self.g_fits: Optional[List[dict]] = None
        self.ratio_maps: Dict[str, Optional[np.ndarray]] = {"height": None,
                                                            "area": None}
        self.ratio_ok: bool = False
        self.ratio_note: str = "no file loaded"
        self.fit_tally: Dict[str, int] = {}
        self.fwhm_map: Optional[np.ndarray] = None
        self.asym_maps: Dict[str, Optional[np.ndarray]] = {
            "hwhm": None, "area": None, "residual": None}
        self.worker: Optional[FitWorker] = None

        self.tabs = QtWidgets.QTabWidget()
        self.tab_map = MapExplorerTab(self)
        self.tab_class = ClassificationTab(self)
        self.tabs.addTab(self.tab_map, "1 · Map Explorer")
        self.tabs.addTab(self.tab_class, "2 · Classification")
        self.tab_spec = SpectraTab(self)
        self.tabs.addTab(self.tab_spec, "3 · Spectra")
        self.tab_stack = StackTab(self)
        self.tabs.addTab(self.tab_stack, "4 · Stacked spectra")
        self.tab_pred = PredictTab(self)
        self.tabs.addTab(self.tab_pred, "5 · Layer prediction")
        self.tab_fit = PeakFitTab(self)
        self.tabs.addTab(self.tab_fit, "6 · Peak fitting")
        self.setCentralWidget(self.tabs)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.status("Ready. %s" % ("lmfit available." if HAVE_LMFIT
                                   else "lmfit not found — using scipy fallback."))
        self._menus()

    # -- menus --------------------------------------------------------------
    def _menus(self):
        m = self.menuBar().addMenu("&File")
        m.addAction("Open Raman map…", self.load_raman, "Ctrl+O")
        m.addAction("Open map folder…", self.load_folder)
        m.addAction("Open optical image…", self.load_optical)
        m.addAction("Add spectra to stack…", self.tab_stack.add_files)
        m.addSeparator()
        m.addAction("Save settings…", self.save_config)
        m.addAction("Load settings…", self.load_config)
        m.addSeparator()
        m.addAction("Quit", self.close, "Ctrl+Q")

        a = self.menuBar().addMenu("&Analysis")
        a.addAction("Fit parameters…", self.edit_fit_params)
        a.addAction("Re-fit all spectra", self.start_fitting)

        h = self.menuBar().addMenu("&Help")
        h.addAction("About", self.about)

    def status(self, text: str):
        self.statusBar().showMessage(text, 8000)

    def suggest_name(self, suffix: str) -> str:
        if not self.mapfile:
            return "map" + suffix
        base = os.path.splitext(self.mapfile.path)[0]
        return base + suffix

    # -- loading ------------------------------------------------------------
    def load_raman(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Raman map", "",
            "Text/map files (*.txt *.dat *.csv *.h5);;All files (*)")
        if path:
            self.open_raman(path)

    def load_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open map folder (one file per spot)")
        if path:
            self.open_raman(path)

    def open_raman(self, path: str):
        # A single spectrum is not a map: it goes to the stack tab instead.
        if not os.path.isdir(path) and is_single_spectrum(path):
            self.tab_stack.add_paths([path])
            self.tabs.setCurrentWidget(self.tab_stack)
            self.status("%s is a single spectrum — added to the Stacked "
                        "spectra tab." % os.path.basename(path))
            return
        try:
            mf = read_spot_folder(path) if os.path.isdir(path) else parse_map_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, APP_NAME, "Could not parse the file:\n\n%s" % exc)
            return
        if mf.layout == "image":
            QtWidgets.QMessageBox.warning(
                self, APP_NAME,
                "This file is a camera image, not a Raman map — its rows carry "
                "one spatial coordinate and a row of pixel values, with no "
                "spectra to fit.\n\nLoading it as the optical image instead.")
            self.optical = mf
            self.optical_image = mf.greyscale()
            self.tab_map.chk_optical.setChecked(True)
            self.tab_map.refresh_map()
            return
        if mf.axis.max() < self.cfg["band_min"] or mf.axis.min() > self.cfg["band_max"]:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME,
                "The axis of this file spans %.0f–%.0f, which does not cover the "
                "2D-band window %.0f–%.0f cm⁻¹.\n\nChange the window under "
                "Analysis ▸ Fit parameters."
                % (mf.axis.min(), mf.axis.max(),
                   self.cfg["band_min"], self.cfg["band_max"]))
        # Honour the instrument's own resolution as the narrow-peak cutoff.
        res = mf.spectral_resolution
        if res and res > 0:
            self.cfg["min_fwhm"] = float(res)
        self.tab_class.close_regions_dialog()
        self.mapfile_raw = mf
        self.mapfile = apply_map_transform(mf, self.current_transform())
        self.layer_by_spec = np.zeros(mf.n_points, dtype=int)
        if self.cfg.get("layer_mode") == "map":
            self.cfg["layer_mode"] = "manual"          # a new map has no layer map
            self.tab_class.rb_manual.setChecked(True)
        self.tab_class.update_map_status()
        t1 = self.tab_map
        t1.sync_transform_ui()
        for sp in (t1.spn_x0, t1.spn_x1):
            sp.blockSignals(True)
            sp.setRange(float(mf.axis[0]), float(mf.axis[-1]))
            sp.blockSignals(False)
        t1._spec = None
        self.tab_spec.reset()
        self.ratio_ok, self.ratio_note = ratio_applicable(mf.axis, self.cfg)
        self.g_fits = None
        self.ratio_maps = {"height": None, "area": None}
        self.tab_class.update_auto_availability()
        self.fits = []
        self.fwhm_map = None
        self.tab_class.layers = None
        bits = ["Loaded %s — %d spectra on a %d × %d grid"
                % (os.path.basename(path), mf.n_points, mf.shape[1], mf.shape[0])]
        if mf.swapped:
            bits.append("X/Y taken from the #AxisType headers (column 1 is X)")
        if res:
            bits.append("narrow-peak cutoff set to %.1f cm⁻¹ from the header" % res)
        self.status("; ".join(bits))
        self.start_fitting()

    def load_optical(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open optical image", "",
            "Text/map files (*.txt *.dat *.csv);;All files (*)")
        if not path:
            return
        try:
            opt = parse_map_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, APP_NAME, "Could not parse the file:\n\n%s" % exc)
            return
        self.optical = opt
        self.optical_image = opt.greyscale()
        kind = "camera image" if opt.is_optical else "image-like map"
        self.status("Loaded %s (%s) — %d × %d pixels, X %.1f–%.1f µm, "
                    "Y %.1f–%.1f µm%s"
                    % (os.path.basename(path), kind,
                       opt.shape[1], opt.shape[0],
                       opt.ux.min(), opt.ux.max(), opt.uy.min(), opt.uy.max(),
                       self._coverage_note(opt)))
        self.tab_map.chk_optical.setChecked(True)
        self.tab_map.refresh_map()
        self.tab_class.refresh_map()

    def _coverage_note(self, opt: MapFile) -> str:
        """Flag a Raman map that falls outside the camera frame.

        Measured against the actual measurement positions, not the displayed
        pixel extents, which are padded by half a step and would make any
        coarse map look like it overflows. A small overhang is just a map
        drawn to the edge of the frame; a large one usually means the two
        coordinate columns were read the wrong way round.
        """
        mf = self.mapfile
        if mf is None or mf.layout != "spectra":
            return ""
        ol, orr, ob, ot = opt.extent
        over = max(ol - mf.x.min(), mf.x.max() - orr,
                   ob - mf.y.min(), mf.y.max() - ot)
        if over <= 1.0:
            return ""
        span = max(mf.x.max() - mf.x.min(), mf.y.max() - mf.y.min(), 1.0)
        if over > 0.1 * span:
            return ("  ⚠ the map sits %.1f µm outside the camera frame — check "
                    "that the X and Y columns were read the right way round"
                    % over)
        return "  (the map reaches %.1f µm past the camera frame edge)" % over

    # -- fitting ------------------------------------------------------------
    def start_fitting(self):
        if self.mapfile is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Load a map first.")
            return
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.worker = FitWorker(self.mapfile, self.cfg, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_fits)
        self.worker.failed.connect(self._on_fit_failed)
        self.worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.statusBar().showMessage(msg)

    def _on_fit_failed(self, msg: str):
        self.progress.setVisible(False)
        self.status(msg.strip().splitlines()[-1] if msg else "Fitting failed.")

    def _on_fits(self, payload):
        self.progress.setVisible(False)
        if isinstance(payload, dict):
            results, gres = payload["twod"], payload.get("g")
        else:                                   # older callers
            results, gres = payload, None
        self.fits = results
        self.g_fits = gres
        tally: Dict[str, int] = {}
        for res in results:
            if not res["ok"]:
                key = res.get("why") or "rejected"
                tally[key] = tally.get(key, 0) + 1
        self.fit_tally = tally
        ok = self.rebuild_maps()
        self.tab_class.update_auto_availability()
        self.tab_map.refresh_map()
        self.tab_spec.refresh_mini()
        mf = self.mapfile
        med = np.nanmedian(self.fwhm_map) if ok else np.nan
        if ok == 0:
            top = max(tally, key=tally.get) if tally else "rejected"
            self.status("No spectrum passed the quality filters — %d of %d "
                        "failed: %s. See the map panel for the full breakdown."
                        % (tally.get(top, 0), mf.n_points, top))
        else:
            self.status("Fitted %d / %d spectra (%d rejected) — median FWHM(2D) "
                        "%.1f cm⁻¹. Click a pixel to inspect it."
                        % (ok, mf.n_points, mf.n_points - ok, med))

    def rebuild_maps(self) -> int:
        """Lay the per-spectrum fits onto the current grid. No refitting.

        Called after fitting and again whenever the orientation changes,
        since a new orientation is a new grid over the same spectra.
        """
        mf = self.mapfile
        if mf is None or not self.fits:
            self.fwhm_map = None
            return 0
        results, gres = self.fits, self.g_fits
        ny, nx = mf.shape
        self.fwhm_map = np.full((ny, nx), np.nan)
        self.asym_maps = {k: np.full((ny, nx), np.nan)
                          for k in ("hwhm", "area", "residual")}
        self.ratio_maps = {"height": None, "area": None}
        if gres is not None:
            self.ratio_maps = {m: np.full((ny, nx), np.nan) for m in ("height", "area")}
        ok = 0
        for r in range(ny):
            for c in range(nx):
                i = mf.index_map[r, c]
                if i < 0:
                    continue
                res = results[i]
                if res["ok"]:
                    ok += 1
                    self.fwhm_map[r, c] = res["fwhm"]
                    for k in self.asym_maps:
                        self.asym_maps[k][r, c] = asym_of(res, k)
                if gres is not None:
                    for m in ("height", "area"):
                        self.ratio_maps[m][r, c] = pixel_ratio(res, gres[i], m)
        return ok

    def layer_grid(self) -> np.ndarray:
        """The per-spectrum layer counts laid onto the current grid."""
        mf = self.mapfile
        g = np.zeros(mf.shape, dtype=int)
        if self.layer_by_spec is not None:
            ok = mf.index_map >= 0
            g[ok] = self.layer_by_spec[mf.index_map[ok]]
        return g

    # -- orientation ---------------------------------------------------------
    def current_transform(self) -> Optional[dict]:
        if self.mapfile_raw is None:
            return None
        return self.cfg.get("map_transform") or default_transform(self.mapfile_raw, self.cfg)

    def set_transform(self, t: Optional[dict]):
        """Re-place the same spectra under a new orientation / offset."""
        self.cfg["map_transform"] = t
        if self.mapfile_raw is None:
            return
        self.mapfile = apply_map_transform(self.mapfile_raw, self.current_transform())
        # Regions were read from the camera image at the old placement; the
        # assigned layer counts survive (they are per spectrum), the dialog
        # does not.
        self.tab_class.close_regions_dialog()
        self.rebuild_maps()
        self.tab_map.sel = None
        self.tab_map._spec = None
        self.tab_class.layers = None
        self.tab_map.sync_transform_ui()
        self.tab_map.refresh_map()
        self.tab_class.refresh_map()
        self.tab_spec.refresh_mini()
        self.tab_spec._rebuild_list()
        note = self._coverage_note(self.optical) if self.optical is not None else ""
        self.status("Map placed as  %s%s" % (describe_transform(self.current_transform()), note))

    def run_auto_align(self):
        """Fit the map's offset to the camera image.

        By default the orientation is locked to the current one (the
        instrument convention unless changed) and only the offset is searched.
        Searching all eight orientations is optional: on images with strong
        illumination gradients it can lock onto the background instead of the
        flake, which is what happened on 004_01_8.
        """
        if self.mapfile_raw is None or self.optical is None:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Auto-align needs both a Raman map and an optical "
                "image loaded.")
            return
        if self.mapfile_raw.layout != "spectra":
            return
        t0 = self.current_transform()
        rng = float(self.cfg.get("align_search_um", 15.0))
        full = bool(self.tab_map.chk_allorient.isChecked())
        self.status("Aligning map to the optical image…")
        QtWidgets.QApplication.processEvents()
        orient = None if full else [(t0["swap"], t0["flip_x"], t0["flip_y"])]
        want = self.tab_map.cmb_contrast.currentIndex()   # 0 auto, 1 brighter, 2 darker
        sign = {0: None, 1: 1, 2: -1}[want]
        cands, sig = auto_align(self.mapfile_raw, self.optical, self.cfg,
                                search_um=rng, orientations=orient, sign=sign)
        cands = [c for c in cands if c[0] > 0]
        if not cands:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME, "Could not find an alignment — is there a band "
                "signal in this map and a flake in the optical image?")
            return
        best = cands[0]
        t = best[2]
        at_edge = max(abs(t["dx"]), abs(t["dy"])) >= rng - 0.5
        self.set_transform(t)
        if full:
            runner = next((c for c in cands[1:]
                           if (c[2]["swap"], c[2]["flip_x"], c[2]["flip_y"]) !=
                           (t["swap"], t["flip_x"], t["flip_y"])), None)
            margin = best[0] - (runner[0] if runner else 0.0)
            verdict = ("clear" if best[0] >= 0.5 and margin >= 0.2 else
                       "tentative — check it by eye, or against LabSpec's overlay")
            extra = ("Next best orientation:  %s   (r = %+.2f)\n\n"
                     % (describe_transform(runner[2]), runner[1]) if runner else "")
        else:
            verdict = ("offset found" if best[0] >= 0.3 else
                       "weak match — check it by eye")
            extra = "Orientation locked; only the offset was searched (±%.0f µm).\n\n" % rng
        msg = ("Aligned using the %s height against the optical image.\n\n"
               "Placement:  %s   (r = %+.2f)\n%s"
               "Result: %s.%s"
               % (sig, describe_transform(t), best[1], extra, verdict,
                  "\n\n⚠ The best offset lies at the edge of the search range — "
                  "the true offset may be larger. Raise the search range." if at_edge else ""))
        QtWidgets.QMessageBox.information(self, "Auto-align", msg)

    # -- settings -----------------------------------------------------------
    def edit_fit_params(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Fit parameters")
        form = QtWidgets.QFormLayout(dlg)
        s_lo = QtWidgets.QDoubleSpinBox(); s_lo.setRange(0, 5000); s_lo.setDecimals(1)
        s_lo.setValue(self.cfg["band_min"])
        s_hi = QtWidgets.QDoubleSpinBox(); s_hi.setRange(0, 5000); s_hi.setDecimals(1)
        s_hi.setValue(self.cfg["band_max"])
        s_snr = QtWidgets.QDoubleSpinBox(); s_snr.setRange(0, 1000); s_snr.setDecimals(1)
        s_snr.setValue(self.cfg["min_snr"])
        s_max = QtWidgets.QDoubleSpinBox(); s_max.setRange(5, 2000); s_max.setDecimals(1)
        s_max.setValue(self.cfg["max_fwhm"])
        s_min = QtWidgets.QDoubleSpinBox(); s_min.setRange(0, 200); s_min.setDecimals(1)
        s_min.setValue(self.cfg["min_fwhm"])
        s_min.setToolTip("Peaks narrower than the spectral resolution are "
                         "artefacts, not bands. Set from the file header "
                         "where one is present.")
        chk_ds = QtWidgets.QCheckBox("Remove cosmic-ray spikes before fitting")
        chk_ds.setChecked(bool(self.cfg["despike"]))
        s_span = QtWidgets.QDoubleSpinBox(); s_span.setRange(0.2, 10.0)
        s_span.setSingleStep(0.1); s_span.setValue(self.cfg["asym_span"])
        s_smooth = QtWidgets.QSpinBox(); s_smooth.setRange(0, 51); s_smooth.setSingleStep(2)
        s_smooth.setValue(int(self.cfg["smooth_window"]))
        cmb_engine = QtWidgets.QComboBox()
        cmb_engine.addItems(["scipy", "lmfit"])
        cmb_engine.setCurrentText(self.cfg.get("engine", "scipy"))
        if not HAVE_LMFIT:
            cmb_engine.model().item(1).setEnabled(False)
            cmb_engine.setToolTip("lmfit is not installed")
        chk_g = QtWidgets.QCheckBox("Let the Lorentzian width vary independently")
        chk_g.setChecked(bool(self.cfg["free_gamma"]))
        chk_p = QtWidgets.QCheckBox("Fit across CPU cores (large maps)")
        chk_p.setChecked(bool(self.cfg["parallel"]))
        form.addRow("2D band window, lower [cm⁻¹]", s_lo)
        form.addRow("2D band window, upper [cm⁻¹]", s_hi)
        form.addRow("Minimum SNR to accept a fit", s_snr)
        form.addRow("Maximum credible FWHM [cm⁻¹]", s_max)
        form.addRow("Minimum credible FWHM [cm⁻¹]", s_min)
        form.addRow(chk_ds)
        form.addRow("Asymmetry window (× FWHM)", s_span)
        form.addRow("Smoothing points (shape metrics)", s_smooth)
        form.addRow("Fitting engine", cmb_engine)
        form.addRow(chk_g)
        form.addRow(chk_p)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        if s_hi.value() <= s_lo.value():
            QtWidgets.QMessageBox.warning(self, APP_NAME,
                                          "Upper bound must exceed lower bound.")
            return
        self.cfg.update(band_min=s_lo.value(), band_max=s_hi.value(),
                        min_snr=s_snr.value(), max_fwhm=s_max.value(),
                        min_fwhm=s_min.value(), despike=chk_ds.isChecked(),
                        asym_span=s_span.value(),
                        smooth_window=int(s_smooth.value()),
                        engine=cmb_engine.currentText(),
                        free_gamma=chk_g.isChecked(),
                        parallel=chk_p.isChecked())
        if self.mapfile is not None:
            self.start_fitting()

    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings", "raman_classifier_settings.json",
            "JSON (*.json)")
        if not path:
            return
        self.tab_class._pull_config()
        out = dict(self.cfg)
        out["fwhm"] = {str(k): v for k, v in self.cfg["fwhm"].items()}
        out["colors"] = {str(k): v for k, v in self.cfg["colors"].items()}
        out["tol"] = {str(k): v for k, v in self.cfg["tol"].items()}
        out["ratio_cut"] = dict(self.cfg["ratio_cut"])
        out["map_transform"] = self.cfg.get("map_transform")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        self.status("Settings saved to %s" % path)

    def load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            cfg = default_config()
            cfg.update(raw)
            cfg["fwhm"] = {int(k): {kk: float(vv) for kk, vv in v.items()}
                           for k, v in raw.get("fwhm", {}).items()} or cfg["fwhm"]
            cfg["colors"] = {int(k): dict(v) for k, v in
                             raw.get("colors", {}).items()} or cfg["colors"]
            # Settings written by version 3 carry no tolerances; fall back to
            # half-gap there, which reproduces what that file meant.
            if raw.get("tol"):
                cfg["tol"] = {int(k): {kk: float(vv) for kk, vv in v.items()}
                              for k, v in raw["tol"].items()}
            else:
                cfg["tol"] = {L: {s: abs(cfg["fwhm"][L]["ABC"]
                                         - cfg["fwhm"][L]["ABA"]) / 2.0
                                  for s in ("ABA", "ABC")} for L in cfg["fwhm"]}
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, APP_NAME,
                                           "Could not read settings:\n\n%s" % exc)
            return
        self.cfg = cfg
        if self.mapfile_raw is not None:
            self.set_transform(cfg.get("map_transform"))
        self.tab_class.spn_tol.setValue(
            float(np.median([t["ABA"] for t in cfg["tol"].values()])))
        self.tab_class.push_config()
        self.tab_map.sld_alpha.setValue(int(cfg["optical_alpha"] * 100))
        self.tab_map.cmb_cmap.setCurrentText(cfg["cmap"])
        self.status("Settings loaded from %s" % path)

    def about(self):
        QtWidgets.QMessageBox.about(
            self, "About " + APP_NAME,
            "<b>%s</b><br><br>"
            "FWHM(2D) mapping and ABA/ABC stacking classification of "
            "multilayer graphene Raman maps.<br><br>"
            "Fit: single Voigt + linear baseline over %.0f–%.0f cm⁻¹ "
            "(%s).<br>FWHM from the Olivero–Longbothum approximation.<br><br>"
            "Default thresholds after Bouhafs <i>et al.</i>, "
            "Carbon <b>177</b> (2021) 282–290, Fig. 2c. These were read off a "
            "published figure — treat them as starting values and calibrate "
            "against your own reference samples."
            % (APP_NAME, self.cfg["band_min"], self.cfg["band_max"],
               self.cfg.get("engine", "scipy")))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------

def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    # Must be set before the QApplication exists, or Retina icons come out
    # at double size.
    for attr in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        if hasattr(QtCore.Qt, attr):
            QtCore.QCoreApplication.setAttribute(getattr(QtCore.Qt, attr), True)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.exists(argv[1]):
        QtCore.QTimer.singleShot(100, lambda: win.open_raman(argv[1]))
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
