# HANDOFF — Raman classifier: graphene 1L/2L on hBN / RuCl₃

Last session: 2026-10-10. Owner: Aalok (CMU). Read this first in any new session, then open the GUI (§8).

---

## 1. Status in one paragraph
The fork of the 2D Raman map classifier GUI now loads single spectra, LabSpec `.h5` files and one-file-per-spot map folders. It has three new tabs: stacked publication spectra (4), 1L/2L layer prediction against reference datasets (5), and Lorentzian peak fitting with Si calibration, background models, automatic or manual peak count and an ω_G–ω_2D strain/doping plot (6). The ML and BL RuCl₃ maps have been analysed. Both cover the same flake: a 1L strip on hBN, plus 2L that is partly on hBN and partly on RuCl₃, where G splits into G⁻/G⁺. The user's ω_G–ω_2D figure has been reproduced by layer and substrate. **Nothing is committed.** Everything is on branch `stacked-spectra-and-folder-maps`. Next steps: get the published strain/doping constants, commit and push, then decide on extending to more layers.

## 2. Files
Repo: `~/Github/Raman_Classifier` (fork `aaloktrii/Raman_Classifier`, upstream `sagnikbajpeyi-rgb/Raman_Classifier`).

| file | what it is | status |
|---|---|---|
| `raman_map_classifier_8 (1).py` | the GUI (original tabs 1–3 + new 4 Stacked spectra, 5 Layer prediction, 6 Peak fitting; Map selector FWHM/G/2D height in tab 1; File ▸ Open map folder…) | modified, uncommitted |
| `raman_tools.py` | Si calibration, backgrounds (linear / polynomial / ALS), Lorentzian fit with fixed or auto peak count, analytic Jacobian | new, uncommitted |
| `raman_peak_fit.py` | per-dataset Lorentzian G/2D fits (auto G⁻/G⁺ split) → `fits.csv`, `summary.csv` (mean ± SD), PDFs | new, uncommitted |
| `graphene_reference.py` | build 1L/2L reference JSON; predict layer number + stacking per spectrum | new, uncommitted |
| `strain_doping.py` | ω_G–ω_2D Lee-2012 decomposition, substrate detection, peak counts, correlation plot | new, uncommitted |
| `reference/graphene_reference.json` (+ `1L_ML_hBN_RuCl3/`, `2L_BL_hBN_RuCl3/`, `reference_summary.pdf`) | the 1L/2L reference values used for prediction | generated |
| `reference/strain_doping/strain_doping_combined.{pdf,csv}` | reproduced ω_G–ω_2D figure, both maps | generated |
| `build_training_data.py`, `training_data/` | earlier copy-into-1L/2L approach, superseded by `reference/` | obsolete: delete? |
| `~/Downloads/<map or dataset>/output/` | per-dataset results: `predictions.csv`, `strain_doping.csv`, `peak_fits.csv`, `fits.csv`, PDFs | generated |
| `~/Downloads/Graphene/overview/` | early standalone plot scripts + `bilayer_spectra.pdf`, map overviews, `si_calibration.pdf` | superseded by GUI tab 4 |
| `.venv/` | PyQt5, numpy, scipy, matplotlib, h5py (no lmfit) | environment |

Data (not in repo): `~/Downloads/ML_hBN_RuCl3_532nm_map1_2__X_-2372` (192 spots, 16×12, 2 µm), `~/Downloads/BL_hBN_RuCl3_532nm_map1_7__X_-2367` (288 spots, 18×16, 2 µm), `~/Downloads/Graphene/007_01`, `007_02` (single spots, 2400 gr/mm, 4.7 cm⁻¹ res.), `~/Downloads/S2_BG_{hBN,Ru}_…_1800grating.txt` (24 cm⁻¹ res.), Si calibration `.h5` in `~/Downloads/Graphene/`.

## 3. Conventions (do not change without updating everything)
- **ML = 1L, BL = 2L** in folder names (the user's convention), but **both maps contain both** thicknesses. Reference statistics use only the "matching" spots: fitted 2D FWHM below 35 cm⁻¹ for 1L, 42–65 cm⁻¹ for 2L. Whole-map statistics are stored but do not separate 1L from 2L.
- Per-spot map files: X/Y come from `__X_<x>__Y_<y>__` in the filename. The loader stores raw0 = −X so the GUI's instrument default (X = −col0) reproduces the filename coordinates.
- Every spectrum is Si-calibrated to **520.7 cm⁻¹** before G/2D positions are used (tab 6 checkbox, on by default; always on in `strain_doping.py`).
- Substrate = **RuCl₃** when the 164 cm⁻¹ mode is more than 10× the noise, otherwise **hBN**.
- Split G (2L on RuCl₃) is fitted as G⁻ held near 1582 ± 10 and G⁺ near 1600 ± 10 cm⁻¹ (FWHM ≤ 40). Correlation points pair by order: (G⁻, 2D⁻), (G⁺, 2D⁺), with 2D forced to 2 Lorentzians for these spots.
- Doping sign: negative n = holes, in units of 10¹³ cm⁻².
- Figures: Arial, vector PDF, 8.7 cm wide; overlays/labels off by default (see the user's figure-style memory).
- Default fit windows: G 1500–1680 cm⁻¹, 2D 2560–2810 cm⁻¹ (stops below the ~2822 cm⁻¹ instrument spike in the 007 files).

## 4. Key formulas / methods
- Lorentzian: h·(w/2)² / ((x−x₀)² + (w/2)²) on a linear baseline; area = π·h·w/2; errors from the covariance scaled by reduced χ².
- Automatic peak count (`raman_tools.fit_peaks`): add a peak only if BIC drops by ≥ 10, the new peak is ≥ 3× noise, every peak carries ≥ 5% of the band area, neighbours are ≥ 0.6 × their mean FWHM apart, and no parameter sits on its bound. A band needs its strongest peak ≥ 5× noise to count at all, otherwise n = 0. Each step tries two starting layouts.
- Layer prediction (`graphene_reference.predict_row`): z-distance to each reference on 2D FWHM and ln I(2D)/I(G).
  - FWHM above 2L mean + 3 SD: ≥3L, with ABA/ABC from the Bouhafs 2021 width table.
  - FWHM between the 1L and 2L 3-SD limits: "2L twisted?".
  - 1L-like but 2D blue-shifted beyond 3 SD: "1L or twisted 2L".
  - Further than 4 SD from both references: unassigned.
- Strain/doping (Lee et al., Nat. Commun. 3, 1024 (2012)): Δω = ε·(a_G, s_ε·a_G) + n·(b_G, s_n·b_G), solved as a 2×2 linear system per point.
- Background options: straight line through the window ends; iterative modified polyfit (Lieber 2003); asymmetric least squares (Eilers 2005, λ, p).

## 5. Key numbers
| quantity | value |
|---|---|
| 1L reference (ML matching, n = 30) | 2D FWHM 24.2 ± 4.8; I2D/IG 3.6 ± 1.7; ω_2D 2685 ± 5; ω_G 1584 ± 0.7 cm⁻¹ (single G) |
| 2L reference (BL matching, n = 104) | 2D FWHM 57.6 ± 4.7; I2D/IG 0.86 ± 0.4; ω_2D 2698 ± 4 cm⁻¹; G⁻ 1585 ± 1.1, G⁺ 1603 ± 1.3 when split |
| Si peak per map | ML 520.63 ± 0.07, BL 520.89 ± 0.06 cm⁻¹ (0.26 offset); instrument Si file 520.64 |
| Strain/doping PARAMS | G0 1578.3, D0 2683.6 (**read off user's figure**); a_G −69.1 cm⁻¹/% and slope 2.2 (Lee 2012); b_G −14.1 cm⁻¹ per 10¹³ (**figure estimate**); doping slope 0.70 |
| Correlation means (both maps) | 2L/RuCl₃ (G⁻,2D⁻): ε −0.058 ± 0.057 %, n −0.17 ± 0.31; (G⁺,2D⁺): ε −0.24 ± 0.15 %, n −0.68 ± 1.07; 2L/hBN: ε −0.11 %, n +0.21; 1L/hBN: ε +0.01 %, n −0.43 (×10¹³ cm⁻²) |
| Peak counts, most common (G, 2D) | 1L/hBN (1,1) 36/49; 2L/hBN (1,4) 53/97; 2L/RuCl₃ (2,2) 60/122 or (2,1) 38/122 |
| 007 predictions | 4× 1L (007_01/4, 007_02/12, 1_r3, 1_r4); 8× 2L AB; 007_01/6_r2 ≥3L; 007_02/1_r1 called 2L but 2D 68 cm⁻¹ at 2713 (check) |
| Peak-count validation | 40/40 synthetic noisy spectra correct; 0.18 s per band |

## 6. Conclusions to carry forward
1. Both RuCl₃ maps show the same flake: 1L strip on hBN at the top, 2L below, partly on hBN and partly on RuCl₃.
2. G splitting into G⁻ ≈ 1585 / G⁺ ≈ 1603 cm⁻¹ occurs only for 2L in contact with RuCl₃ (58 of 62 split spots show the RuCl₃ mode), consistent with layer-asymmetric hole doping; 1L does not split.
3. The (G⁺, 2D⁺) layer carries the strain and doping; (G⁻, 2D⁻) is near neutral at about −0.06% strain.
4. The ML/BL folder names cannot serve as per-spot labels; labels must come from the spectra.
5. The original classifier (Bouhafs table) only covers 3–9 layers and mislabels these bilayers as 3L in tab 2.
6. Without Si calibration, the 0.26 cm⁻¹ map-to-map offset would appear as a false strain/doping difference.

## 7. Open items (next session)
- [ ] Get the published ω_G0, ω_2D0 and doping rate from the user's paper and set `strain_doping.PARAMS`; the current values are read off the figure.
- [ ] Commit the branch and push to the fork (user has not yet approved).
- [ ] Delete `build_training_data.py` + `training_data/` (superseded) if the user agrees.
- [ ] Check 007_02/1_r1 (broad, blue-shifted 2D) and the ~10 "1L on RuCl₃" spots (likely boundary spots).
- [ ] Predict the user's real new dataset: `graphene_reference.py predict <folder>` or tab 5. There is no `new_dataset` folder; that name was a placeholder.
- [ ] Extend to more layers. Needs labelled reference data per N and substrate, and for N ≥ 4 low-frequency shear (C) mode data (below ~50 cm⁻¹, needs a notch or BragGrate filter). Planned: a probabilistic classifier per reference cluster, plus 2D asymmetry for ABA/ABC.
- [ ] Optional suggestions for upstream: rename the script, add `requirements.txt` / `.gitignore` / README, add a 1L/2L gate before the Bouhafs classification, add a despike option for single spectra.

## 8. How to update
1. GUI: `cd ~/Github/Raman_Classifier && .venv/bin/python "raman_map_classifier_8 (1).py"`. A running instance must be restarted to pick up code changes.
2. Rebuild references: `.venv/bin/python graphene_reference.py build --ref 1L ~/Downloads/ML_hBN_RuCl3_532nm_map1_2__X_-2372 --ref 2L ~/Downloads/BL_hBN_RuCl3_532nm_map1_7__X_-2367`
3. Predict a dataset: `.venv/bin/python graphene_reference.py predict <folder>` → `<folder>/output/predictions.csv`, `prediction.pdf`
4. Peak statistics: `.venv/bin/python raman_peak_fit.py <folder> [--ng auto|1|2] [--n2d N]`
5. Strain/doping + peak counts: `.venv/bin/python strain_doping.py <folder> [<folder> …] --out reference/strain_doping` (parallel; about 1 min for 288 spots)
6. Parallel tests must run from a script file, not stdin: spawn workers re-import `__main__`.

## 9. Session log
- 2026-10-06 → 10: handoff created. Covers the fork setup, tabs 4–6, the references, Si calibration, substrate detection, peak counting, and the reproduced ω_G–ω_2D figure.
