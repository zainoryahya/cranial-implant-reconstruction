# Patient-Specific Cranial Implant Reconstruction — Source Code

Pure-Python/NumPy pipeline for reconstructing patient-specific cranial implants from CT DICOM data by mirror-symmetric reconstruction, with two independent surface-generation methods (Butterfly-Optimization-Algorithm-tuned curve fitting, and global Delaunay triangulation). Companion code for:

> Yahya, Z.R., Majeed, A., Abdullah, J.Y., Rusdi, N.A., Hasan, Z.A., Wan Muhamad, W.Z.A. "Patient-Specific Cranial Implant Reconstruction from CT Imaging: A Mirror-Symmetric Approach Using Butterfly-Optimized Curve Fitting and Delaunay Surface Triangulation — A Three-Patient Pilot Study." (manuscript in preparation).

No third-party medical-imaging or mesh library is used — DICOM parsing, connected-component labelling, cuberille meshing, curve fitting, triangulation, and manifold auditing are all implemented from scratch on top of NumPy, so every step is independently auditable.

## Pipeline overview

1. **`mindicom.py`** — minimal pure-Python DICOM parser (implicit/explicit VR little endian), extracts pixel data and the header tags the pipeline needs (spacing, orientation, patient/series info).
2. **`build_skull_data.py`** — thresholds and caches the skull volume from a DICOM series into a compressed `.npz` for reuse by later stages.
3. **`extract_craniofacial_skull.py`** / **`voxelmesh.py`** — 6-connected component growing to isolate the skull, cuberille (voxel-face) surface meshing, and binary STL export.
4. **`defect_localize.py`** / **`defect_localize2.py`** / **`scan_defect.py`** — per-slice angular boundary-gap detection to locate the defect region and its corner points; `defect_localize2.py` is the robust full-resolution version (uses full connected-component enumeration rather than a nearest-seed heuristic).
5. **`symmetry_fullres.py`** — estimates the sagittal symmetry axis by mirror-overlap maximisation over defect-free slices.
6. **`build_defect_target.py`** — builds the mirrored-reference target patch for each defect slice.
7. **`boa.py`** — Butterfly Optimization Algorithm (Arora & Singh, 2019), implemented from scratch.
8. **`method_a_boa.py`** — Method A: per-slice cubic Bézier/Ball curve fitting with BOA-optimised free control points.
9. **`method_b_delaunay.py`** — Method B: single global Delaunay triangulation over the pooled boundary point cloud.
10. **`combine_skull_implant.py`** — fuses implant and skull meshes into one watertight STL.
11. **`manifold_check.py`** — quantitative edge-manifold and winding-consistency audit (topological watertightness check, independent of the volume-sign check).
12. **`render_3d_views.py`**, **`render_fig1_skull_views.py`**, **`render_final_figs.py`** — Matplotlib rendering of the figures used in the paper.
13. **`run_pipeline.py`** — end-to-end driver for the primary (Patient 1) case study.
14. **`process_dataset_d19.py`** — full pipeline applied to Patient 2 (large decompressive-craniectomy defect).
15. **`process_dataset_patient3.py`** — full pipeline applied to Patient 3 (paediatric, midline-crossing defect); also documents, in its own docstring, the diagnostic checks used to detect the midline-crossing limitation.

## Requirements

Python 3.10+, NumPy, Matplotlib. No other third-party dependencies.

```bash
pip install numpy matplotlib
```

## Usage

Each `process_dataset_*.py` / `run_pipeline.py` script is a self-contained driver: point it at a folder of DICOM files and it runs the full pipeline (segmentation → defect localisation → symmetry estimation → both reconstruction methods → mesh fusion → verification), writing STL meshes and preview PNGs to an output directory. See the docstring at the top of each script for the exact expected input layout.

```bash
python run_pipeline.py /path/to/dicom_series /path/to/output_dir
```

## Data availability

Patient CT data used to develop and validate this pipeline are **not included in this repository** for patient-privacy reasons, and no patient names, dates of birth, scan dates, or other identifiers appear anywhere in this code. Dataset paths are read from the `DICOM_DATA_DIR` environment variable (default `./data`) with generic per-patient subfolder names (`patient1_dicom_series`, `patient2_dicom_series`, `patient3_dicom_series`); point it at your own DICOM series to run the pipeline.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite the paper above (full citation to be updated once published/assigned a DOI).
