# RBF-Comparisons

**Normal-Based Mesh Comparison Toolkit for Digital Human Model Morphing**

This repository provides a Python-based framework for comparing and validating morphed anatomical surface meshes (e.g., 50th percentile vs. seated or scaled morphs).  
It is optimized for evaluating geometric fidelity using **normal-projected raycasting** between reference and target meshes.

---

## 🔍 Overview

The toolkit computes **normal-based surface distances** between two meshes (e.g., 50th Male → Morphed model) using **radial basis function (RBF)** deformation validation logic.

Each cell (triangle) of the **source mesh** emits a ray along its surface normal.  
The intersection point on the **target mesh** is measured to compute the **hit distance**.  
This produces both **absolute** and **signed** distance fields, allowing visualization and quantitative analysis of morphing accuracy.

<p align="center">
  <img src="docs/morph_concept.png" width="600" alt="Normal-based morphing concept diagram">
</p>

### Key Features
- ⚙️ **Batch raycasting** across thousands of triangles  
- 🧵 **Parallel execution** (multiprocessing, Windows-safe)  
- 💾 **Caching** with `.npz` results for fast restyling/re-plotting  
- 🧭 **Directional control** — cast from reference or morphed surface (`--cast-from morph`)  
- 🎨 **ParaView-ready outputs**:  
  - Glyphs for normal vectors  
  - Colored morph surfaces (per-cell distance)  
  - CSV summaries for further analysis  
- 📊 **Optional global distance fields** and contour generation  

---

## 🧩 Repository Structure

```
RBF-Comparisons/
├── rbf_cli.py                # Main entry point (CLI)
├── rbf_runner.py             # Orchestrates batch modes and exports
├── rbf_args.py               # CLI argument parser
├── rbf_batch.py              # Sequential + parallel raycasting
├── rbf_raycast.py            # Single-ray intersection logic
├── rbf_normals.py            # Surface normal and glyph utilities
├── rbf_distance.py           # Distance coloring on meshes
├── rbf_io.py                 # Load/save helpers for VTK/VTU/VTP
├── rbf_logging.py            # Logging setup
├── Meshes/
│   ├── Input/                # Reference and target meshes (not committed)
│   ├── output_*.vtp          # Generated ParaView meshes (ignored by Git)
│   └── output_*.npz          # Cached data (ignored by Git)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone and enter repository
```bash
git clone https://github.com/<your-org>/RBF-Comparisons.git
cd RBF-Comparisons
```

### 2. Create and activate environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can install manually:
```bash
pip install pyvista numpy
```

---

## 🚀 How to Run

Below are the most common example workflows.

### 🔹 Compare Morph vs. Reference (cast from morph)

Use the **morphed** surface as the source mesh (typically denser), and project its normals toward the reference (e.g., 50th percentile).

```bash
python rbf_cli.py \
  --ref Meshes/Input/50th_Male.vtu \
  --target Meshes/Input/Skin_morphed.vtu \
  --cast-from morph \
  --multi-auto-percent 0.8 \
  --multi-auto-mode random \
  --parallel 8 \
  --chunksize 128 \
  --progress-every 50 \
  --export-multi-normals-glyph Meshes/output_multi_normals_on_morph.vtp \
  --export-batch-hit-morph Meshes/output_morph_0p8pct_by_distance.vtp \
  --export-distances-csv Meshes/output_multi_normals_results.csv \
  --save-batch Meshes/output_batch_0p8pct_cache.npz \
  --ray-len-factor 3.0 \
  --verbose
```

### 🔹 Compare Reference vs. Morph (cast from 50th)

If you prefer to emit rays from the reference model instead:

```bash
python rbf_cli.py \
  --ref Meshes/Input/50th_Male.vtu \
  --target Meshes/Input/Skin_morphed.vtu \
  --cast-from ref \
  --multi-auto-percent 0.8 \
  --multi-auto-mode random \
  --parallel 8 \
  --chunksize 128 \
  --progress-every 50 \
  --export-multi-normals-glyph Meshes/output_multi_normals_on_50th.vtp \
  --export-batch-hit-morph Meshes/output_morph_0p8pct_by_distance.vtp \
  --export-distances-csv Meshes/output_multi_normals_results.csv \
  --save-batch Meshes/output_batch_0p8pct_cache.npz \
  --ray-len-factor 3.0 \
  --verbose
```

---

## ⚙️ Key Arguments

| Argument | Description |
|-----------|-------------|
| `--ref` | Path to reference mesh (e.g., 50th percentile) |
| `--target` | Path to morphed or seated mesh |
| `--cast-from morph` | Use morphed mesh as the source (denser, emits rays) |
| `--multi-auto-percent` | Select percentage of source triangles for batch raycasting |
| `--parallel N` | Use N processes for faster execution |
| `--save-batch` / `--load-batch` | Save or load cached `.npz` results |
| `--export-batch-hit-morph` | Write VTP mesh colored by distance |
| `--export-multi-normals-glyph` | Write glyph arrows for normal visualization |
| `--export-distances-csv` | Export all results in CSV format for analysis |
| `--progress-every` | Frequency (in iterations) of progress log updates |
| `--ray-len-factor` | Ray length multiplier (relative to bounding box diagonal) |

---

## 🎨 Visualizing in ParaView

Once the run finishes, you’ll get one or more `.vtp` output files.

### 1️⃣ View Morph Distances
- Open `Meshes/output_morph_*.vtp` in ParaView  
- In **Properties → Coloring**, choose `batch_hit_distance` (Cell Data)
- Click **Rescale to Data Range**
- Apply a **Diverging Colormap (e.g., CoolWarm)**  
- Optionally, **invert** colors so **blue = good (small distance)**, **red = bad (large distance)**  

### 2️⃣ View Normals
- Open `Meshes/output_multi_normals_on_*.vtp`
- Set the color array to `hit_distance`
- Adjust **Glyph Scale Factor** to make the arrows visible

---

## 💾 Cached Results Workflow

Large morph comparisons can take time — so results are automatically cacheable.

To skip recomputation and recolor from cache:
```bash
python rbf_cli.py \
  --ref Meshes/Input/50th_Male.vtu \
  --target Meshes/Input/Skin_morphed.vtu \
  --load-batch Meshes/output_batch_0p8pct_cache.npz \
  --recolor-batch-hit-morph Meshes/output_morph_recolor.vtp \
  --recolor-use-signed
```

This instantly regenerates the colored morph file using previously computed distances.

---

## 🧠 Methodology Summary

For each triangle \( T_i \) in the **source mesh** with centroid \( \mathbf{c}_i \) and unit normal \( \mathbf{n}_i \),  
a ray is cast along \( \pm \mathbf{n}_i \) to intersect the **target surface** \( S' \).  
The **signed hit distance** is computed as:

\[
d_i = \operatorname{sign}(\mathbf{n}_i \cdot (\mathbf{p}_i - \mathbf{c}_i)) \, \|\mathbf{p}_i - \mathbf{c}_i\|
\]

where \( \mathbf{p}_i \) is the nearest intersection point.  

Distances are aggregated and visualized per-cell, highlighting geometric deviations between meshes.  
This enables quantitative verification of morphing accuracy across posture, percentile, or sex-specific models.

---

## 🧩 Typical Applications

- Validation of **RBF-based digital human morphing**
- Quantification of **geometric deformation** across posture changes
- Comparison of **population-scale models** (50th → 95th percentile)
- Mesh-based **fidelity and penetration** analysis for FE simulation setup

---

## 🧰 Troubleshooting

| Problem | Cause | Solution |
|----------|--------|----------|
| `IndexError: cell_index ... out of range` | Source mesh has fewer triangles than selected indices | Use `--cast-from morph` if the morph is denser |
| No colors visible in ParaView | Data array not selected | Choose `batch_hit_distance` under *Cell Data* |
| Script appears idle | Parallel workers are computing | Use `--progress-every 50` for periodic updates |
| File overwriting | Old results not renamed | Add `output_` prefix and use `.gitignore` for safety |

---

## 🧾 License

MIT License © 2025 Pennsylvania State University – Computational Biomechanics Lab

---

## 👥 Contributors

- **Dr. Reuben Kraft** – Penn State University  
- **Dr. Jingwen Hu** – University of Michigan Transportation Research Institute (UMTRI)  
- **AI-assisted engineering tools by ChatGPT (OpenAI)**

---

## 📘 Citation

If you use this code in academic or technical work, please cite:

> Kraft, R. H., Hu, J., *et al.* (2025).  
> *Digital Twin Simulation of Pilot Spinal Degeneration Using Seat-Embedded Acceleration and Personalized Anatomical Models.*  
> AFOSR Proposal / Penn State Computational Biomechanics Lab.
