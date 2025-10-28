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
- 💾 **Caching** with `.npz` results for fast re-plotting (no recomputation)  
- 🧭 **Directional control** — cast from reference or morphed surface (`--cast-from morph`)  
- 🎨 **ParaView-ready outputs**  
  - Glyphs for normal vectors  
  - Colored morph surfaces (per-cell distance)  
  - CSV summaries for further analysis  
- 📊 **Post-processing utilities** for histograms, CDFs, and boxplots  
- 🪄 **Customizable visualization** — background, color map, legend title, and scalar bar styling

---

## 🧩 Repository Structure

```
RBF-Comparisons/
├── rbf_cli.py                # Main entry point (computation + export)
├── rbf_runner.py             # Orchestrates computation, caching, and export
├── rbf_args.py               # CLI argument parser
├── rbf_batch.py              # Sequential + parallel batch raycasting
├── rbf_raycast.py            # Single-ray intersection logic
├── rbf_normals.py            # Surface normal & glyph utilities
├── rbf_distance.py           # Distance-to-color mapping
├── rbf_io.py                 # Load/save helpers (VTU/VTP/NPZ)
├── rbf_logging.py            # Logging and console formatting
├── rbf_post_cli.py           # Post-processing CLI (plots + visualization)
├── rbf_post.py               # Histogram, CDF, and boxplot generation
├── Meshes/
│   ├── Input/                # Reference and target meshes (not committed)
│   ├── output_*.vtp          # ParaView visualization meshes (ignored by Git)
│   ├── output_*.npz          # Cached batch results (ignored by Git)
│   └── output_*.png          # Plots and screenshots (ignored by Git)
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

If missing, install manually:
```bash
pip install pyvista numpy matplotlib meshio
```

---

## 🚀 How to Run

### 🔹 Compare Morph vs. Reference (cast from morph)

Use the **morphed surface** as the source mesh (typically denser), and project its normals toward the reference (e.g., 50th percentile):

```bash
python rbf_cli.py ^
  --ref Meshes/Input/50th_Male.vtu ^
  --target Meshes/Input/Skin_morphed.vtu ^
  --cast-from morph ^
  --multi-auto-percent 0.8 ^
  --multi-auto-mode random ^
  --parallel 8 ^
  --chunksize 128 ^
  --progress-every 50 ^
  --export-multi-normals-glyph Meshes/output_multi_normals_on_morph.vtp ^
  --export-batch-hit-morph Meshes/output_morph_0p8pct_by_distance.vtp ^
  --export-distances-csv Meshes/output_multi_normals_results.csv ^
  --save-batch Meshes/output_batch_0p8pct_cache.npz ^
  --ray-len-factor 3.0 ^
  --verbose
```

### 🔹 Compare Reference vs. Morph (cast from 50th)

```bash
python rbf_cli.py ^
  --ref Meshes/Input/50th_Male.vtu ^
  --target Meshes/Input/Skin_morphed.vtu ^
  --cast-from ref ^
  --multi-auto-percent 0.8 ^
  --multi-auto-mode random ^
  --parallel 8 ^
  --chunksize 128 ^
  --progress-every 50 ^
  --export-multi-normals-glyph Meshes/output_multi_normals_on_50th.vtp ^
  --export-batch-hit-morph Meshes/output_morph_0p8pct_by_distance.vtp ^
  --export-distances-csv Meshes/output_multi_normals_results.csv ^
  --save-batch Meshes/output_batch_0p8pct_cache.npz ^
  --ray-len-factor 3.0 ^
  --verbose
```

---

## ⚙️ Key Arguments

| Argument | Description |
|-----------|-------------|
| `--ref` | Path to reference mesh (e.g., 50th percentile) |
| `--target` | Path to morphed or seated mesh |
| `--cast-from morph` | Use morphed mesh as the source (denser, emits rays) |
| `--multi-auto-percent` | Percent of source triangles to cast normals from |
| `--parallel N` | Number of CPU processes for parallel execution |
| `--save-batch` / `--load-batch` | Save or reuse cached `.npz` batch data |
| `--export-batch-hit-morph` | Output morph surface colored by hit distances |
| `--export-multi-normals-glyph` | Export glyph arrows for visualization |
| `--export-distances-csv` | Save per-triangle hit results for analysis |
| `--progress-every` | Log update frequency (in rays processed) |
| `--ray-len-factor` | Ray length multiplier (relative to mesh diagonal) |

---

## 💾 Cached Results & Post-Processing

Once you have a cached `.npz` file, you can generate visualizations and plots **without recomputing distances**.

### 🔹 Recolor morph directly from cache

```bash
python rbf_cli.py ^
  --ref Meshes/Input/50th_Male.vtu ^
  --target Meshes/Input/Skin_morphed.vtu ^
  --load-batch Meshes/output_batch_0p8pct_cache.npz ^
  --recolor-batch-hit-morph Meshes/output_morph_recolor.vtp ^
  --recolor-use-signed
```

---

### 🔹 Create plots and styled visualization from cache

```bash
python rbf_post_cli.py ^
  --ref Meshes/Input/50th_Male.vtu ^
  --target Meshes/Input/Skin_morphed.vtu ^
  --load-batch Meshes/output_batch_0p8pct_cache.npz ^
  --cast-from auto ^
  --use-signed ^
  --export-morph-colored Meshes/output_morph_signed_from_cache.vtp ^
  --screenshot Meshes/output_morph_signed_from_cache.png ^
  --bg black ^
  --cmap coolwarm ^
  --scalarbar-title "Signed Distance (mm)" ^
  --plots-prefix Meshes/output_signed ^
  --bins 60 ^
  --hist-xmin -50 --hist-xmax 50 ^
  --cdf-xmin -50 --cdf-xmax 50 ^
  --verbose
```

**Outputs generated:**
- `output_morph_signed_from_cache.vtp` → colored mesh for ParaView  
- `output_morph_signed_from_cache.png` → screenshot with colorbar  
- `output_signed_hist.png` → histogram of distances  
- `output_signed_cdf.png` → cumulative distribution function (CDF)  
- `output_signed_box.png` → boxplot of distance distribution  

---

## 📊 Plot Interpretations

| Plot | Description |
|------|--------------|
| **Histogram** | Shows how many surface triangles fall within each distance bin — reveals spread of morphing error (in mm). |
| **CDF (Cumulative Distribution Function)** | Displays the fraction of triangles below a given distance threshold; a steeper curve indicates higher morph accuracy. |
| **Boxplot** | Summarizes the distance distribution (median, quartiles, whiskers, outliers). Useful for identifying bias and extreme deviations. |

All distances are expressed in **millimeters (mm)**.

---

## 🎨 ParaView Visualization

1️⃣ Open the colored mesh (`output_morph_*.vtp`)  
2️⃣ In **Properties → Coloring**, choose `batch_hit_distance` (Cell Data)  
3️⃣ **Rescale to Data Range**  
4️⃣ Choose a **Diverging colormap (CoolWarm)**  
5️⃣ Optionally **invert colors** → blue = good (small deviation), red = bad (large deviation)  

---

## 🧠 Methodology Summary

For each triangle \( T_i \) in the **source mesh** with centroid \( \mathbf{c}_i \) and unit normal \( \mathbf{n}_i \),  
a ray is cast along \( \pm \mathbf{n}_i \) to intersect the **target surface** \( S' \).  
The **signed hit distance** is computed as:

\[
d_i = \operatorname{sign}ig(\mathbf{n}_i \cdot (\mathbf{p}_i - \mathbf{c}_i)ig) \, \|\mathbf{p}_i - \mathbf{c}_i\|
\]

where \( \mathbf{p}_i \) is the nearest intersection point.

Distances are aggregated per triangle, highlighting geometric deviations between meshes.  
This enables quantitative verification of morphing accuracy across posture, percentile, or sex-specific models.

---

## 🧩 Typical Applications

- Validation of **RBF-based digital human morphing**
- Quantification of **geometric deformation** between postures
- Comparison of **population-scale body models** (50th → 95th)
- **Mesh penetration / fidelity** checks for finite-element simulation setup

---

## 🧰 Troubleshooting

| Problem | Cause | Solution |
|----------|--------|----------|
| `IndexError: cell_index ... out of range` | Source mesh has fewer triangles than selected indices | Switch `--cast-from morph` if morph mesh is denser |
| No colors visible in ParaView | Wrong data array selected | Choose `batch_hit_distance` under *Cell Data* |
| Script appears idle | Parallel workers are processing | Use `--progress-every 50` for periodic updates |
| Missing `meshio` import error | PyVista export helper not found | Run `pip install meshio` |
| Too wide histogram range | Default auto-scaling to 99.5th percentile | Use `--hist-xmin`, `--hist-xmax`, or `--hist-percentile` |
