# RBF-Comparisons

**Normal-Based Mesh Comparison Toolkit for Digital Human Model Morphing**

This repository provides a Python-based framework for comparing and validating morphed anatomical surface meshes (e.g., 50th percentile to seated or scaled morphs).  
It is optimized for evaluating geometric fidelity using **normal-projected raycasting** between reference and target meshes.

---

## 🔍 Overview

The toolkit computes **normal-based surface distances** between two meshes (e.g., 50th Male → Morphed model) using **radial basis function (RBF)** deformation validation logic.

Each cell (triangle) of the **source mesh** emits a ray along its surface normal. The intersection point on the **target mesh** is measured to compute the **hit distance**.  
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

