# Model & Asset Licenses

This repository ships **code, configs, and generated reports/metrics** only.

- **Base / Control / Perceptual / Identity Models**  
  Referenced at runtime but **not redistributed** in this repo.  
  Users must accept the original licenses and download from the official sources.

- **Style Reference Images**  
  User-provided examples only. Do not redistribute artworks without permission.  
  Any images included in this repo are for demonstration under fair-use/educational context.

- **Generated Outputs**  
  Images and metric files under `out/` and `report/` are produced by running the pipeline.  
  You may redistribute your own outputs; for third-party content, follow original rights.

- **Third-Party Python Packages**  
  Governed by their own licenses (see `requirements/`).  
  Notably, we use `opencv-contrib-python` instead of `opencv-python` to avoid conflicts.

If you are integrating additional models or assets, please add their names, source URLs,
and licenses here (or link to their license files).
