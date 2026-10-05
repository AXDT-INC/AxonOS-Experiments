# AxonOS Experiments

Reproducible scientific and AI workloads executed on AxonOS GPU infrastructure.

## Experiments

### Brain Tumor MRI Segmentation

A reproducible 3D glioma segmentation workload using the BraTS 2023 GLI dataset,
PyTorch, MONAI, and an NVIDIA Tesla V100-SXM2-32GB.

- 300 subjects
- 240 training / 60 validation
- 50-epoch follow-up experiment
- Mean foreground Dice: 0.7051
- GPU telemetry and recorded training metrics included
- Dataset itself is not redistributed

See:

[`brain-tumor-segmentation/`](brain-tumor-segmentation/)

### CellModeller Multi-GPU

A work-in-progress distributed CellModeller experiment on AxonDAO-hosted AxonOS.

- One- and two-GPU headless runs both reached and held 100,000 cells
- Two-GPU recorded runtime was 9.51× longer; no speedup is claimed
- Raw logs, telemetry, analysis scripts, figures and provenance limitations included
- Four-GPU GUI capacity estimate documented separately from achieved results

See:
[`cellmodeller-multigpu/`](cellmodeller-multigpu/)

---

Additional AxonOS experiments will be added to this repository.
