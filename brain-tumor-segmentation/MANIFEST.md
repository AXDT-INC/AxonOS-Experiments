# BraTS 2023 GLI — Reproducibility Package Manifest

This manifest lists every file included in this reproducibility package with its
SHA-256 checksum. It accompanies `README.md`.

## Verification usage

From the **root of this package** (the `reproducibility/` directory):

```bash
sha256sum -c MANIFEST.sha256
```

All checksums below use the paths relative to the package root.

## Manifest

| File | SHA-256 |
|---|---|
| `config/config.yaml` | `d3ca1f4714d4b71f209d698a526bbbc1225379e9cfd66ff7cb530d304d331dac` |
| `config/config-50epoch.yaml` | `c11f818ed2053089c7f931c6200e5c214887ee12343805a6d9d65d4734d7504c` |
| `data/manifest.json` | `47939b549259cb4e9743d45ff19f171594f01bf4ade6addc4ce9498c17707cd9` |
| `data/manifest-subject-ids.csv` | `2fa154f849458360ca21d526de41e3ada3d311eae846d5751742b0a33e09a15b` |
| `data/splits/split_ids.json` | `4a03d2bc14d403e3e0be29a1fdd24d9c9849df5c35b7ca69658ed557cc5038fc` |
| `data/_download_report.json` | `ea49061517470ef9a6acadd569d0e9d1a6e01fc1beef291697a6db4443cb7209` |
| `environment.lock.txt` | `241168b1e3603cea3b0f088bdb5d6d0ec44c4eecc29eaa18e0be32633f90a4ed` |
| `metrics/train_metrics.csv` | `eda03d53f3003974859f3108876beb7c059443833e85008fa3918729c81de1ee` |
| `metrics/final_metrics.json` | `52de6d77e3f7ba11c1cd9d72da91a19fd6f14b614f954013e3d741b0375a3c9d` |
| `metrics/eval_metrics.json` | `8f74dfdb15aa3f022afffbc885080cfc6015391aad8c2dbf4942efdc438cbe95` |
| `results/figures/learning_curve.png` | `9cdcf35fa773bae1228129d75e0c3210d9a96193c358d5a2f2ddd299b326e1d9` |
| `results/figures/per_class_dice.png` | `b0179d93703489ee1da8ef8a7b5427620c5605e0ea629433ad716468212b7198` |
| `results/figures/gpu_telemetry.png` | `da8e26f0dd28a31bf259d1a280904043e2ef7fdec2d48aa82704d72d501bc3cc` |
| `results/visualizations/subject-BraTS-GLI-00046-000.png` | `c4ad5b1e1dd1e198924e67860cb3151086bbfc71799d582d8ed9762ab3ddd2e9` |
| `results/visualizations/subject-BraTS-GLI-00186-000.png` | `45aa58435ff197a16dcdc103f33b71a02d157b953f43098c7009a1c156ffc159` |
| `results/visualizations/subject-BraTS-GLI-00242-000.png` | `f72f5e3702dd89d3aebf7cc733d0d32afa126e8c47ab8ce303cf9fe7b4fc68c1` |
| `results/visualizations/subject-BraTS-GLI-00267-000.png` | `be6d25d5a5ae774cf4b2c0ab57e5dff5bf5e6c762788aaef8ff0283a66441695` |
| `results/visualizations/subject-BraTS-GLI-00343-000.png` | `77fa331c258a6e748692d06655f4d6d198758615775d768c50f37209b7a57a60` |
| `results/visualizations/viz_manifest.json` | `a9697aab3867f8918d6d454e375f4b406b7893b4b12b5fb29e9e6697f4a3ca3b` |
| `scripts/common.py` | `1a62a8424d6a925a7e51a5e9c5166d09af068b685222fa2fde15d8b774bee09d` |
| `scripts/train.py` | `ddfe6f641470c35b0c323275f68b68b359fecb05e13917bb7927b7852cb5822e` |
| `scripts/evaluate.py` | `c1d8b8a5f8677baf0dd1cdd0acc86b5d1fbed47cd8c29c6b98daed238af9ef72` |
| `scripts/visualize.py` | `234f0d00ac08d26db1faef5b9facbfc2808b4dc48ec51e6bab964d04b620eeb6` |
| `scripts/download_braTS2023_gli.py` | `41fd903e1b260b41bda46501f6dcf12b004dd6952198af1e72aab201d690bc0a` |
| `scripts/make_synthetic_dataset.py` | `a59d8831f91c998cefbecc58c85f137715a5441dd9739abf038c3bfeb2694dcd` |
| `scripts/smoke_test.py` | `538e600f7051573dc72f31d68ed1fb234135dea8d6defc9d6b35a80aec0e999b` |
| `scripts/make_figures.py` | `e8ff688b083ce4b88b26b1a97a1c4ef25d3bedda5dfee8f601999b8a9a2f3258` |
| `telemetry/gpu0.csv` | `8c2bd11c4e3aaef059d52e4b13b895b6955b5667bc1f5f510d99e2f06feb7edd` |
| `README.md` | See `README.md` (documentation; not checksummed to allow editorial changes) |
| `MANIFEST.md` | This file (self-referential; not checksummed) |

## Checksums of key external references

The following verify the provenance of the split and dataset subset. These are
recorded in the original artifacts and reproduced here for reference:

- Dataset `subset_id_hash_sha256` (in `data/manifest.json`):
  `779269e8b528040ec69b3098098e54b9651d3e164df3c638f850dcf760d18ed4`
- Download integrity (`data/_download_report.json`): 300/300 subjects,
  1500/1500 files, `3,094,014,357` bytes, 0 failures.

## What is intentionally NOT included

- **Raw MRI dataset** (~2.88 GB BraTS 2023 GLI NIfTI files). See `README.md`
  for acquisition instructions and `data/manifest.json` for the exact source
  URLs per file.
- **Model checkpoints** (`.pt` files). These are large binary artifacts; the
  recorded metrics and figures are the scientific result and are included.
- **Secrets, credentials, tokens, private paths, wallet information, IP
  addresses.** None are present in this package.
- **Caches or environment-specific junk.**

## Notes

- The `README.md` file in this package is documentation; it is intentionally
  excluded from checksum verification because editorial changes to
  documentation should not require re-hashing.
- `metrics/eval_metrics.json` originally recorded an absolute local
  filesystem path in its `checkpoint` field (from the original execution host).
  For public distribution this was sanitized to the relative path
  `outputs/50epoch/checkpoints/best.pt`. The original, unmodified artifact is
  preserved at `outputs/50epoch/metrics/eval_metrics.json` in the source
  project. All scientific values in the file are unchanged.
