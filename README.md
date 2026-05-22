# DRAMA

PyTorch implementation of the KDD 2026 paper **“Drift-Aware Memory-Augmented Spatio-Temporal Graph Attention for Industrial Anomaly Detection.”**

This repository provides the official implementation of DRAMA for anomaly detection on ICS and telemetry benchmarks, including **SWaT**, **WADI**, **SMAP**, and **MSL**. DRAMA combines reconstruction-based detection, memory-based temporal retrieval, topology-guided spatial attention, and drift-aware spatio-temporal fusion.

## Quickstart

```bash
pip install -r requirements.txt
python preprocessing.py --dataset <swat|wadi|smap|msl>
python train.py --dataset <swat|wadi|smap|msl> --tag <tag>
python test.py --dataset <swat|wadi|smap|msl> --tag <tag>
```

## Data Layout

Raw datasets are not included in this repository. Place the downloaded files as follows:

```text
data/<swat|wadi>/train.csv
data/<swat|wadi>/test.csv

data/NASA/train/*.npy
data/NASA/test/*.npy
data/NASA/labeled_anomalies.csv
```

Preprocessed SWaT/WADI files are written to `data/<dataset>/preprocessed/`. SMAP/MSL files are written as entity-level arrays under `data/smap/` and `data/msl/`.

## Training

```bash
python train.py --dataset <swat|wadi|smap|msl> --tag drama
```

Resume from an existing checkpoint:

```bash
python train.py --dataset swat --tag drama --pretrained
```

Checkpoints are saved under `models/<dataset>/`.

## Evaluation

```bash
python test.py --dataset swat --tag drama
```

Useful options:

```bash
python test.py --dataset swat --tag drama --pretested
python test.py --dataset swat --tag drama --search_metric <point-adjusted|point-wise|composite>
```

Results and cached scores are saved under `results/<dataset>/`.
