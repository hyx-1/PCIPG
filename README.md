Code for protein complex method PCIPG. Human and Saccharomyces cerevisiae PPI datasets for PCIPG are in https://zenodo.org/records/18254107.
# PCIPG (SAGE-GAT) — Unsupervised Protein Complex Discovery

This repository provides an end-to-end **unsupervised GNN pipeline** for protein complex discovery. It constructs a **PPI graph** augmented with **structure-informed residue/contact features**, trains a **GraphSAGE/GAT-style model** with an **unsupervised edge/non-edge objective** to learn protein embeddings / interaction scores, generates **candidate protein complexes**, and evaluates them against a **gold-standard complex set**.

## Layout
- `main.sh`: end-to-end pipeline (preprocess → train → test → evaluate)  
- `environment.yml`: conda environment  
- `Data_Process.py`: preprocessing (ID mapping, feature export, PPI edges)  
- `train.py`: unsupervised training (checkpoints + logs)  
- `test.py`: inference and candidate complex generation (top-k + merged results)  
- `Select_eva.py`: post-filtering and evaluation (Precision/Recall/F1/ACC)  
- `sage_gat.py`, `unsupvise_loss.py`, `evaluation.py`, `utils.py`: model/loss/metrics/utilities  
- Generated folders: `data/`, `models/`, `result/`, `logs/`

## Installation
```bash
cd code
conda env create -f environment.yml
conda activate PCIPG
```

## Installation
```bash
bash main.sh
```
